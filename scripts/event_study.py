#!/usr/bin/env python3
"""
事件研究（Event Study）— 直接檢驗產品主張「預熱🌡 = 早期進場訊號」

方法：
    1. 用「逐日擴張窗口」重建每個題材的歷史相位
       （每天只用當天為止的資料算均值/標準差，與系統上線時的運作完全一致，
        絕不偷看未來資料，避免 lookahead bias）。
    2. 找出每一次「進入預熱」的事件（前一天非預熱 → 當天預熱）。
    3. 統計事件後 1~20 天 SMI（類股等權）的累積報酬。
    4. 與「隨機任一天進場」的基準報酬比較，做 Welch t 檢定。

判讀：
    若預熱事件後的報酬「顯著高於」隨機基準 → 預熱訊號有早期進場價值 ✅
    若與隨機無異或更差 → 預熱訊號不具進場參考價值 ❌

⚠️ 樣本有限，事件不完全獨立，本檢定為初步驗證。不構成投資建議。

用法：
    python scripts/event_study.py
    python scripts/event_study.py --signal-phase 預熱 --horizons 5 10 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

from src.analyzers.theme_radar import compute_nii_slope

# 中文字型
for fpath in [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]:
    if Path(fpath).exists():
        try:
            font_manager.fontManager.addfont(fpath)
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=fpath).get_name()
            ]
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False


def load_display_names() -> dict[str, str]:
    cfg = yaml.safe_load(open(ROOT / "config" / "topics.yaml", encoding="utf-8"))
    return {t: c.get("display_name", t) for t, c in cfg.items() if isinstance(c, dict)}


def reconstruct_phases(nii: pd.Series, slope_window: int = 14) -> pd.Series:
    """
    用逐日擴張窗口重建歷史相位（與 detect_phase 規則一致，但逐日計算）。

    每天 t 只用 nii[0:t+1] 計算 mean/std（point-in-time，無 lookahead），
    斜率用 trailing slope_window 天（亦為因果，只看過去）。
    """
    valid = nii.dropna()
    slopes = compute_nii_slope(valid, window=slope_window)  # 因果：只用過去 window 天
    phases = pd.Series(index=valid.index, dtype=object)

    for i in range(len(valid)):
        if i < slope_window:          # 歷史不足，無法判定
            phases.iloc[i] = None
            continue
        window_data = valid.iloc[: i + 1]   # 擴張窗口：只到當天
        mu = window_data.mean()
        sigma = window_data.std()
        latest = float(valid.iloc[i])
        slope = slopes.iloc[i]
        slope = 0.0 if pd.isna(slope) else float(slope)

        if latest >= mu + sigma:
            ph = "發燒"
        elif latest >= mu:
            ph = "降溫" if slope < 0 else "發燒"
        else:
            ph = "預熱" if slope > 0 else "冷卻"
        phases.iloc[i] = ph

    return phases


def find_entry_events(phases: pd.Series, target: str = "預熱") -> list:
    """找出「進入 target 相位」的事件日（前一日非 target → 當日 target）。"""
    prev = phases.shift(1)
    mask = (phases == target) & (prev != target) & prev.notna()
    return list(phases.index[mask])


def forward_returns(smi_level: pd.Series, event_dates: list,
                    horizons: range) -> dict[int, list[float]]:
    """計算事件日後 h 天的 SMI 累積報酬。"""
    pos = {d: i for i, d in enumerate(smi_level.index)}
    out: dict[int, list[float]] = {h: [] for h in horizons}
    n = len(smi_level)
    for d in event_dates:
        if d not in pos:
            continue
        i = pos[d]
        base = smi_level.iloc[i]
        if base <= 0 or pd.isna(base):
            continue
        for h in horizons:
            if i + h < n:
                fwd = smi_level.iloc[i + h]
                if not pd.isna(fwd):
                    out[h].append(fwd / base - 1.0)
    return out


def baseline_returns(smi_level: pd.Series, horizons: range) -> dict[int, list[float]]:
    """所有交易日後 h 天的累積報酬（隨機進場基準）。"""
    out: dict[int, list[float]] = {h: [] for h in horizons}
    n = len(smi_level)
    for i in range(n):
        base = smi_level.iloc[i]
        if base <= 0 or pd.isna(base):
            continue
        for h in horizons:
            if i + h < n:
                fwd = smi_level.iloc[i + h]
                if not pd.isna(fwd):
                    out[h].append(fwd / base - 1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="預熱訊號事件研究")
    parser.add_argument("--signal-phase", default="預熱",
                        help="要檢驗的相位（預設 預熱）")
    parser.add_argument("--max-horizon", type=int, default=20,
                        help="最大持有天數（預設 20）")
    parser.add_argument("--slope-window", type=int, default=14)
    args = parser.parse_args()

    horizons = range(1, args.max_horizon + 1)
    key_h = [h for h in (5, 10, 20) if h <= args.max_horizon]

    display_names = load_display_names()

    # 跨題材彙整事件與基準
    all_event: dict[int, list[float]] = {h: [] for h in horizons}
    all_base: dict[int, list[float]] = {h: [] for h in horizons}
    per_topic_events: dict[str, int] = {}

    for topic in display_names:
        nii_path = PROCESSED_DIR / f"{topic}_nii.parquet"
        smi_path = PROCESSED_DIR / f"{topic}_smi.parquet"
        if not (nii_path.exists() and smi_path.exists()):
            continue
        nii = pd.read_parquet(nii_path)["nii"]
        nii.index = pd.to_datetime(nii.index)
        smi = pd.read_parquet(smi_path)
        smi.index = pd.to_datetime(smi.index)
        smi_level = smi["smi_level"]

        phases = reconstruct_phases(nii, args.slope_window)
        events = find_entry_events(phases, args.signal_phase)
        per_topic_events[topic] = len(events)

        ev = forward_returns(smi_level, events, horizons)
        bs = baseline_returns(smi_level, horizons)
        for h in horizons:
            all_event[h].extend(ev[h])
            all_base[h].extend(bs[h])

    # ── 輸出 ──────────────────────────────────────────────────────────
    print("=" * 78)
    print(f"事件研究 — 「進入{args.signal_phase}」訊號後的報酬表現（跨 7 題材彙整）")
    print("=" * 78)
    print("方法：逐日擴張窗口重建相位（無 lookahead）｜股價=SMI 類股等權累積報酬")
    print("-" * 78)
    print("各題材偵測到的進入事件數：")
    total_events = sum(per_topic_events.values())
    for t, c in per_topic_events.items():
        print(f"  {display_names[t]:<18} {c} 次")
    print(f"  → 合計 {total_events} 次事件")
    print("-" * 78)
    print(f"{'持有天數':>6}{'事件平均報酬':>14}{'隨機基準報酬':>14}"
          f"{'超額報酬':>12}{'t值':>8}{'p值':>9}  顯著?")
    print("-" * 78)

    any_significant_positive = False
    for h in key_h:
        ev, bs = all_event[h], all_base[h]
        if len(ev) < 5:
            print(f"{h:>5}天   事件樣本不足（{len(ev)}）")
            continue
        ev_mean, bs_mean = np.mean(ev), np.mean(bs)
        excess = ev_mean - bs_mean
        t_stat, p_val = stats.ttest_ind(ev, bs, equal_var=False)
        sig = "✅是" if (p_val < 0.05 and excess > 0) else \
              ("⚠️顯著但為負" if (p_val < 0.05 and excess < 0) else "否")
        if p_val < 0.05 and excess > 0:
            any_significant_positive = True
        print(f"{h:>5}天{ev_mean*100:>+13.2f}%{bs_mean*100:>+13.2f}%"
              f"{excess*100:>+11.2f}%{t_stat:>8.2f}{p_val:>9.3f}  {sig}")

    print("-" * 78)
    print("\n【結論】")
    if total_events < 10:
        print(f"  ⚠️ 事件數僅 {total_events} 次，樣本過少，統計檢定力不足，"
              "結論僅供參考。")
    if any_significant_positive:
        print(f"  ✅ 「進入{args.signal_phase}」後報酬顯著高於隨機進場，"
              "訊號具早期進場參考價值。")
    else:
        print(f"  ❌ 「進入{args.signal_phase}」後的報酬，與隨機任一天進場相比"
              "「沒有顯著優勢」。")
        print(f"     這代表：產品主張的『{args.signal_phase}=早期進場訊號』"
              "目前缺乏統計證據支持。")

    # ── 繪圖：事件後平均報酬路徑 vs 基準 ────────────────────────────
    hs = list(horizons)
    ev_mean = [np.mean(all_event[h]) * 100 if all_event[h] else np.nan for h in hs]
    ev_se = [np.std(all_event[h]) / np.sqrt(len(all_event[h])) * 100
             if len(all_event[h]) > 1 else np.nan for h in hs]
    bs_mean = [np.mean(all_base[h]) * 100 if all_base[h] else np.nan for h in hs]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ev_mean_arr = np.array(ev_mean)
    ev_se_arr = np.array(ev_se)
    ax.plot(hs, ev_mean, color="#C55A11", lw=2.2,
            marker="o", ms=4, label=f"進入{args.signal_phase}後（事件組）")
    ax.fill_between(hs, ev_mean_arr - ev_se_arr, ev_mean_arr + ev_se_arr,
                    color="#C55A11", alpha=0.15, label="事件組 ±1 標準誤")
    ax.plot(hs, bs_mean, color="#555555", lw=1.8, ls="--",
            marker="s", ms=3, label="隨機任一天進場（基準）")
    ax.axhline(0, color="black", lw=0.6)

    ax.set_title(
        f"事件研究：進入「{args.signal_phase}」後的 SMI 平均累積報酬　"
        f"（{total_events} 次事件，跨 7 題材）",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("進場後持有天數（交易日）", fontsize=10)
    ax.set_ylabel("平均累積報酬（%）", fontsize=10)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.25)

    out_path = REPORTS_DIR / "event_study_warming.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n圖表已輸出：{out_path}")
    print("\n⚠️ 樣本僅約 116 天、事件不完全獨立，本檢定為初步驗證。")


if __name__ == "__main__":
    main()
