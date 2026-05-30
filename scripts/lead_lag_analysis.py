#!/usr/bin/env python3
"""
領先/落後交叉相關分析（Lead-Lag Cross-Correlation）

驗證題材雷達的核心假設：「NII（新聞強度）是否領先股價？」

方法：
    計算 corr(ΔNII_t, 報酬_{t+k})，k 從 -MAX_LAG 到 +MAX_LAG 天。
    - k > 0（正 lag）相關性高 → NII 領先股價（系統有預測價值）✅
    - k = 0 相關性高       → 同步反應（新聞與股價同時動）
    - k < 0（負 lag）相關性高 → NII 落後股價（新聞追著股價跑，追高風險）❌

股價代理：
    SMI（類股等權報酬）為主——一籃子概念股，雜訊低，最能代表整個題材。

統計顯著性：
    近似 95% 信賴區間 = ±1.96 / sqrt(N)。
    超出此區間的相關係數才視為顯著（非隨機雜訊）。

⚠️ 本分析為方法論驗證工具，不構成投資建議。

用法：
    python scripts/lead_lag_analysis.py
    python scripts/lead_lag_analysis.py --max-lag 15 --nii-col nii
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# 中文字型：直接註冊 macOS 系統字型檔，避免 matplotlib 找不到字型名
for fpath in [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Microsoft JhengHei.ttf",
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


def load_series(topic: str, nii_col: str = "nii",
                signal_mode: str = "change") -> pd.DataFrame | None:
    """
    回傳含 signal 與 smi_return 的對齊 DataFrame。

    signal_mode:
        "change" — ΔNII（日變化），統計嚴謹、避免假相關
        "level"  — NII 絕對水位，貼近 Phase 實際邏輯，但須留意假相關
    """
    nii_path = PROCESSED_DIR / f"{topic}_nii.parquet"
    smi_path = PROCESSED_DIR / f"{topic}_smi.parquet"
    if not (nii_path.exists() and smi_path.exists()):
        return None

    nii = pd.read_parquet(nii_path)
    smi = pd.read_parquet(smi_path)
    nii.index = pd.to_datetime(nii.index)
    smi.index = pd.to_datetime(smi.index)

    if signal_mode == "level":
        signal = nii[nii_col]
    else:
        signal = nii[nii_col].diff()  # 預設：變化量

    df = pd.DataFrame({
        "signal": signal,
        "smi_return": smi["smi_return"],
    }).dropna()
    return df if len(df) >= 30 else None


def cross_correlation(
    signal: pd.Series, price: pd.Series, max_lag: int
) -> pd.DataFrame:
    """
    計算 corr(signal_t, price_{t+k})，k 從 -max_lag 到 +max_lag。

    price.shift(-k) 把 price_{t+k} 對齊到 index t，
    因此 signal.corr(price.shift(-k)) = corr(signal_t, price_{t+k})。
    """
    rows = []
    for k in range(-max_lag, max_lag + 1):
        shifted = price.shift(-k)
        common = signal.notna() & shifted.notna()
        s, p = signal[common], shifted[common]
        if len(s) < 20:
            rows.append({"lag": k, "corr": np.nan, "n": len(s)})
            continue
        r = float(np.corrcoef(s, p)[0, 1])
        rows.append({"lag": k, "corr": r, "n": len(s)})
    return pd.DataFrame(rows)


def summarize(ccf: pd.DataFrame) -> dict:
    """從交叉相關函數萃取關鍵指標。"""
    valid = ccf.dropna(subset=["corr"])
    n_mean = int(valid["n"].mean())
    ci = 1.96 / np.sqrt(n_mean)  # 近似 95% 信賴區間

    # 峰值（取絕對值最大者）
    peak_idx = valid["corr"].abs().idxmax()
    peak = valid.loc[peak_idx]
    peak_lag = int(peak["lag"])
    peak_corr = float(peak["corr"])

    # 正 lag（NII 領先）vs 負 lag（NII 落後）最強相關
    lead_side = valid[valid["lag"] > 0]["corr"]
    lag_side = valid[valid["lag"] < 0]["corr"]
    contemp = float(valid[valid["lag"] == 0]["corr"].iloc[0])

    lead_max = float(lead_side.abs().max()) if len(lead_side) else 0.0
    lag_max = float(lag_side.abs().max()) if len(lag_side) else 0.0

    # 判定（同時考慮 lag 方向與相關正負號）
    if abs(peak_corr) < ci:
        verdict = "無顯著訊號"
    elif peak_lag > 0 and peak_corr > 0:
        verdict = "NII 正向領先 ✅"      # 新聞熱 → 股價漲（真正可用的領先訊號）
    elif peak_lag > 0 and peak_corr < 0:
        verdict = "NII 反向領先（反指標）⚠️"  # 新聞熱 → 股價跌
    elif peak_lag < 0:
        verdict = "NII 落後股價 ❌"       # 股價先動，新聞才跟上
    else:
        verdict = "同步（無領先性）"

    return {
        "peak_lag": peak_lag,
        "peak_corr": peak_corr,
        "contemp_corr": contemp,
        "lead_max_abs": lead_max,
        "lag_max_abs": lag_max,
        "ci_95": ci,
        "significant": abs(peak_corr) >= ci,
        "verdict": verdict,
        "n": n_mean,
    }


def plot_ccf(results: dict[str, dict], display_names: dict[str, str],
             max_lag: int, out_path: Path) -> None:
    topics = list(results.keys())
    n = len(topics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows))
    axes = axes.flatten()

    for i, topic in enumerate(topics):
        ax = axes[i]
        ccf = results[topic]["ccf"]
        summ = results[topic]["summary"]
        ci = summ["ci_95"]

        colors = ["#C62828" if l < 0 else "#1B5E20" if l > 0 else "#888888"
                  for l in ccf["lag"]]
        ax.bar(ccf["lag"], ccf["corr"], color=colors, width=0.7, alpha=0.85)

        # 信賴區間
        ax.axhline(ci, ls="--", color="gray", lw=0.8)
        ax.axhline(-ci, ls="--", color="gray", lw=0.8)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6, alpha=0.3)

        # 標註峰值
        pl, pc = summ["peak_lag"], summ["peak_corr"]
        ax.scatter([pl], [pc], color="orange", zorder=5, s=40,
                   edgecolors="black", linewidths=0.6)

        dn = display_names.get(topic, topic)
        ax.set_title(f"{dn}\n{summ['verdict']}（峰值 lag={pl:+d}, r={pc:+.2f}）",
                     fontsize=10)
        ax.set_xlabel("lag（天）　負=NII落後 ◀  ▶ 正=NII領先", fontsize=8)
        ax.set_ylabel("相關係數", fontsize=8)
        ax.set_xticks(range(-max_lag, max_lag + 1, 2))
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "題材雷達：NII 領先/落後股價 交叉相關分析　"
        "（綠=NII領先 ✅　紅=NII落後 ❌　灰虛線=95%顯著門檻）",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="領先/落後交叉相關分析")
    parser.add_argument("--max-lag", type=int, default=10,
                        help="最大領先/落後天數（預設 10）")
    parser.add_argument("--nii-col", default="nii",
                        help="NII 欄位名（預設 nii）")
    parser.add_argument("--signal-mode", choices=["change", "level"],
                        default="change",
                        help="change=ΔNII變化量（嚴謹）；level=NII絕對水位（貼近Phase）")
    args = parser.parse_args()

    display_names = load_display_names()
    results: dict[str, dict] = {}

    mode_label = "ΔNII（日變化）" if args.signal_mode == "change" else "NII（絕對水位）"
    print("=" * 78)
    print(f"題材雷達 — NII 領先/落後股價 交叉相關分析　[{mode_label}]")
    print("=" * 78)
    print(f"股價代理：SMI（類股等權報酬）　訊號：{mode_label}　"
          f"最大 lag：±{args.max_lag} 天")
    print("-" * 78)
    print(f"{'題材':<18}{'樣本':>5}{'同步r':>8}{'峰值lag':>8}{'峰值r':>8}"
          f"{'95%門檻':>9}  判定")
    print("-" * 78)

    for topic in display_names:
        df = load_series(topic, args.nii_col, args.signal_mode)
        if df is None:
            print(f"{display_names[topic]:<18}  資料不足，跳過")
            continue
        ccf = cross_correlation(df["signal"], df["smi_return"], args.max_lag)
        summ = summarize(ccf)
        results[topic] = {"ccf": ccf, "summary": summ}

        dn = display_names[topic]
        print(f"{dn:<18}{summ['n']:>5}{summ['contemp_corr']:>+8.3f}"
              f"{summ['peak_lag']:>+8d}{summ['peak_corr']:>+8.3f}"
              f"{summ['ci_95']:>9.3f}  {summ['verdict']}")

    print("-" * 78)

    # 整體結論（依精煉後的判定分類）
    def count(key):
        return sum(1 for r in results.values() if key in r["summary"]["verdict"])

    pos_lead = count("正向領先")
    rev_lead = count("反向領先")
    lag_cnt = count("落後")
    sync_cnt = count("同步")
    insig_cnt = count("無顯著")

    print(f"\n【整體結論】共 {len(results)} 個題材：")
    print(f"  ✅ NII 正向領先股價（新聞熱→股價漲）：{pos_lead} 個")
    print(f"  ⚠️ NII 反向領先（反指標，新聞熱→股價跌）：{rev_lead} 個")
    print(f"  ❌ NII 落後股價（股價先動，新聞才跟）：{lag_cnt} 個")
    print(f"  ⏸ 同步無領先性（lag=0 最強）：　{sync_cnt} 個")
    print(f"  ⚪ 無顯著訊號（落在雜訊範圍）：　{insig_cnt} 個")

    if pos_lead >= len(results) / 2:
        print("\n  → 多數題材顯示 NII 正向領先股價，系統核心假設「初步成立」。")
    elif insig_cnt + sync_cnt >= len(results) / 2:
        print("\n  → 關鍵發現：多數題材的 NII 與股價是「同步」或「無顯著關聯」，"
              "\n    幾乎沒有題材呈現乾淨的「正向領先」。")
        print("    這代表在目前 116 天的資料下，NII 並未展現可靠的『領先』預測力——")
        print("    最強的關聯（矽光子、被動元件）都是『同步』的，意味新聞與股價同時反應，")
        print("    無法用來『提前』布局。把 Phase 訊號當進場依據，目前缺乏統計支持。")
    else:
        print("\n  → 警訊：NII 多為落後或反向，新聞熱度可能『跟著股價跑』，"
              "拿來進場有追高風險。")

    out_path = REPORTS_DIR / f"lead_lag_{args.signal_mode}.png"
    plot_ccf(results, display_names, args.max_lag, out_path)
    print(f"\n圖表已輸出：{out_path}")
    print("\n⚠️ 樣本僅約 116 天，本結論為初步驗證，需累積更多資料再確認。")


if __name__ == "__main__":
    main()
