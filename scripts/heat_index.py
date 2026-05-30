#!/usr/bin/env python3
"""
熱度指標 v2（Attention Heat Index）— P1 地基版

重新定義「市場熱度」，取代有結構缺陷的舊 NII。

設計原則（依與使用者的討論定案）：
  Q1 準確描述關注度，股價僅作對照（不餵進指標）
  Q2 來源 = 投資人 + 優質媒體；砍掉一般大眾 Google 搜尋
  Q3 同時輸出「水位」與「加速度」

核心公式（在單篇內容層級合成，根治舊版尺度錯配）：
  某日熱度 = Σ 每篇新聞：來源可信度權重(media)
  （P1 先做媒體管道；P2 再加 YouTube 觀看數、PTT 討論量等投資人觸及訊號）

輸出：
  - 水位（level）：7 日平滑後的加權熱度
  - 加速度（accel）：7 日滾動線性斜率
  - 對照：舊 NII、SMI 股價（疊圖，獨立層）

⚠️ 熱度 = 純關注量，不含情緒方向（情緒為另一獨立維度）。

用法：
  python scripts/heat_index.py            # 跑全部題材，輸出對照表 + 圖
  python scripts/heat_index.py --smooth 7 --slope 7
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT / "data" / "raw" / "news"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

# 中文字型
for fpath in ["/System/Library/Fonts/PingFang.ttc",
              "/System/Library/Fonts/STHeiti Light.ttc"]:
    if Path(fpath).exists():
        font_manager.fontManager.addfont(fpath)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fpath).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


# ── 設定載入 ─────────────────────────────────────────────────────────

def load_authority() -> tuple[dict, float]:
    cfg = yaml.safe_load(open(ROOT / "config" / "source_authority.yaml", encoding="utf-8"))
    # 以小寫 key 方便不分大小寫比對
    weights = {str(k).strip().lower(): float(v) for k, v in cfg.get("sources", {}).items()}
    return weights, float(cfg.get("default", 0.3))


def load_display_names() -> dict[str, str]:
    cfg = yaml.safe_load(open(ROOT / "config" / "topics.yaml", encoding="utf-8"))
    return {t: c.get("display_name", t) for t, c in cfg.items() if isinstance(c, dict)}


# ── 熱度計算 ─────────────────────────────────────────────────────────

def weight_for(source: str, weights: dict, default: float) -> float:
    return weights.get(str(source).strip().lower(), default)


def daily_weighted_heat(topic: str, weights: dict, default: float) -> pd.Series:
    """讀取題材所有新聞，去重後，逐日加總來源權重。"""
    f = NEWS_DIR / f"{topic}_googlenews.parquet"
    if not f.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(f)
    # 同一篇文章可能被多個關鍵字查到 → 以 url 去重
    df = df.drop_duplicates(subset=["url"])
    df["w"] = df["source"].map(lambda s: weight_for(s, weights, default))
    df["date"] = pd.to_datetime(df["published"]).dt.normalize()
    daily = df.groupby("date")["w"].sum().sort_index()
    return daily


def rolling_slope(s: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    xm, xv = x.mean(), ((x - x.mean()) ** 2).sum()
    def _sl(y):
        if np.isnan(y).any():
            return np.nan
        return ((x - xm) * (y - y.mean())).sum() / xv
    return s.rolling(window).apply(_sl, raw=True)


def build_heat(topic: str, weights: dict, default: float,
               smooth: int, slope: int) -> pd.DataFrame:
    """回傳 index=date，欄位 [raw, level, accel] 的熱度表，對齊交易日。"""
    raw = daily_weighted_heat(topic, weights, default)
    if raw.empty:
        return pd.DataFrame()

    # 對齊到 NII 既有的交易日索引（含週末併入下一交易日的處理）
    nii = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
    nii.index = pd.to_datetime(nii.index)
    cal = nii.index

    # 把每篇新聞的日期 reindex 到交易日：非交易日往後併入下一個交易日
    raw_aligned = pd.Series(0.0, index=cal)
    for d, v in raw.items():
        # 找 >= d 的第一個交易日
        future = cal[cal >= d]
        if len(future):
            raw_aligned.loc[future[0]] += v

    level = raw_aligned.rolling(smooth, min_periods=1).mean()
    accel = rolling_slope(level, slope)
    return pd.DataFrame({"raw": raw_aligned, "level": level, "accel": accel})


# ── 主程式 ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="熱度指標 v2（P1）")
    ap.add_argument("--smooth", type=int, default=7, help="水位平滑窗口（天）")
    ap.add_argument("--slope", type=int, default=7, help="加速度斜率窗口（天）")
    args = ap.parse_args()

    weights, default = load_authority()
    names = load_display_names()

    results = {}
    rows = []
    for topic in names:
        heat = build_heat(topic, weights, default, args.smooth, args.slope)
        if heat.empty:
            continue
        nii = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
        results[topic] = {"heat": heat, "nii": nii["nii"]}
        rows.append({
            "題材": names[topic],
            "新聞總量": int((NEWS_DIR / f"{topic}_googlenews.parquet").exists()
                          and len(pd.read_parquet(NEWS_DIR / f"{topic}_googlenews.parquet")
                                  .drop_duplicates(subset=["url"]))),
            "舊NII均值": round(float(nii["nii"].mean()), 1),
            "新熱度均值": round(float(heat["level"].mean()), 1),
        })

    df = pd.DataFrame(rows)
    df["舊排名"] = df["舊NII均值"].rank(ascending=False).astype(int)
    df["新排名"] = df["新熱度均值"].rank(ascending=False).astype(int)
    df = df.sort_values("新熱度均值", ascending=False)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print("=" * 70)
    print("熱度指標 v2（P1：權威加權新聞量）— 與舊 NII 對照")
    print("=" * 70)
    print(df.to_string(index=False))
    print()
    print("重點觀察：")
    print("  • 舊 NII 約 99% 由 Google Trends 決定，與真實新聞量脫節")
    print("  • 新熱度 = 權威加權的真實新聞量，名實相符")

    # ── 繪圖：每個題材 新熱度 vs 舊NII vs 股價 ────────────────────
    topics = list(results.keys())
    n = len(topics)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 3.0 * ((n + 1) // 2)))
    axes = axes.flatten()
    for i, topic in enumerate(topics):
        ax = axes[i]
        heat = results[topic]["heat"]
        nii = results[topic]["nii"].reindex(heat.index)
        # 正規化到 0~100 方便同圖比較形狀
        def norm(s):
            s = s.astype(float)
            rng = s.max() - s.min()
            return (s - s.min()) / (rng + 1e-9) * 100
        ax.plot(heat.index, norm(heat["level"]), color="#C55A11", lw=2,
                label="新熱度（權威加權新聞）")
        ax.plot(nii.index, norm(nii), color="#888888", lw=1.4, ls="--",
                label="舊 NII（≈Google Trends）")
        # 股價疊圖（SMI）
        smi_path = PROCESSED_DIR / f"{topic}_smi.parquet"
        if smi_path.exists():
            smi = pd.read_parquet(smi_path); smi.index = pd.to_datetime(smi.index)
            ax.plot(smi.index, norm(smi["smi_level"].reindex(heat.index)),
                    color="#1976D2", lw=1.2, alpha=0.7, label="股價 SMI（對照）")
        ax.set_title(names[topic], fontsize=11)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("正規化 0~100", fontsize=8)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("熱度指標 v2：新熱度（橘）vs 舊NII（灰虛，≈Trends）vs 股價（藍，對照）",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPORTS_DIR / "heat_index_v2_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n圖表已輸出：{out}")


if __name__ == "__main__":
    main()
