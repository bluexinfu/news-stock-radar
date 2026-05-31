#!/usr/bin/env python3
"""
熱度指標 v2 — 多源合成（P2 高潮）

合成「新聞（權威加權）」+「YouTube（觀看加權）」兩個投資人/媒體關注來源。
（PTT 經原型驗證題材訊號太稀疏，架構保留但暫不納入合成。）

關鍵：避免重蹈舊 NII 的尺度錯配——
  每個來源各自「全域正規化到 0~100」後再加權合成，確保兩源同尺度。

輸出：
  - 合成熱度（水位）+ 跨題材排名
  - 驗證：合成熱度 vs 純新聞 vs 舊NII，三者與股價（SMI 報酬）的交叉相關

⚠️ 股價僅作對照，不餵進指標（呼應 Q1）。樣本有限，初步驗證。

用法：
  python scripts/heat_composite.py
  python scripts/heat_composite.py --w-news 0.5 --w-youtube 0.5
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROCESSED_DIR = ROOT / "data" / "processed"
YT_DIR = ROOT / "data" / "raw" / "youtube"
REPORTS_DIR = ROOT / "reports"

from heat_index import load_authority, build_heat, load_display_names

for fpath in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if Path(fpath).exists():
        font_manager.fontManager.addfont(fpath)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fpath).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False


def youtube_daily(topic: str, calendar: pd.DatetimeIndex, smooth: int = 7) -> pd.Series:
    """YouTube 每日加權觸及，對齊交易日後做 7 日滾動和（把稀疏影片事件平滑成水位）。"""
    f = YT_DIR / f"{topic}_youtube.parquet"
    if not f.exists():
        return pd.Series(0.0, index=calendar)
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    reach = df["reach"].groupby(df.index).sum()
    s = reach.reindex(calendar, fill_value=0.0)
    return s.rolling(smooth, min_periods=1).sum()


def minmax(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return (s - lo) / (hi - lo + 1e-9) * 100


def rolling_slope(s: pd.Series, window: int = 7) -> pd.Series:
    x = np.arange(window, dtype=float)
    xm, xv = x.mean(), ((x - x.mean()) ** 2).sum()
    return s.rolling(window).apply(
        lambda y: np.nan if np.isnan(y).any() else ((x - xm) * (y - y.mean())).sum() / xv,
        raw=True)


def ccf_contemp_peak(signal: pd.Series, ret: pd.Series, max_lag: int = 10):
    s = signal.diff()
    best_lag, best_r, contemp = 0, 0.0, 0.0
    for k in range(-max_lag, max_lag + 1):
        sh = ret.shift(-k)
        m = s.notna() & sh.notna()
        if m.sum() < 20:
            continue
        r = float(np.corrcoef(s[m], sh[m])[0, 1])
        if k == 0:
            contemp = r
        if abs(r) > abs(best_r):
            best_r, best_lag = r, k
    n = int((s.notna() & ret.notna()).sum())
    return contemp, best_lag, best_r, 1.96 / np.sqrt(max(n, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description="熱度合成 + 驗證")
    # 定案（設計選項 B）：新聞主導，幾乎保住股價關聯，
    # YouTube 作為可見的「影片型關注」輔助維度（不稀釋）。
    ap.add_argument("--w-news", type=float, default=0.8)
    ap.add_argument("--w-youtube", type=float, default=0.2)
    args = ap.parse_args()

    weights, default = load_authority()
    names = load_display_names()

    # Pass 1：建每題材的新聞水位、YouTube 水位（同一交易日曆）
    news_raw, yt_raw, nii_raw, ret_raw = {}, {}, {}, {}
    for topic in names:
        heat = build_heat(topic, weights, default, smooth=7, slope=7)
        if heat.empty:
            continue
        cal = heat.index
        news_raw[topic] = heat["level"]
        yt_raw[topic] = youtube_daily(topic, cal)
        nii = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
        nii.index = pd.to_datetime(nii.index)
        nii_raw[topic] = nii["nii"].reindex(cal)
        smi = pd.read_parquet(PROCESSED_DIR / f"{topic}_smi.parquet")
        smi.index = pd.to_datetime(smi.index)
        ret_raw[topic] = smi["smi_return"].reindex(cal)

    # 全域正規化邊界（跨題材池化，保留題材間差異）
    news_lo = min(s.min() for s in news_raw.values()); news_hi = max(s.max() for s in news_raw.values())
    yt_lo = min(s.min() for s in yt_raw.values()); yt_hi = max(s.max() for s in yt_raw.values())

    wsum = args.w_news + args.w_youtube
    wn, wy = args.w_news / wsum, args.w_youtube / wsum

    composite, summ = {}, []
    for topic in news_raw:
        nn = minmax(news_raw[topic], news_lo, news_hi)
        yy = minmax(yt_raw[topic], yt_lo, yt_hi)
        comp = wn * nn + wy * yy
        composite[topic] = comp
        summ.append({
            "題材": names[topic],
            "新聞型關注": round(float(nn.mean()), 1),
            "影片型關注": round(float(yy.mean()), 1),
            "綜合熱度": round(float(comp.mean()), 1),
        })

    sdf = pd.DataFrame(summ).sort_values("綜合熱度", ascending=False)
    pd.set_option("display.unicode.east_asian_width", True); pd.set_option("display.width", 200)
    print("=" * 70)
    print(f"熱度合成 v2（新聞 {wn:.0%} + YouTube {wy:.0%}）— 跨題材排名")
    print("=" * 70)
    print(sdf.to_string(index=False))

    # 驗證：三訊號 vs 股價
    print("\n" + "=" * 70)
    print("驗證：與股價（SMI 報酬）同步相關 —— 舊NII vs 純新聞 vs 合成熱度")
    print("=" * 70)
    print(f"{'題材':<16}{'舊NII':>9}{'純新聞':>9}{'合成':>9}   最佳")
    print("-" * 70)
    acc = {"nii": [], "news": [], "comp": []}
    for topic in news_raw:
        # 驗證窗：YouTube 有資料起算
        start = yt_raw[topic][yt_raw[topic] > 0].index.min()
        sl = slice(start, None)
        ret = ret_raw[topic]
        c_nii, *_ = ccf_contemp_peak(nii_raw[topic][sl], ret[sl])
        c_news, *_ = ccf_contemp_peak(news_raw[topic][sl], ret[sl])
        c_comp, *_ = ccf_contemp_peak(composite[topic][sl], ret[sl])
        acc["nii"].append(abs(c_nii)); acc["news"].append(abs(c_news)); acc["comp"].append(abs(c_comp))
        best = max([("舊NII", abs(c_nii)), ("純新聞", abs(c_news)), ("合成", abs(c_comp))], key=lambda x: x[1])[0]
        print(f"{names[topic]:<16}{c_nii:>+9.3f}{c_news:>+9.3f}{c_comp:>+9.3f}   {best}")
    print("-" * 70)
    print(f"{'平均|相關|':<16}{np.mean(acc['nii']):>9.3f}{np.mean(acc['news']):>9.3f}"
          f"{np.mean(acc['comp']):>9.3f}")

    if np.mean(acc["comp"]) >= max(np.mean(acc["news"]), np.mean(acc["nii"])):
        print("\n→ 合成熱度與股價的貼合度『最佳』，加入 YouTube 觸及確實提升了關注度代理品質。")
    elif np.mean(acc["news"]) >= np.mean(acc["nii"]):
        print("\n→ 純新聞已優於舊NII；合成未再明顯提升（YouTube 邊際貢獻有限）。")

    # 圖：合成熱度 vs 股價（每題材）
    topics = list(composite.keys()); n = len(topics)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 3.0 * ((n + 1) // 2)))
    axes = axes.flatten()
    for i, t in enumerate(topics):
        ax = axes[i]
        comp = composite[t]
        start = yt_raw[t][yt_raw[t] > 0].index.min()
        def nrm(s):
            s = s.loc[start:].astype(float); rng = s.max() - s.min()
            return (s - s.min()) / (rng + 1e-9) * 100
        ax.plot(comp.loc[start:].index, nrm(comp), color="#C55A11", lw=2, label="合成熱度")
        ax.plot(nii_raw[t].loc[start:].index, nrm(nii_raw[t]), color="#999", lw=1.2, ls="--", label="舊 NII")
        smi = pd.read_parquet(PROCESSED_DIR / f"{t}_smi.parquet"); smi.index = pd.to_datetime(smi.index)
        ax.plot(smi["smi_level"].loc[start:].index, nrm(smi["smi_level"].loc[start:]),
                color="#1976D2", lw=1.2, alpha=0.7, label="股價(對照)")
        ax.set_title(names[t], fontsize=11); ax.tick_params(labelsize=7)
        if i == 0: ax.legend(fontsize=8)
    for j in range(n, len(axes)): axes[j].axis("off")
    fig.suptitle("熱度合成 v2：合成熱度(橘) vs 舊NII(灰虛) vs 股價(藍對照)",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPORTS_DIR / "heat_composite_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n圖表已輸出：{out}")


if __name__ == "__main__":
    main()
