#!/usr/bin/env python3
"""
驗證：新熱度 vs 舊 NII，哪個跟股價關係更好？

對每個題材，計算「訊號日變化」與「SMI 報酬」的交叉相關（lag −10~+10）：
  - 舊 NII（≈Google Trends）
  - 新熱度（權威加權新聞量）
比較兩者的「同步相關」與「峰值」，看新熱度是否更貼近市場。

呼應 Q1：股價只作對照，不餵進指標。本檢驗純為診斷。

⚠️ 樣本約 116 天，初步驗證；相關不等於因果，不構成投資建議。

用法：
  python scripts/validate_heat_vs_price.py
"""

from __future__ import annotations

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
REPORTS_DIR = ROOT / "reports"

from heat_index import load_authority, build_heat, load_display_names

for fpath in ["/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/STHeiti Light.ttc"]:
    if Path(fpath).exists():
        font_manager.fontManager.addfont(fpath)
        plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=fpath).get_name()]
        break
plt.rcParams["axes.unicode_minus"] = False

MAX_LAG = 10


def ccf(signal: pd.Series, price: pd.Series, max_lag: int = MAX_LAG) -> pd.DataFrame:
    rows = []
    for k in range(-max_lag, max_lag + 1):
        shifted = price.shift(-k)
        m = signal.notna() & shifted.notna()
        s, p = signal[m], shifted[m]
        if len(s) < 20:
            rows.append({"lag": k, "corr": np.nan}); continue
        rows.append({"lag": k, "corr": float(np.corrcoef(s, p)[0, 1])})
    return pd.DataFrame(rows)


def summarize(c: pd.DataFrame, n: int):
    v = c.dropna()
    ci = 1.96 / np.sqrt(n)
    contemp = float(v[v["lag"] == 0]["corr"].iloc[0])
    peak = v.loc[v["corr"].abs().idxmax()]
    return contemp, int(peak["lag"]), float(peak["corr"]), ci


def main() -> None:
    weights, default = load_authority()
    names = load_display_names()

    results = {}
    print("=" * 84)
    print("驗證：新熱度 vs 舊 NII —— 與股價（SMI 報酬）的交叉相關")
    print("=" * 84)
    print("訊號皆取『日變化』vs SMI 報酬（嚴謹版，避免假相關）")
    print("-" * 84)
    print(f"{'題材':<16}{'舊NII同步':>10}{'舊NII峰值':>14}"
          f"{'新熱度同步':>12}{'新熱度峰值':>14}   誰較貼股價")
    print("-" * 84)

    win_new = win_old = 0
    for topic in names:
        heat = build_heat(topic, weights, default, smooth=7, slope=7)
        if heat.empty:
            continue
        nii = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
        nii.index = pd.to_datetime(nii.index)
        smi = pd.read_parquet(PROCESSED_DIR / f"{topic}_smi.parquet")
        smi.index = pd.to_datetime(smi.index)
        ret = smi["smi_return"]

        idx = heat.index
        d_nii = nii["nii"].reindex(idx).diff()
        d_heat = heat["level"].diff()
        ret = ret.reindex(idx)

        c_old = ccf(d_nii, ret)
        c_new = ccf(d_heat, ret)
        n = int(min(d_nii.notna().sum(), d_heat.notna().sum(), ret.notna().sum()))
        co, lo, po, ci = summarize(c_old, n)
        cn, ln, pn, _ = summarize(c_new, n)
        results[topic] = {"old": c_old, "new": c_new, "ci": ci}

        better = "🟠 新熱度" if abs(cn) > abs(co) else "⚪ 舊NII"
        if abs(cn) > abs(co): win_new += 1
        else: win_old += 1
        print(f"{names[topic]:<16}{co:>+10.3f}{f'{po:+.2f}@{lo:+d}':>14}"
              f"{cn:>+12.3f}{f'{pn:+.2f}@{ln:+d}':>14}   {better}")

    print("-" * 84)
    print(f"95% 顯著門檻 ≈ ±{ci:.3f}")
    print(f"\n【同步相關（與當期股價貼合度）】新熱度勝 {win_new} 個 / 舊NII勝 {win_old} 個")
    avg_old = np.mean([abs(summarize(results[t]['old'], 100)[0]) for t in results])
    avg_new = np.mean([abs(summarize(results[t]['new'], 100)[0]) for t in results])
    print(f"平均同步相關絕對值：舊NII={avg_old:.3f}　新熱度={avg_new:.3f}")
    if avg_new > avg_old:
        print("→ 新熱度與股價的同步貼合度『整體較高』，是更好的關注度代理。")
    else:
        print("→ 新熱度與股價的貼合度未明顯優於舊NII，需進一步檢視。")

    # 圖：每題材 CCF 疊圖（舊 vs 新）
    topics = list(results.keys())
    n_t = len(topics)
    fig, axes = plt.subplots((n_t + 1) // 2, 2, figsize=(13, 2.8 * ((n_t + 1) // 2)))
    axes = axes.flatten()
    for i, t in enumerate(topics):
        ax = axes[i]
        co = results[t]["old"]; cn = results[t]["new"]; ci = results[t]["ci"]
        ax.plot(co["lag"], co["corr"], color="#999999", lw=1.5, ls="--", marker="s", ms=3, label="舊 NII")
        ax.plot(cn["lag"], cn["corr"], color="#C55A11", lw=2, marker="o", ms=3, label="新熱度")
        ax.axhline(ci, ls=":", color="gray", lw=0.7); ax.axhline(-ci, ls=":", color="gray", lw=0.7)
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="black", lw=0.5, alpha=0.3)
        ax.set_title(names[t], fontsize=10)
        ax.set_xlabel("lag（負=訊號落後 ◀ ▶ 正=訊號領先）", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0: ax.legend(fontsize=8)
    for j in range(n_t, len(axes)): axes[j].axis("off")
    fig.suptitle("新熱度 vs 舊NII：與股價的交叉相關（橘=新熱度，灰虛=舊NII）",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = REPORTS_DIR / "heat_vs_price_validation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n圖表已輸出：{out}")
    print("\n⚠️ 樣本約 116 天，初步驗證。")


if __name__ == "__main__":
    main()
