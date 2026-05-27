"""
Phase 1 HTML 報告產生器

執行方式：
  python -m src.report_generator --topic cowos
"""

from __future__ import annotations

import argparse
import base64
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"

TICKER_NAMES = {
    "TWII":    "台灣加權",
    "2330.TW": "台積電",
    "3413.TW": "京鼎",
    "3583.TW": "辛耘",
    "3680.TWO": "家登",
    "3131.TWO": "弘塑",
}

STOCK_TICKERS = ["2330.TW", "3413.TW", "3583.TW", "3680.TWO", "3131.TWO"]


def img_tag(path: Path, alt: str = "", style: str = "max-width:100%;") -> str:
    if not path.exists():
        return f'<p style="color:gray">[圖片未找到：{path.name}]</p>'
    data = base64.b64encode(path.read_bytes()).decode()
    ext = path.suffix.lstrip(".")
    return f'<img src="data:image/{ext};base64,{data}" alt="{alt}" style="{style}">'


def pearson_r(s1: pd.Series, s2: pd.Series):
    common = s1.index.intersection(s2.index)
    a, b = s1.loc[common].dropna(), s2.loc[common].dropna()
    c = a.index.intersection(b.index)
    r, p = stats.pearsonr(a.loc[c], b.loc[c])
    return round(r, 4), round(p, 5), len(c)


def safe_col(ticker: str) -> str:
    return "close_" + ticker.replace(".", "_").replace("^", "")


def load_results(topic: str):
    aligned = pd.read_parquet(PROCESSED_DIR / f"{topic}_aligned.parquet")
    nii_df  = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    nii_df.index  = pd.to_datetime(nii_df.index)
    return aligned, nii_df


def build_a2_table(aligned: pd.DataFrame, nii_df: pd.DataFrame) -> str:
    nii_sig = nii_df["nii"].diff()
    rows = []
    for ticker in STOCK_TICKERS:
        col = safe_col(ticker)
        if col not in aligned.columns:
            continue
        price_ret = aligned[col].pct_change()
        r, p, n = pearson_r(nii_sig, price_ret)
        sig = "✓" if p < 0.05 else "✗"
        color = "#1a7340" if p < 0.05 else "#888"
        rows.append(
            f"<tr>"
            f"<td>{ticker}</td>"
            f"<td>{TICKER_NAMES.get(ticker, ticker)}</td>"
            f"<td style='color:{color};font-weight:bold'>{r:+.4f}</td>"
            f"<td>{p:.5f}</td>"
            f"<td>{n}</td>"
            f"<td style='color:{color}'>{sig}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def build_robustness_table(aligned: pd.DataFrame, nii_df: pd.DataFrame) -> str:
    periods = {
        "全期": (None, None),
        "早期": (None, "2025-07-01"),
        "近期": ("2025-07-01", None),
    }
    nii_sig = nii_df["nii"].diff()
    header = "<tr><th>標的</th><th>名稱</th>" + "".join(f"<th>r ({p})</th>" for p in periods) + "</tr>"
    rows = [header]
    for ticker in STOCK_TICKERS:
        col = safe_col(ticker)
        if col not in aligned.columns:
            continue
        cells = [f"<td>{ticker}</td><td>{TICKER_NAMES.get(ticker, '')}</td>"]
        for p_label, (start, end) in periods.items():
            mask = pd.Series(True, index=nii_sig.index)
            if start:
                mask &= nii_sig.index >= pd.Timestamp(start)
            if end:
                mask &= nii_sig.index < pd.Timestamp(end)
            r, p, _ = pearson_r(nii_sig[mask], aligned[col].pct_change()[mask])
            sig = "✓" if p < 0.05 else ""
            color = "#1a7340" if p < 0.05 else "#888"
            cells.append(f"<td style='color:{color}'>{r:+.4f} {sig}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def generate(topic: str = "cowos") -> Path:
    aligned, nii_df = load_results(topic)
    nii = nii_df["nii"]
    start_date = aligned.index.min().strftime("%Y-%m-%d")
    end_date   = aligned.index.max().strftime("%Y-%m-%d")

    a2_rows  = build_a2_table(aligned, nii_df)
    rb_rows  = build_robustness_table(aligned, nii_df)

    # 最強相關判斷
    nii_sig = nii.diff()
    best_r = 0.0
    for ticker in STOCK_TICKERS:
        col = safe_col(ticker)
        if col not in aligned.columns:
            continue
        r, _, _ = pearson_r(nii_sig, aligned[col].pct_change())
        if abs(r) > abs(best_r):
            best_r = r

    if abs(best_r) >= 0.3:
        verdict = "✅ 有清晰訊號（|r| ≥ 0.3）→ 建議進入 Phase 2"
        verdict_color = "#1a7340"
    elif abs(best_r) >= 0.1:
        verdict = "⚠️ 訊號模糊（0.1 ≤ |r| < 0.3）→ 建議補齊資料後再判斷"
        verdict_color = "#856404"
    else:
        verdict = "❌ 無明顯訊號（|r| < 0.1）→ 考慮換主題"
        verdict_color = "#842029"

    # 圖片
    def img(name, alt="", w="100%"):
        return img_tag(REPORTS_DIR / name, alt, f"max-width:{w};border-radius:6px;")

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoWoS 訊息強度 × 股價 Phase 1 報告</title>
<style>
  body {{ font-family: -apple-system, "PingFang TC", "Microsoft JhengHei", sans-serif;
          margin: 0; background: #f5f5f5; color: #1a1a1a; line-height: 1.7; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 40px 24px; }}
  h1 {{ font-size: 2rem; border-bottom: 3px solid #1565C0; padding-bottom: 12px; }}
  h2 {{ font-size: 1.4rem; color: #1565C0; margin-top: 48px; border-left: 4px solid #1565C0; padding-left: 12px; }}
  h3 {{ font-size: 1.1rem; color: #333; margin-top: 28px; }}
  .exec-box {{ background: #fff; border-left: 6px solid #1565C0;
               border-radius: 8px; padding: 24px 28px; margin: 24px 0;
               box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
  .verdict {{ font-size: 1.15rem; font-weight: bold; padding: 14px 20px;
              border-radius: 6px; margin: 16px 0; color: white;
              background: {verdict_color}; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: #fff;
           border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  th {{ background: #1565C0; color: white; padding: 10px 14px; text-align: left; font-size: .9rem; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #e8e8e8; font-size: .9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f0f4ff; }}
  .fig-wrap {{ background: #fff; border-radius: 8px; padding: 16px;
               box-shadow: 0 1px 4px rgba(0,0,0,.07); margin: 20px 0; }}
  .caption {{ font-size: .8rem; color: #666; margin-top: 8px; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f9a825;
           padding: 12px 16px; border-radius: 4px; font-size: .9rem; margin: 16px 0; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
          font-size: .8rem; background: #e3f2fd; color: #1565C0; margin: 2px; }}
  footer {{ margin-top: 60px; font-size: .8rem; color: #999; text-align: center; border-top: 1px solid #ddd; padding-top: 16px; }}
</style>
</head>
<body>
<div class="container">

<h1>CoWoS 訊息強度 × 相關概念股<br><small style="font-size:1rem;color:#555">Phase 1 分析報告</small></h1>

<p style="color:#666">
  分析期間：{start_date} ~ {end_date}　｜
  產出日期：{date.today()}　｜
  專案：news-stock-correlation v0.1
</p>

<!-- Executive Summary -->
<div class="exec-box">
  <h2 style="margin-top:0;border:none;padding:0">Executive Summary</h2>
  <div class="verdict">{verdict}</div>

  <p><strong>研究問題</strong>：CoWoS 先進封裝的訊息強度（Google Trends 搜尋熱度 + 新聞數量），
  是否與相關概念股的股價存在可觀察的時序關係？</p>

  <h3 style="margin-top:16px">主要發現</h3>
  <ul>
    <li><strong>辛耘（3583.TW）訊號最強</strong>：全期 r = +0.149（p &lt; 0.001），
        2025-07 後上升至 r = +0.239（p = 0.0004）</li>
    <li><strong>台積電無顯著相關</strong>（r = +0.073，p = 0.103）：體量過大，CoWoS 訊號被多重因子稀釋</li>
    <li><strong>Lead-Lag</strong>：辛耘、家登 lag ≤ 1 天（幾乎同步）；訊息不具明顯的超前預測能力</li>
    <li><strong>穩健性</strong>：訊號在三種 NII 權重、三個時間段均維持正相關方向；
        2025-07 後訊號更強，指向早期資料覆蓋不足為主因</li>
    <li><strong>週報酬率</strong>配 Trends 週頻後，家登（r = +0.196）、辛耘（r = +0.188）訊號進一步增強</li>
  </ul>

  <h3>資料限制</h3>
  <ul>
    <li>Google News RSS 僅保留近 12 個月，2024-05~2025-06 新聞數依賴 GDELT</li>
    <li>GDELT CoWoS 關鍵字在 2024-08 後受限流，覆蓋不完整</li>
    <li>相關不等於因果；未控制大盤共振效應（A7 留待 Phase 2）</li>
  </ul>
</div>

<!-- 訊息強度 NII -->
<h2>1. NII 訊息強度指數</h2>
<p>
  <code>NII = 0.5 × Trends_norm + 0.5 × NewsCount_zscore</code>
  <span class="tag">Trends</span><span class="tag">Google News RSS</span><span class="tag">GDELT</span>
</p>
<div class="fig-wrap">
  {img("eda_trends.png", "Google Trends")}
  <p class="caption">Google Trends CoWoS 搜尋熱度（台灣，週頻）。2024-06 出現全期最高峰（100），對應 CoWoS 擴產新聞集中期。</p>
</div>
<div class="fig-wrap">
  {img("eda_news.png", "每日新聞數")}
  <p class="caption">每日新聞數（Google News RSS + GDELT）。紅色陰影區為 Google News RSS 無歷史資料期間。</p>
</div>

<!-- A1 疊圖 -->
<h2>A1：NII × 個股 雙軸疊圖</h2>
<p>左軸（藍）為 NII；右軸（彩色）為個股收盤價。</p>
<div class="fig-wrap">
  {img("a1_overlay_3583_TW.png", "辛耘")}
  <p class="caption">辛耘（3583.TW）：訊號最強標的。2025-03 訊息低谷與股價低點同步；2026-04 後股價大漲但 NII 未同步拉升，
  指向該波漲勢由其他因素驅動。</p>
</div>
<div class="grid2">
  <div class="fig-wrap">{img("a1_overlay_3680_TWO.png", "家登")}<p class="caption">家登（3680.TWO）</p></div>
  <div class="fig-wrap">{img("a1_overlay_3413_TW.png", "京鼎")}<p class="caption">京鼎（3413.TW）</p></div>
  <div class="fig-wrap">{img("a1_overlay_3131_TWO.png", "弘塑")}<p class="caption">弘塑（3131.TWO）</p></div>
  <div class="fig-wrap">{img("a1_overlay_2330_TW.png", "台積電")}<p class="caption">台積電（2330.TW）— 幾乎無相關</p></div>
</div>

<!-- A2 相關係數 -->
<h2>A2：全期間皮爾森相關係數</h2>
<div class="note">以日報酬率計算，避免共同趨勢造成的偽相關（spurious correlation）</div>
<table>
  <tr><th>代號</th><th>名稱</th><th>皮爾森 r</th><th>p-value</th><th>樣本數</th><th>顯著</th></tr>
  {a2_rows}
</table>
<div class="fig-wrap">
  {img("a6_heatmap.png", "相關熱圖", "70%")}
  <p class="caption">A6：多標的相關係數熱圖（斜線格 = p ≥ 0.05，不顯著）</p>
</div>

<!-- A3 滾動相關 -->
<h2>A3：滾動相關係數（30 / 60 天）</h2>
<div class="fig-wrap">
  {img("a3_rolling_3583_TW.png", "辛耘滾動相關")}
  <p class="caption">辛耘（3583.TW）：大部分時間維持正相關，2024-06 前後相關較強，2025-01 出現短暫負相關。
  訊號不算穩定，但正相關方向持續。</p>
</div>
<div class="grid2">
  <div class="fig-wrap">{img("a3_rolling_3680_TWO.png", "家登")}<p class="caption">家登（3680.TWO）</p></div>
  <div class="fig-wrap">{img("a3_rolling_3413_TW.png", "京鼎")}<p class="caption">京鼎（3413.TW）</p></div>
</div>

<!-- A4 Lead-Lag -->
<h2>A4：Lead-Lag 分析（lag = -10 ~ +10 天）</h2>
<div class="fig-wrap">
  {img("a4_lag_3583_TW.png", "辛耘 lag")}
  <p class="caption">辛耘：最佳 lag = 0（NII 與股價同步），r = 0.149，95% Bootstrap CI = [0.047, 0.242]。
  lag 0~3 均顯著，代表 NII 在同日或未來 1~3 天內對股價有微弱關聯。</p>
</div>
<div class="grid2">
  <div class="fig-wrap">{img("a4_lag_3680_TWO.png", "家登 lag")}<p class="caption">家登：lag = +1 天（NII 領先 1 天）</p></div>
  <div class="fig-wrap">{img("a4_lag_2330_TW.png", "台積電 lag")}<p class="caption">台積電：lag = +5 天，但 CI 下緣趨近 0，不穩定</p></div>
</div>

<!-- A5 事件研究 -->
<h2>A5：事件研究 — NII 峰值日 ±10 天</h2>
<p>識別出 6 個 NII 高峰事件日（NII &gt; mean + 1.5σ）：
  2024-06-03、2024-06-28、2024-07-19、2024-10-14、2025-01-13、2026-04-20</p>
<div class="fig-wrap">
  {img("a5_event_3583_TW.png", "辛耘事件研究")}
  <p class="caption">辛耘：事件日後平均累積報酬呈正向，但個別事件差異大（灰色細線），
  代表效果不穩定、可能受事件性質影響。</p>
</div>
<div class="grid2">
  <div class="fig-wrap">{img("a5_event_3680_TWO.png", "家登事件研究")}<p class="caption">家登（3680.TWO）</p></div>
  <div class="fig-wrap">{img("a5_event_2330_TW.png", "台積電事件研究")}<p class="caption">台積電（2330.TW）</p></div>
</div>

<!-- 穩健性 -->
<h2>穩健性檢驗摘要</h2>
<div class="fig-wrap">
  {img("r1_weight_sensitivity.png", "權重敏感度")}
  <p class="caption">R1：三組 NII 權重（w=0.3/0.5/0.7）的相關係數。方向一致，偏重新聞數（w=0.3）時各標的 r 略高。</p>
</div>
<h3>R3：時間段切割</h3>
<div class="note">關鍵發現：2025-07 後（新聞覆蓋完整期）訊號顯著增強，辛耘 r 從 0.108 升至 0.239</div>
<table>
  <tr><th>標的</th><th>名稱</th><th>r（全期）</th><th>r（早期）</th><th>r（近期）</th></tr>
  {rb_rows}
</table>
<div class="grid2">
  <div class="fig-wrap">{img("r3_time_period.png", "時間段")}<p class="caption">R3：三個時段比較（藍色=顯著，灰色=不顯著）</p></div>
  <div class="fig-wrap">{img("r5_return_frequency.png", "報酬率頻率")}<p class="caption">R5：日報酬率 vs 週報酬率（週頻與 Trends 頻率對齊）</p></div>
</div>

<!-- 結論 -->
<h2>Phase 1 結論與建議</h2>
<div class="exec-box">
  <div class="verdict">{verdict}</div>

  <h3>建議下一步（Phase 2 預備）</h3>
  <ul>
    <li><strong>補齊 GDELT 資料</strong>：以修正後的 sleep 間隔重跑，填補 2024-08~2025-07 的 CoWoS 新聞缺漏</li>
    <li><strong>改用週報酬率配合 Trends 週頻</strong>：辛耘、家登、弘塑的 r 可望提升至 0.15~0.20</li>
    <li><strong>加入大盤對照（A7）</strong>：計算超額相關，排除大盤共振的影響</li>
    <li><strong>情緒分析（Phase 2）</strong>：目前只用新聞數量，加入正/負向情緒可提升 NII 精準度</li>
  </ul>

  <div class="note" style="margin-top:16px">
    ⚠️ <strong>相關不等於因果</strong>。本報告所有相關係數均為觀察性統計，
    受限於資料覆蓋期間與方法論選擇，不應直接用於投資決策。
  </div>
</div>

<footer>
  資料來源：Google Trends（pytrends）、Google News RSS（feedparser）、GDELT 2.0、Yahoo Finance（yfinance）<br>
  分析方法：皮爾森相關（日/週報酬率）、滾動相關、Lead-Lag、Bootstrap CI、事件研究（CAR）<br>
  產出：news-stock-correlation v0.1 | {date.today()}
</footer>

</div>
</body>
</html>"""

    out_path = REPORTS_DIR / "phase1_cowos_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"報告已輸出：{out_path}  ({out_path.stat().st_size // 1024} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="cowos")
    args = parser.parse_args()
    generate(args.topic)


if __name__ == "__main__":
    main()
