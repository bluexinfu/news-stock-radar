"""
題材雷達 HTML 報告產生器

生成包含以下內容的單一 HTML 檔案（所有圖表 base64 嵌入）：
  - 題材熱度排行榜（表格，依「預熱優先 + 斜率」排序）
  - 每個主題的 NII 走勢迷你圖（spark line）
  - 「目前最值得關注」高亮標記（預熱階段）
  - 每個題材的概念股 SMI 走勢（供對照）

⚠️ 所有指標均為觀察性統計，相關不等於因果。
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import date
from pathlib import Path

import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang TC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

log = logging.getLogger(__name__)

# ── 顏色設定 ─────────────────────────────────────────────────────────
NII_COLOR = "#1565C0"
SMI_COLOR = "#6A1B9A"

PHASE_COLOR = {
    "冷卻": "#90A4AE",
    "預熱": "#FF8F00",
    "發燒": "#C62828",
    "降溫": "#1976D2",
}
PHASE_BG = {
    "冷卻": "#ECEFF1",
    "預熱": "#FFF8E1",
    "發燒": "#FFEBEE",
    "降溫": "#E3F2FD",
}

CHIPS_PHASE_COLOR = {
    "法人強買": "#1B5E20",
    "法人買進": "#388E3C",
    "中性":     "#757575",
    "法人賣出": "#E53935",
    "法人強賣": "#B71C1C",
}


# ── 圖表產生 ─────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _sparkline(
    nii: pd.Series,
    title: str,
    phase: str,
    height_inches: float = 2.0,
) -> str:
    """NII 走勢迷你圖，回傳 base64 PNG。"""
    fig, ax = plt.subplots(figsize=(7, height_inches))

    color = PHASE_COLOR.get(phase, NII_COLOR)
    ax.fill_between(nii.index, nii, alpha=0.18, color=NII_COLOR)
    ax.plot(nii.index, nii, color=NII_COLOR, linewidth=1.5)

    mu    = nii.mean()
    sigma = nii.std()
    ax.axhline(mu,          color="gray",  linewidth=0.7, linestyle="--", label=f"均值 {mu:.1f}")
    ax.axhline(mu + sigma,  color="#C62828", linewidth=0.6, linestyle=":", alpha=0.8, label=f"+1σ {mu+sigma:.1f}")

    # 高亮最新值
    last_date = nii.index[-1]
    last_val  = float(nii.iloc[-1])
    ax.scatter([last_date], [last_val], color=color, s=40, zorder=5)
    ax.annotate(f"{last_val:.1f}", (last_date, last_val),
                textcoords="offset points", xytext=(6, 4), fontsize=8, color=color)

    ax.set_ylabel("NII", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    plt.title(f"NII 走勢：{title}", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


def _smi_chart(
    smi: pd.Series,
    nii: pd.Series,
    title: str,
    height_inches: float = 2.0,
) -> str:
    """SMI 等權指數（基期=100）+ NII 雙軸圖，回傳 base64 PNG。"""
    smi_level = (1 + smi.fillna(0)).cumprod() * 100

    fig, ax1 = plt.subplots(figsize=(7, height_inches))

    ax1.plot(nii.index, nii, color=NII_COLOR, linewidth=1.2, alpha=0.55, label="NII")
    ax1.set_ylabel("NII", color=NII_COLOR, fontsize=8)
    ax1.tick_params(axis="y", labelcolor=NII_COLOR, labelsize=7)

    ax2 = ax1.twinx()
    ax2.plot(smi_level.index, smi_level, color=SMI_COLOR, linewidth=1.8, label="SMI")
    ax2.set_ylabel("SMI（=100 基期）", color=SMI_COLOR, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=SMI_COLOR, labelsize=7)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y/%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=20, fontsize=7)
    plt.title(f"SMI 等權指數 vs NII：{title}", fontsize=9)
    plt.tight_layout()
    return _fig_to_b64(fig)


# ── HTML 組裝 ─────────────────────────────────────────────────────────

def _row_html(row: pd.Series, nii_b64: str, smi_b64: str | None) -> str:
    phase    = row["phase"]
    bg       = PHASE_BG.get(phase, "#fff")
    fc       = PHASE_COLOR.get(phase, "#333")
    emoji    = row.get("phase_emoji", "")
    name     = row["display_name"]
    z        = row["nii_zscore"]
    slope7   = row["nii_7d_slope"]
    pct30    = row["nii_30d_pct_chg"]

    slope_arrow = "▲" if slope7 > 0 else "▼"
    slope_color = "#C62828" if slope7 > 0 else "#1976D2"

    smi_section = (
        f'<img src="data:image/png;base64,{smi_b64}" style="max-width:100%">'
        if smi_b64 else '<p style="color:gray;font-size:12px">SMI 資料尚未生成</p>'
    )

    return f"""
    <div style="border:1px solid #ddd; border-radius:8px; margin:12px 0;
                background:{bg}; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,.08)">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px">
        <div>
          <span style="font-size:22px; font-weight:700; color:{fc}">{emoji} {name}</span>
          <span style="margin-left:12px; padding:2px 8px; border-radius:12px;
                       background:{fc}; color:#fff; font-size:12px; font-weight:600">{phase}</span>
        </div>
        <div style="text-align:right; font-size:13px; color:#555">
          NII <b>{row['nii_latest']}</b>（z={z:+.2f}）
          <span style="color:{slope_color}">{slope_arrow} 7d斜率 {slope7:+.3f}</span>
          30d變化 <b>{pct30:+.1f}%</b>
        </div>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
        <div>
          <img src="data:image/png;base64,{nii_b64}" style="max-width:100%">
        </div>
        <div>{smi_section}</div>
      </div>
    </div>
    """


def _build_table_html(
    radar_df: pd.DataFrame,
    topic_chips_map: dict[str, pd.DataFrame] | None = None,
) -> str:
    from src.processors.chips_signal import chips_phase

    rows_html = ""
    for _, row in radar_df.iterrows():
        phase  = row["phase"]
        fc     = PHASE_COLOR.get(phase, "#333")
        emoji  = row.get("phase_emoji", "")
        slope7 = row["nii_7d_slope"]
        arrow  = "▲" if slope7 > 0 else "▼"
        sc     = "#C62828" if slope7 > 0 else "#1976D2"

        # Chips 欄位
        topic = row["topic"]
        chips_cell = "<td style='text-align:center; color:#aaa; font-size:11px'>—</td>"
        if topic_chips_map and topic in topic_chips_map:
            chips_df = topic_chips_map[topic]
            if not chips_df.empty and "composite_chips" in chips_df.columns:
                composite = chips_df["composite_chips"].dropna()
                if not composite.empty:
                    cv = float(composite.iloc[-1])
                    cphase = chips_phase(cv)
                    cc = CHIPS_PHASE_COLOR.get(cphase, "#757575")
                    chips_cell = (
                        f"<td style='text-align:center; color:{cc}; "
                        f"font-weight:600; font-size:12px'>{cphase}</td>"
                    )

        rows_html += f"""
        <tr>
          <td style="text-align:center">{int(row['rank'])}</td>
          <td><b>{row['display_name']}</b></td>
          <td style="color:{fc}; font-weight:600">{emoji} {phase}</td>
          <td style="text-align:right">{row['nii_latest']}</td>
          <td style="text-align:right">{row['nii_zscore']:+.2f}</td>
          <td style="text-align:right; color:{sc}">{arrow} {slope7:+.3f}</td>
          <td style="text-align:right">{row['nii_30d_pct_chg']:+.1f}%</td>
          {chips_cell}
        </tr>"""
    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px">
      <thead>
        <tr style="background:#37474F; color:#fff">
          <th style="padding:8px">#</th>
          <th style="padding:8px; text-align:left">主題</th>
          <th style="padding:8px; text-align:left">NII階段</th>
          <th style="padding:8px; text-align:right">NII</th>
          <th style="padding:8px; text-align:right">z分數</th>
          <th style="padding:8px; text-align:right">7d斜率</th>
          <th style="padding:8px; text-align:right">30d漲跌</th>
          <th style="padding:8px; text-align:center">籌碼</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ── 主函式 ───────────────────────────────────────────────────────────

def generate_radar(
    radar_df: pd.DataFrame,
    topic_nii_map: dict[str, pd.Series],
    topic_smi_map: dict[str, pd.Series],
    display_names: dict[str, str],
    output_path: Path,
    topic_chips_map: dict[str, pd.DataFrame] | None = None,
) -> None:
    """
    生成題材雷達 HTML 報告。

    Parameters
    ----------
    radar_df         : rank_themes() 的回傳值
    topic_nii_map    : {topic: NII Series}
    topic_smi_map    : {topic: SMI 日報酬率 Series}
    display_names    : {topic: "顯示名稱"}
    output_path      : 輸出 HTML 路徑
    topic_chips_map  : {topic: chips_signal DataFrame}（選用，含籌碼訊號）
    """
    today = date.today().strftime("%Y-%m-%d")

    # ── 亮點：預熱主題 ────────────────────────────────────────────────
    warming_topics = radar_df[radar_df["phase"] == "預熱"]
    if warming_topics.empty:
        highlight_html = '<p style="color:#888">目前無主題處於「預熱」階段</p>'
    else:
        cards = ""
        for _, row in warming_topics.iterrows():
            cards += f"""
            <span style="display:inline-block; margin:4px 6px; padding:6px 14px;
                         background:#FF8F00; color:#fff; border-radius:20px;
                         font-weight:600; font-size:14px">
              🌡️ {row['display_name']}
              <small>（7d斜率 {row['nii_7d_slope']:+.3f}）</small>
            </span>"""
        highlight_html = cards

    # ── 排行表 ────────────────────────────────────────────────────────
    table_html = _build_table_html(radar_df, topic_chips_map)

    # ── 各主題詳情卡片 ────────────────────────────────────────────────
    detail_cards = ""
    for _, row in radar_df.iterrows():
        topic = row["topic"]
        nii   = topic_nii_map.get(topic)
        smi   = topic_smi_map.get(topic)

        if nii is None or nii.dropna().empty:
            continue

        nii_b64 = _sparkline(nii.dropna(), row["display_name"], row["phase"])
        smi_b64 = None
        if smi is not None and not smi.dropna().empty:
            smi_b64 = _smi_chart(smi, nii.dropna(), row["display_name"])

        detail_cards += _row_html(row, nii_b64, smi_b64)

    # ── 組合 HTML ─────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>題材熱度雷達 — {today}</title>
  <style>
    body {{ font-family: "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", Arial, sans-serif;
           max-width: 1100px; margin: 0 auto; padding: 24px; background: #FAFAFA; color: #212121; }}
    h1   {{ color: #1565C0; margin-bottom: 4px; }}
    h2   {{ color: #37474F; border-bottom: 2px solid #90A4AE; padding-bottom: 6px; margin-top: 28px; }}
    .disclaimer {{ font-size:11px; color:#aaa; margin-top:4px; }}
    .legend {{ display:flex; gap:16px; font-size:13px; margin:12px 0; flex-wrap:wrap; }}
    .legend span {{ padding:3px 10px; border-radius:12px; font-weight:600; }}
  </style>
</head>
<body>
  <h1>📡 題材熱度雷達</h1>
  <p style="color:#555; font-size:14px">產生日期：{today} ｜ 共 {len(radar_df)} 個主題</p>
  <p class="disclaimer">⚠️ 本報告所有指標均為觀察性統計，相關不等於因果。NII 為新聞強度指數，不構成任何投資建議。</p>

  <div class="legend">
    <span style="background:#ECEFF1; color:#546E7A">❄️ 冷卻：NII低且下行</span>
    <span style="background:#FFF8E1; color:#E65100">🌡️ 預熱：NII低但上行（早期訊號）</span>
    <span style="background:#FFEBEE; color:#B71C1C">🔥 發燒：NII高</span>
    <span style="background:#E3F2FD; color:#0D47A1">📉 降溫：NII高但下行</span>
  </div>

  <details style="margin:16px 0; background:#fff; border:1px solid #ddd; border-radius:8px; padding:4px 16px">
    <summary style="cursor:pointer; font-size:14px; font-weight:600; color:#37474F; padding:10px 0">
      💡 什麼是「相位」？（點擊展開說明）
    </summary>
    <div style="padding:8px 0 16px 0; font-size:13px; color:#444; line-height:1.8">
      <p style="margin:0 0 12px 0">
        <b>NII（新聞強度指數）</b>衡量某個投資題材在媒體上的熱度，綜合新聞篇數、Google 搜尋量等資料計算得出。
        「相位」就是根據 NII 目前的<b>絕對高低</b>與<b>變化方向</b>，判斷這個題材目前處於哪個階段。
      </p>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px">
        <div style="background:#ECEFF1; border-radius:8px; padding:12px">
          <div style="font-size:18px; margin-bottom:4px">❄️ <b style="color:#546E7A">冷卻</b></div>
          <div><b>消息面安靜</b>，媒體很少報導這個題材，搜尋量也低。</div>
          <div style="margin-top:6px; color:#777">→ 市場還沒關注，通常是等待期</div>
        </div>
        <div style="background:#FFF8E1; border-radius:8px; padding:12px">
          <div style="font-size:18px; margin-bottom:4px">🌡️ <b style="color:#E65100">預熱</b></div>
          <div><b>消息面開始升溫</b>，新聞量與搜尋量從低點往上走，但還不算多。</div>
          <div style="margin-top:6px; color:#777">→ 市場開始關注，股價可能還沒完全反應，屬於<b>早期訊號</b></div>
        </div>
        <div style="background:#FFEBEE; border-radius:8px; padding:12px">
          <div style="font-size:18px; margin-bottom:4px">🔥 <b style="color:#B71C1C">發燒</b></div>
          <div><b>消息面非常熱絡</b>，大量新聞與報導湧現，題材成為市場焦點。</div>
          <div style="margin-top:6px; color:#777">→ 熱度高峰，股價通常已充分反應，需留意追高風險</div>
        </div>
        <div style="background:#E3F2FD; border-radius:8px; padding:12px">
          <div style="font-size:18px; margin-bottom:4px">📉 <b style="color:#0D47A1">降溫</b></div>
          <div><b>消息面退燒</b>，熱度從高點往下走，媒體關注度下降。</div>
          <div style="margin-top:6px; color:#777">→ 題材熱潮消退，可能是考慮減碼的時機</div>
        </div>
      </div>
      <p style="margin:12px 0 0 0; color:#888; font-size:12px">
        ⚠️ 相位反映的是「消息面熱度週期」，不直接等於股價漲跌，投資決策仍需結合基本面與籌碼面綜合判斷。
      </p>
    </div>
  </details>

  <h2>⚡ 目前最值得關注（預熱中）</h2>
  <div>{highlight_html}</div>

  <h2>📊 熱度排行榜</h2>
  {table_html}

  <h2>🔍 各主題詳情</h2>
  {detail_cards}

  <hr style="margin-top:40px; border-color:#ddd">
  <p style="font-size:11px; color:#aaa; text-align:center">
    資料來源：Google Trends + Google News RSS + GDELT + Yahoo Finance<br>
    ⚠️ 本報告僅供研究與學習用途，不構成投資建議
  </p>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    log.info("HTML 報告已寫入：%s（%.1f KB）", output_path, len(html) / 1024)
