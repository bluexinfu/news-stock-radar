"""
題材熱度雷達 — Material Design 報告產生器（v2）

呈現市場熱度指標 v2：
  - 今日綜合熱度排名（含新聞型/影片型子分數、加速度、相位）
  - 每題材卡片：代表個股、新聞/影片數量、近期相位變化、
    綜合熱度走勢 + 股價對照（Chart.js 互動圖）
  - 相位說明 + 底部方法論說明
  - Google Material Design 風格（Roboto、卡片陰影、響應式）

⚠️ 所有指標均為觀察性，股價僅作對照，不構成投資建議。

用法（由 run_pipeline 呼叫）：
  generate_heat_report(heat_map, smi_map, display_names, output_path)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS_DIR = ROOT / "data" / "raw" / "news"
YT_DIR = ROOT / "data" / "raw" / "youtube"
CONFIG = ROOT / "config" / "topics.yaml"

PHASE = {
    "冷卻": {"emoji": "❄️", "color": "#546E7A", "bg": "#ECEFF1", "label": "冷卻",
            "desc": "關注度低且持平或下降，題材尚未受市場注意。"},
    "預熱": {"emoji": "🌡️", "color": "#EF6C00", "bg": "#FFF3E0", "label": "預熱",
            "desc": "關注度仍低，但開始往上走，題材正在醞釀。"},
    "發燒": {"emoji": "🔥", "color": "#C62828", "bg": "#FFEBEE", "label": "發燒",
            "desc": "關注度顯著偏高（超過均值＋1σ），題材成為焦點。"},
    "降溫": {"emoji": "📉", "color": "#1565C0", "bg": "#E3F2FD", "label": "降溫",
            "desc": "關注度仍高但開始回落，熱度從高峰消退。"},
}


def _accel_arrow(a: float) -> tuple[str, str]:
    if a > 0.5:
        return "▲", "#C62828"
    if a < -0.5:
        return "▼", "#1565C0"
    return "▬", "#9E9E9E"


def _load_stocks() -> dict[str, list[str]]:
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    out = {}
    for t, c in cfg.items():
        if not isinstance(c, dict):
            continue
        rel = c.get("related_stocks", {})
        prim = rel.get("primary", []) + rel.get("secondary", [])
        out[t] = [f"{s['ticker'].split('.')[0]} {s['name']}" for s in prim[:4]]
    return out


def _counts(topic: str) -> tuple[int, int]:
    n = v = 0
    try:
        n = len(pd.read_parquet(NEWS_DIR / f"{topic}_googlenews.parquet").drop_duplicates(subset=["url"]))
    except Exception:
        pass
    try:
        v = int(pd.read_parquet(YT_DIR / f"{topic}_youtube.parquet")["videos"].sum())
    except Exception:
        pass
    return n, v


def _last_phase_change(phases: pd.Series) -> tuple | None:
    p = phases.dropna()
    for i in range(len(p) - 1, 0, -1):
        if p.iloc[i] != p.iloc[i - 1]:
            return p.index[i], p.iloc[i - 1], p.iloc[i]
    return None


def _prep_topic(topic: str, df: pd.DataFrame, smi: pd.Series | None,
                stocks: list[str]) -> dict:
    d = df.dropna(subset=["composite"])
    dates = [x.strftime("%m/%d") for x in d.index]
    composite = [round(float(v), 1) for v in d["composite"]]
    price = None
    if smi is not None:
        s = smi.reindex(d.index)
        price = [None if pd.isna(v) else round(float(v), 1) for v in s]
    latest = d.iloc[-1]
    n_news, n_video = _counts(topic)
    chg = _last_phase_change(df["phase"])
    chg_str = None
    if chg:
        chg_str = f"{chg[1]}→{chg[2]} ({chg[0].strftime('%m/%d')})"
    return {
        "topic": topic, "dates": dates, "composite": composite, "price": price,
        "latest_composite": round(float(latest["composite"]), 1),
        "news": round(float(latest["news_norm"]), 1),
        "youtube": round(float(latest["youtube_norm"]), 1),
        "accel": round(float(latest["accel"]), 2) if pd.notna(latest["accel"]) else 0.0,
        "phase": latest["phase"], "stocks": stocks,
        "n_news": n_news, "n_video": n_video, "phase_change": chg_str,
    }


def generate_heat_report(heat_map: dict[str, pd.DataFrame],
                         smi_map: dict[str, pd.Series],
                         display_names: dict[str, str],
                         output_path: str | Path) -> Path:
    stocks_map = _load_stocks()
    rows = []
    for t, df in heat_map.items():
        if df.empty or df["composite"].dropna().empty:
            continue
        rows.append(_prep_topic(t, df, smi_map.get(t), stocks_map.get(t, [])))
    rows.sort(key=lambda r: -r["latest_composite"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["name"] = display_names.get(r["topic"], r["topic"])

    today = date.today().strftime("%Y-%m-%d")
    html = _render(rows, today)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def _render(rows: list[dict], today: str) -> str:
    table_rows = ""
    for r in rows:
        ph = PHASE.get(r["phase"], PHASE["冷卻"])
        arrow, acolor = _accel_arrow(r["accel"])
        table_rows += f"""
        <tr>
          <td class="rank">{r['rank']}</td>
          <td class="tname">{r['name']}</td>
          <td><span class="chip" style="background:{ph['bg']};color:{ph['color']}">{ph['emoji']} {ph['label']}</span></td>
          <td class="num"><b>{r['latest_composite']:.0f}</b></td>
          <td class="num news">{r['news']:.0f}</td>
          <td class="num yt">{r['youtube']:.0f}</td>
          <td class="num" style="color:{acolor}">{arrow} {abs(r['accel']):.1f}</td>
        </tr>"""

    cards = ""
    for r in rows:
        ph = PHASE.get(r["phase"], PHASE["冷卻"])
        arrow, acolor = _accel_arrow(r["accel"])
        stocks_html = " · ".join(f"<span class='stk'>{s}</span>" for s in r["stocks"])
        chg_html = (f"<span class='pchg'>🔄 {r['phase_change']}</span>"
                    if r["phase_change"] else "")
        cards += f"""
      <div class="card topic">
        <div class="topic-top">
          <div class="topic-name">{r['name']}</div>
          <span class="chip" style="background:{ph['bg']};color:{ph['color']}">{ph['emoji']} {ph['label']}</span>
        </div>
        <div class="stocks">{stocks_html}</div>
        <div class="metric-row">
          <div class="big">{r['latest_composite']:.0f}<span class="unit">/100</span></div>
          <div class="accel" style="color:{acolor}">{arrow} {abs(r['accel']):.1f}<div class="accel-cap">加速度</div></div>
        </div>
        <div class="bars">
          <div class="bar-row"><span class="bar-lab">📰 新聞型</span>
            <div class="bar"><div class="bar-fill news" style="width:{r['news']:.0f}%"></div></div>
            <span class="bar-val">{r['news']:.0f}</span></div>
          <div class="bar-row"><span class="bar-lab">▶️ 影片型</span>
            <div class="bar"><div class="bar-fill yt" style="width:{r['youtube']:.0f}%"></div></div>
            <span class="bar-val">{r['youtube']:.0f}</span></div>
        </div>
        <div class="counts">📰 {r['n_news']} 篇新聞 · ▶️ {r['n_video']} 部影片 {chg_html}</div>
        <div class="chart-wrap"><canvas id="chart-{r['topic']}"></canvas></div>
      </div>"""

    # 相位說明卡
    phase_legend = ""
    for k in ["冷卻", "預熱", "發燒", "降溫"]:
        p = PHASE[k]
        phase_legend += f"""
        <div class="legend-item" style="border-left:4px solid {p['color']}">
          <div class="legend-h">{p['emoji']} <b style="color:{p['color']}">{p['label']}</b></div>
          <div class="legend-d">{p['desc']}</div>
        </div>"""

    chart_data = json.dumps({r["topic"]: {"dates": r["dates"], "composite": r["composite"],
                                          "price": r["price"]} for r in rows}, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>題材熱度雷達 · {today}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{ --primary:#1565C0; --primary-d:#0D47A1; --surface:#fff; --bg:#F1F3F4;
    --on-surface:#202124; --muted:#5F6368; --news:#1E88E5; --yt:#E53935; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--on-surface);
    font-family:'Roboto','Noto Sans TC',-apple-system,sans-serif; }}
  .app-bar {{ background:linear-gradient(135deg,var(--primary),var(--primary-d));
    color:#fff; padding:20px 24px; box-shadow:0 2px 6px rgba(0,0,0,.2); position:sticky; top:0; z-index:10; }}
  .app-bar h1 {{ margin:0; font-size:22px; font-weight:700; letter-spacing:.5px; }}
  .app-bar .date {{ opacity:.9; font-size:13px; margin-top:2px; }}
  main {{ max-width:1100px; margin:0 auto; padding:20px 16px 60px; }}
  .disclaimer {{ background:#FFF8E1; border-left:4px solid #F9A825; color:#5F4300;
    padding:10px 14px; border-radius:6px; font-size:12.5px; margin-bottom:20px; }}
  .card {{ background:var(--surface); border-radius:14px; padding:18px 20px;
    box-shadow:0 1px 3px rgba(0,0,0,.12),0 1px 2px rgba(0,0,0,.08); margin-bottom:20px; }}
  .card h2 {{ margin:0 0 14px; font-size:17px; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ text-align:left; padding:9px 8px; border-bottom:1px solid #ECEFF1; }}
  th {{ color:var(--muted); font-weight:500; font-size:12px; }}
  td.num,th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  td.rank {{ color:var(--muted); width:28px; }}
  td.tname {{ font-weight:500; }}
  td.news {{ color:var(--news); }} td.yt {{ color:var(--yt); }}
  .chip {{ display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; white-space:nowrap; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:18px; }}
  .topic-top {{ display:flex; justify-content:space-between; align-items:center; }}
  .topic-name {{ font-size:16px; font-weight:700; }}
  .stocks {{ margin:6px 0 2px; font-size:11.5px; color:var(--muted); }}
  .stk {{ display:inline-block; background:#F1F3F4; border-radius:4px; padding:1px 6px; margin:2px 2px 0 0; }}
  .metric-row {{ display:flex; align-items:flex-end; justify-content:space-between; margin:8px 0 14px; }}
  .big {{ font-size:42px; font-weight:700; line-height:1; color:var(--primary); }}
  .big .unit {{ font-size:15px; color:var(--muted); font-weight:400; }}
  .accel {{ text-align:right; font-size:18px; font-weight:700; }}
  .accel-cap {{ font-size:10px; color:var(--muted); font-weight:400; }}
  .bars {{ margin-bottom:8px; }}
  .bar-row {{ display:flex; align-items:center; gap:8px; margin:5px 0; font-size:12px; }}
  .bar-lab {{ width:64px; color:var(--muted); }}
  .bar {{ flex:1; height:8px; background:#ECEFF1; border-radius:4px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:4px; }}
  .bar-fill.news {{ background:var(--news); }} .bar-fill.yt {{ background:var(--yt); }}
  .bar-val {{ width:26px; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }}
  .counts {{ font-size:11.5px; color:var(--muted); margin:4px 0 2px; }}
  .pchg {{ background:#EDE7F6; color:#5E35B1; border-radius:4px; padding:1px 6px; margin-left:4px; }}
  .chart-wrap {{ position:relative; height:165px; margin-top:8px; }}
  .legend-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }}
  .legend-item {{ background:#FAFAFA; border-radius:8px; padding:10px 12px; }}
  .legend-h {{ font-size:14px; margin-bottom:3px; }}
  .legend-d {{ font-size:12px; color:var(--muted); line-height:1.5; }}
  .method p {{ font-size:13px; line-height:1.8; color:#3C4043; margin:6px 0; }}
  .method b {{ color:var(--on-surface); }}
  .method .formula {{ background:#F1F3F4; border-radius:6px; padding:8px 12px; font-size:12.5px; margin:8px 0; }}
  footer {{ text-align:center; color:var(--muted); font-size:12px; padding:10px 16px 40px; }}
  @media(max-width:480px) {{ .app-bar h1{{font-size:19px;}} .big{{font-size:36px;}} }}
</style>
</head>
<body>
  <div class="app-bar">
    <h1>📡 題材熱度雷達</h1>
    <div class="date">{today} · 市場熱度指標 v2（權威新聞 + 影片觀看）</div>
  </div>
  <main>
    <div class="disclaimer">⚠️ 本報告衡量「題材關注熱度」，所有指標均為觀察性統計，股價僅作對照。相關不等於因果，不構成投資建議。</div>

    <div class="card">
      <h2>今日熱度排行</h2>
      <table>
        <thead><tr>
          <th>#</th><th>題材</th><th>相位</th>
          <th class="num">綜合</th><th class="num">新聞型</th><th class="num">影片型</th><th class="num">加速</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>

    <div class="grid">{cards}</div>

    <div class="card">
      <h2>📖 相位說明</h2>
      <div class="legend-grid">{phase_legend}</div>
    </div>

    <div class="card method">
      <h2>🔬 分析方法論</h2>
      <p>本系統衡量的是「<b>市場對題材的關注熱度</b>」，而非預測股價。熱度由兩個來源合成：</p>
      <p>① <b>新聞型關注</b>：各媒體每篇報導，依「來源可信度」加權（經濟日報、工商時報、MoneyDJ、DIGITIMES 等專業財經/半導體媒體權重高；內容農場、自動生成稿權重低）。</p>
      <p>② <b>影片型關注</b>：財經 YouTube 影片的觀看數加權（僅計訂閱數 ≥ 2 萬的頻道，並依頻道可信度分級，過濾明牌台與雜訊小頻道）。</p>
      <div class="formula">綜合熱度 = 新聞型關注 × 80% + 影片型關注 × 20%　（兩者皆先全域正規化到 0~100，避免尺度失衡）</div>
      <p><b>相位</b>：依熱度的相對高低（與該題材自身歷史均值±1σ 比較）與升降方向，分為冷卻／預熱／發燒／降溫四階段。<b>加速度</b>：熱度近期的變化速率（正值升溫、負值降溫）。</p>
      <p style="color:#C62828"><b>重要：</b>經歷史回測驗證，本熱度與股價多為「<b>同步</b>」關係而非「領先」——它是一支準確的「<b>題材熱度溫度計</b>」，可掌握當下哪個題材最受關注，但<b>不是預測股價的水晶球</b>，請勿單以熱度作為進出場依據。</p>
    </div>
  </main>
  <footer>題材熱度雷達 · 自動產生於 {today}<br>關注度 = 權威加權新聞 + YouTube 觀看加權（訂閱≥2萬）｜僅供研究參考，不構成投資建議</footer>

<script>
const DATA = {chart_data};
const PRIMARY = '#1565C0', PRICE = '#9E9E9E';
for (const [topic, d] of Object.entries(DATA)) {{
  const ctx = document.getElementById('chart-' + topic);
  if (!ctx) continue;
  const ds = [{{
    label:'綜合熱度', data:d.composite, borderColor:PRIMARY, backgroundColor:'rgba(21,101,192,.08)',
    borderWidth:2, fill:true, tension:.3, pointRadius:0, yAxisID:'y'
  }}];
  if (d.price && d.price.some(v=>v!==null)) ds.push({{
    label:'股價(對照)', data:d.price, borderColor:PRICE, borderWidth:1.5,
    borderDash:[4,3], fill:false, tension:.3, pointRadius:0, yAxisID:'y1'
  }});
  new Chart(ctx, {{
    type:'line', data:{{ labels:d.dates, datasets:ds }},
    options:{{
      responsive:true, maintainAspectRatio:false, interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{display:true, labels:{{boxWidth:12, font:{{size:11}}}}}} }},
      scales:{{
        x:{{ ticks:{{maxTicksLimit:6, font:{{size:10}}}}, grid:{{display:false}} }},
        y:{{ position:'left', title:{{display:true,text:'熱度',font:{{size:10}}}}, ticks:{{font:{{size:10}}}} }},
        y1:{{ position:'right', grid:{{display:false}}, title:{{display:true,text:'股價',font:{{size:10}}}}, ticks:{{font:{{size:10}}}} }}
      }}
    }}
  }});
}}
</script>
</body>
</html>"""
