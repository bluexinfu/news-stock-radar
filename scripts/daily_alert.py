#!/usr/bin/env python3
"""
題材雷達日報警報

每日管線結束後自動執行，偵測「相位轉換」並推播 Telegram 通知。

用法：
    python scripts/daily_alert.py                   # 正常執行
    python scripts/daily_alert.py --dry-run         # 只印訊息，不實際推播
    python scripts/daily_alert.py --force-send      # 無論是否有異動都發全報

環境變數（設定於 .env）：
    TELEGRAM_BOT_TOKEN  Bot Token（@BotFather 取得）
    TELEGRAM_CHAT_ID    你的 Chat ID

⚠️ 所有訊號均為觀察性指標，不構成投資建議。
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("daily_alert")


# ── 相位設定 ─────────────────────────────────────────────────────────

PHASE_EMOJI = {
    "冷卻": "❄️",
    "預熱": "🌡",
    "發燒": "🔥",
    "降溫": "📉",
}

# 值得推播的相位轉換（from → to）
# 註：經三方驗證，相位訊號與股價多為「同步」而非「領先」，
#     以下標籤一律描述「熱度變化」，不暗示進出場時機。
ALERT_TRANSITIONS = {
    ("冷卻", "預熱"): ("🌡️ 熱度翻揚", "NII 低檔回升，題材開始升溫"),
    ("降溫", "預熱"): ("🌡️ 二次翻揚", "前期降溫後熱度再度回升"),
    ("冷卻", "發燒"): ("🔥 急速升溫", "直接從冷卻跳至發燒，熱度快速放大"),
    ("預熱", "發燒"): ("🔥 確認發燒", "熱度突破均值＋1σ"),
    ("發燒", "降溫"): ("📉 高檔回落", "熱度從高檔反轉向下"),
    ("預熱", "降溫"): ("📉 升溫中止", "升溫失敗，熱度轉向走弱"),
    ("發燒", "冷卻"): ("❄️ 急速冷卻", "熱度快速消退"),
}


# ── 資料讀取（市場熱度 v2）─────────────────────────────────────────

def load_topic_heat(topic: str) -> pd.DataFrame | None:
    """讀取新熱度資料（已含 composite、news_norm、youtube_norm、accel、phase）。"""
    path = ROOT / "data" / "processed" / f"{topic}_heat.parquet"
    if not path.exists():
        log.debug("[%s] 無熱度資料（%s）", topic, path)
        return None
    df = pd.read_parquet(path).dropna(subset=["composite"])
    return df if len(df) >= 2 else None


def load_topics_yaml() -> dict:
    with open(ROOT / "config" / "topics.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 訊息格式化 ────────────────────────────────────────────────────────

def build_message(
    transitions: list[dict],
    all_status: list[dict],
    today: date,
    report_filename: str | None = None,
) -> str:
    # REPORT_BASE_URL：GitHub Actions 傳入公開網址；本機預設 localhost
    import os
    base_url = os.getenv("REPORT_BASE_URL", "http://localhost:8080").rstrip("/")
    report_url = f"{base_url}/{report_filename}" if report_filename else base_url

    lines = [
        f"<b>📡 題材熱度雷達 {today.strftime('%Y-%m-%d')}</b>",
        f'🔗 <a href="{report_url}">開啟完整報告</a>',
        "",
    ]

    if transitions:
        lines.append("⚡ <b>相位異動</b>")
        for t in transitions:
            label, note = ALERT_TRANSITIONS.get(
                (t["from_phase"], t["to_phase"]),
                ("🔄 相位變化", "")
            )
            lines.append(
                f"  {label}：{t['display_name']}\n"
                f"  {PHASE_EMOJI[t['from_phase']]} {t['from_phase']} → "
                f"{PHASE_EMOJI[t['to_phase']]} {t['to_phase']}"
            )
            if note:
                lines.append(f"  <i>({note})</i>")
        lines.append("")

    lines.append("📋 <b>今日各題材熱度</b>")
    for s in all_status:
        a = s["accel"]
        arrow = "▲" if a > 0.5 else ("▼" if a < -0.5 else "▬")
        lines.append(
            f"  {PHASE_EMOJI[s['phase']]} {s['display_name']}"
            f"  熱度={s['composite']:.0f}  {arrow}{abs(a):.1f}"
        )

    lines += [
        "",
        "⚠️ <i>僅供參考，不構成投資建議</i>",
    ]
    return "\n".join(lines)


# ── 主邏輯 ───────────────────────────────────────────────────────────

def run(dry_run: bool = False, force_send: bool = False, report_filename: str | None = None) -> None:
    topics_cfg = load_topics_yaml()

    transitions: list[dict] = []
    all_status: list[dict] = []

    for topic, cfg in topics_cfg.items():
        display_name = cfg.get("display_name", topic)
        heat = load_topic_heat(topic)
        if heat is None:
            log.warning("[%s] 無法讀取熱度資料，跳過", topic)
            continue

        today_phase = str(heat["phase"].iloc[-1])
        yesterday_phase = str(heat["phase"].iloc[-2])
        latest = heat.iloc[-1]
        stats = {
            "composite": round(float(latest["composite"]), 1),
            "accel": round(float(latest["accel"]), 2) if pd.notna(latest["accel"]) else 0.0,
            "news": round(float(latest["news_norm"]), 1),
            "youtube": round(float(latest["youtube_norm"]), 1),
        }

        log.info("[%s] %s → %s  熱度=%.1f  加速=%.2f",
                 topic, yesterday_phase, today_phase, stats["composite"], stats["accel"])

        all_status.append({
            "topic": topic, "display_name": display_name,
            "phase": today_phase, **stats,
        })

        if today_phase != yesterday_phase:
            transitions.append({
                "topic": topic, "display_name": display_name,
                "from_phase": yesterday_phase, "to_phase": today_phase, **stats,
            })

    # 優先顯示：預熱 > 發燒 > 降溫 > 冷卻
    _priority = {"預熱": 1, "發燒": 2, "降溫": 3, "冷卻": 4}
    all_status.sort(key=lambda x: (_priority.get(x["phase"], 5), -x["composite"]))

    should_send = force_send or bool(transitions)

    # 若未傳入，自動找今天的報告檔名
    if report_filename is None:
        today_str = date.today().strftime("%Y%m%d")
        report_path = ROOT / "reports" / f"radar_{today_str}.html"
        report_filename = report_path.name if report_path.exists() else None

    message = build_message(transitions, all_status, date.today(), report_filename)

    print("\n" + "=" * 60)
    print(message)
    print("=" * 60 + "\n")

    if dry_run:
        log.info("--dry-run：跳過實際推播")
        return

    if not should_send:
        log.info("今日無相位異動，跳過推播（使用 --force-send 強制發送）")
        return

    from src.notifiers.telegram import send

    # 發送文字摘要（連結已內嵌在訊息中）
    ok = send(message)
    if ok:
        log.info("✅ Telegram 推播完成")
    else:
        log.warning("⚠️ Telegram 推播失敗（請確認 .env 設定）")


# ── CLI 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="題材雷達日報警報")
    parser.add_argument("--dry-run", action="store_true",
                        help="只印訊息，不實際推播")
    parser.add_argument("--force-send", action="store_true",
                        help="無論是否有異動都發送完整日報")
    args = parser.parse_args()
    run(dry_run=args.dry_run, force_send=args.force_send)
