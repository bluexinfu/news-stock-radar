#!/usr/bin/env python3
"""
Telegram Bot 指令監聽器
每 5 分鐘由 GitHub Actions 執行一次，輪詢 Bot 收件匣。

支援指令：
  /run    — 立即觸發日報管線（約 20 分鐘後收到報告）
  /status — 查詢管線執行狀態
  /help   — 顯示可用指令

環境變數（由 GitHub Actions 注入）：
  TELEGRAM_BOT_TOKEN   Bot Token
  TELEGRAM_CHAT_ID     授權的 Chat ID（只回應此 chat）
  GH_TOKEN             GitHub Token（用於觸發 workflow）
  GITHUB_REPOSITORY    repo 全名，例如 bluexinfu/news-stock-radar
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── 設定 ──────────────────────────────────────────────────────────────
TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID  = str(os.getenv("TELEGRAM_CHAT_ID", ""))
GH_TOKEN = os.getenv("GH_TOKEN", "")
REPO     = os.getenv("GITHUB_REPOSITORY", "")   # "bluexinfu/news-stock-radar"

OFFSET_FILE = Path("_data_cache/telegram_offset.txt")

BASE_TG = f"https://api.telegram.org/bot{TOKEN}"
GH_HEADERS = {
    "Authorization": f"Bearer {GH_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
}


# ── Telegram 工具函式 ─────────────────────────────────────────────────

def tg_request(method: str, params: dict | None = None, post_data: dict | None = None):
    url = f"{BASE_TG}/{method}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(post_data).encode() if post_data else None
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def tg_send(text: str) -> None:
    """發送 HTML 格式訊息到授權 chat。"""
    try:
        tg_request("sendMessage", post_data={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        })
    except Exception as e:
        print(f"[WARN] Telegram 發送失敗：{e}")


# ── GitHub 工具函式 ───────────────────────────────────────────────────

def gh_get(path: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers=GH_HEADERS,
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def gh_post(path: str, body: dict) -> bool:
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=json.dumps(body).encode(),
        headers=GH_HEADERS,
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=15)
        return True
    except urllib.error.HTTPError as e:
        print(f"[ERROR] GitHub API {path}: {e.code} — {e.read().decode()}")
        return False


# ── 指令處理 ─────────────────────────────────────────────────────────

def handle_run() -> None:
    """觸發 daily_pipeline.yml。"""
    ok = gh_post(
        f"/repos/{REPO}/actions/workflows/daily_pipeline.yml/dispatches",
        {"ref": "main"},
    )
    if ok:
        tg_send(
            "🚀 <b>日報管線已啟動！</b>\n"
            "⏳ 約 20 分鐘後會收到完整報告通知。\n\n"
            f'🔍 <a href="https://github.com/{REPO}/actions">查看執行進度</a>'
        )
        print("✅ 已觸發 daily_pipeline.yml")
    else:
        tg_send(
            "❌ <b>啟動失敗</b>\n"
            f'請至 <a href="https://github.com/{REPO}/actions">GitHub Actions</a> 手動觸發。'
        )


def handle_status() -> None:
    """查詢最近一次 daily_pipeline.yml 的執行狀態。"""
    try:
        data = gh_get(
            f"/repos/{REPO}/actions/workflows/daily_pipeline.yml/runs?per_page=1"
        )
        runs = data.get("workflow_runs", [])
        if not runs:
            tg_send("ℹ️ 尚無執行紀錄。")
            return

        run = runs[0]
        status     = run["status"]       # queued / in_progress / completed
        conclusion = run["conclusion"]   # success / failure / cancelled / None
        run_url    = run["html_url"]
        created    = run["created_at"][:16].replace("T", " ")  # 2026-05-27 12:30

        if status == "in_progress":
            msg = (
                f"🔄 <b>管線執行中</b>\n"
                f"⏱ 開始時間：{created} UTC\n"
                f'🔍 <a href="{run_url}">查看進度</a>'
            )
        elif status == "queued":
            msg = f"⏳ <b>管線排隊中</b>（{created} UTC）"
        elif conclusion == "success":
            msg = (
                f"✅ <b>上次執行成功</b>\n"
                f"📅 {created} UTC\n"
                f'🔍 <a href="{run_url}">查看結果</a>'
            )
        elif conclusion == "failure":
            msg = (
                f"❌ <b>上次執行失敗</b>\n"
                f"📅 {created} UTC\n"
                f'🔍 <a href="{run_url}">查看錯誤</a>'
            )
        elif conclusion == "cancelled":
            msg = f"🚫 上次執行已取消（{created} UTC）"
        else:
            msg = f"ℹ️ 狀態：{status} / {conclusion}（{created} UTC）"

    except Exception as e:
        msg = f"⚠️ 無法查詢狀態：{e}"

    tg_send(msg)


def handle_help() -> None:
    tg_send(
        "📋 <b>可用指令</b>\n\n"
        "/run — 立即執行日報（約 20 分鐘後收到報告）\n"
        "/status — 查詢管線執行狀態\n"
        "/help — 顯示此說明\n\n"
        "⚠️ <i>僅限授權使用者操作</i>"
    )


# ── 主流程 ───────────────────────────────────────────────────────────

def main() -> None:
    if not TOKEN or not CHAT_ID:
        print("[ERROR] 缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，跳過")
        sys.exit(0)

    # 讀取上次的 offset（避免重複處理舊訊息）
    offset = 0
    if OFFSET_FILE.exists():
        try:
            offset = int(OFFSET_FILE.read_text().strip()) + 1
        except ValueError:
            offset = 0
    print(f"ℹ️  從 offset={offset} 開始輪詢")

    # 取得新訊息（最多 20 則）
    result = tg_request("getUpdates", params={"offset": offset, "limit": 20, "timeout": 0})
    updates = result.get("result", [])
    print(f"ℹ️  收到 {len(updates)} 則更新")

    last_id = offset - 1
    run_triggered = False  # 同一輪只觸發一次管線

    for update in updates:
        last_id = update["update_id"]
        msg      = update.get("message", {})
        from_id  = str(msg.get("chat", {}).get("id", ""))
        raw_text = (msg.get("text") or "").strip()
        cmd      = raw_text.split()[0].lower() if raw_text else ""

        # 只回應授權的 chat
        if from_id != CHAT_ID:
            print(f"[SKIP] 忽略來自 chat_id={from_id} 的訊息（非授權）")
            continue

        print(f"[CMD] 收到：{raw_text!r}")

        if cmd in ("/run", "/run@" + REPO.split("/")[-1].lower()):
            if not run_triggered:
                handle_run()
                run_triggered = True
            else:
                tg_send("ℹ️ 管線已在本輪啟動，請稍候。")
        elif cmd in ("/status",):
            handle_status()
        elif cmd in ("/help", "/start"):
            handle_help()
        else:
            tg_send(
                f"❓ 不認識的指令：<code>{raw_text}</code>\n"
                "輸入 /help 查看可用指令。"
            )

    # 儲存最新 offset
    if last_id >= 0:
        OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
        OFFSET_FILE.write_text(str(last_id))
        print(f"✅ offset 已更新：{last_id}")
    else:
        print("ℹ️  無新訊息，offset 不變")


if __name__ == "__main__":
    main()
