#!/usr/bin/env python3
"""
向 Telegram 登記 Webhook URL（只需執行一次）

用法：
    python scripts/register_webhook.py <WORKER_URL>

範例：
    python scripts/register_webhook.py https://radar-bot.yourname.workers.dev
"""

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_env():
    env_path = ROOT / ".env"
    values = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def main():
    if len(sys.argv) < 2:
        print("用法：python scripts/register_webhook.py <WORKER_URL>")
        print("範例：python scripts/register_webhook.py https://radar-bot.yourname.workers.dev")
        sys.exit(1)

    worker_url = sys.argv[1].rstrip("/")
    env = load_env()
    token = env.get("TELEGRAM_BOT_TOKEN", "")

    if not token:
        print("❌ 找不到 TELEGRAM_BOT_TOKEN，請確認 .env 檔案")
        sys.exit(1)

    print(f"\n登記 Webhook URL：{worker_url}")
    print(f"Bot Token：{token[:10]}***\n")

    # 先刪除舊的 webhook（確保乾淨）
    delete_url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    req = urllib.request.Request(delete_url, method="GET")
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())
        print(f"刪除舊 Webhook：{result.get('description', result)}")

    # 登記新的 webhook
    set_url = f"https://api.telegram.org/bot{token}/setWebhook"
    data = json.dumps({"url": worker_url}).encode()
    req = urllib.request.Request(
        set_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())

    if result.get("ok"):
        print(f"✅ Webhook 登記成功！")
        print(f"   URL：{worker_url}")
        print(f"\n現在傳訊息給 Bot 就會立刻收到回覆（無延遲）")
    else:
        print(f"❌ 登記失敗：{result.get('description', result)}")
        sys.exit(1)

    # 確認目前 webhook 狀態
    info_url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    req = urllib.request.Request(info_url)
    with urllib.request.urlopen(req) as r:
        info = json.loads(r.read()).get("result", {})
    print(f"\nWebhook 狀態確認：")
    print(f"  URL：{info.get('url', '(未設定)')}")
    print(f"  待處理訊息數：{info.get('pending_update_count', 0)}")
    if info.get("last_error_message"):
        print(f"  最近錯誤：{info['last_error_message']}")


if __name__ == "__main__":
    main()
