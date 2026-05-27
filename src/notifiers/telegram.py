"""
Telegram Bot 推播模組

快速設定（2 分鐘）：
  1. 在 Telegram 搜尋 @BotFather，輸入 /newbot，取得 TELEGRAM_BOT_TOKEN
  2. 對你的 Bot 發送任意一則訊息
  3. 瀏覽 https://api.telegram.org/bot<TOKEN>/getUpdates
     找到 "chat" → "id" 的數字，即為 TELEGRAM_CHAT_ID
  4. 在 .env 填入：
       TELEGRAM_BOT_TOKEN=your_token
       TELEGRAM_CHAT_ID=your_chat_id

使用方式：
    from src.notifiers.telegram import send
    send("你的訊息")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"
_SEND_DOC_URL = "https://api.telegram.org/bot{token}/sendDocument"


def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def send(message: str, token: str | None = None, chat_id: str | None = None) -> bool:
    """
    透過 Telegram Bot API 發送訊息。

    Parameters
    ----------
    message : str
        要發送的文字（支援 HTML 標籤，如 <b>粗體</b>、<i>斜體</i>）
    token : str, optional
        Bot Token，未提供則從環境變數 TELEGRAM_BOT_TOKEN 讀取
    chat_id : str, optional
        聊天 ID，未提供則從環境變數 TELEGRAM_CHAT_ID 讀取

    Returns
    -------
    bool
        True = 成功，False = 失敗
    """
    _load_dotenv()

    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or token.startswith("your_"):
        log.warning("TELEGRAM_BOT_TOKEN 未設定，跳過推播（請參考 .env.example）")
        return False
    if not chat_id or chat_id.startswith("your_"):
        log.warning("TELEGRAM_CHAT_ID 未設定，跳過推播")
        return False

    try:
        import requests
    except ImportError:
        log.error("requests 未安裝，請執行：pip install requests")
        return False

    try:
        resp = requests.post(
            _SEND_URL.format(token=token),
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            log.info("Telegram 推播成功")
            return True
        else:
            log.warning("Telegram 回傳錯誤：%s", data.get("description", resp.text))
            return False
    except Exception as e:
        log.error("Telegram 推播失敗：%s", e)
        return False


def send_document(
    file_path: str,
    caption: str = "",
    token: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """
    透過 Telegram Bot API 發送文件（如 HTML 報告）。

    Parameters
    ----------
    file_path : str
        本地檔案路徑
    caption : str
        附帶說明文字（支援 HTML）
    token : str, optional
        Bot Token
    chat_id : str, optional
        聊天 ID

    Returns
    -------
    bool
        True = 成功
    """
    _load_dotenv()
    import requests as req

    token = token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or token.startswith("your_"):
        log.warning("TELEGRAM_BOT_TOKEN 未設定，跳過檔案傳送")
        return False
    if not chat_id or chat_id.startswith("your_"):
        log.warning("TELEGRAM_CHAT_ID 未設定，跳過檔案傳送")
        return False

    from pathlib import Path
    fp = Path(file_path)
    if not fp.exists():
        log.warning("檔案不存在：%s", fp)
        return False

    try:
        with open(fp, "rb") as f:
            resp = req.post(
                _SEND_DOC_URL.format(token=token),
                data={
                    "chat_id": chat_id,
                    "caption": caption[:1024] if caption else "",
                    "parse_mode": "HTML",
                },
                files={"document": (fp.name, f, "text/html")},
                timeout=30,
            )
        data = resp.json()
        if data.get("ok"):
            log.info("Telegram 文件傳送成功：%s", fp.name)
            return True
        else:
            log.warning("Telegram 文件傳送失敗：%s", data.get("description", resp.text))
            return False
    except Exception as e:
        log.error("Telegram 文件傳送失敗：%s", e)
        return False
