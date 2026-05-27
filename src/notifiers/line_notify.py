"""
LINE Notify 推播模組

使用方式：
    from src.notifiers.line_notify import send

    send("你的訊息")

環境設定：
    在 .env 檔案（或系統環境變數）中設定：
        LINE_NOTIFY_TOKEN=your_token_here

    取得 Token：https://notify-bot.line.me/my/
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"


def _load_dotenv() -> None:
    """從專案根目錄的 .env 讀取環境變數（不依賴 python-dotenv）。"""
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


def send(message: str, token: str | None = None) -> bool:
    """
    透過 LINE Notify API 發送訊息。

    Parameters
    ----------
    message : str
        要發送的文字（LINE Notify 限制 1000 字以內）
    token : str, optional
        LINE Notify Token。若未提供，從環境變數 LINE_NOTIFY_TOKEN 讀取。

    Returns
    -------
    bool
        True = 發送成功，False = 失敗（含原因 log）
    """
    _load_dotenv()

    token = token or os.getenv("LINE_NOTIFY_TOKEN", "").strip()
    if not token or token in ("your_line_notify_token_here", ""):
        log.warning("LINE_NOTIFY_TOKEN 未設定，跳過推播")
        return False

    try:
        import requests
    except ImportError:
        log.error("requests 未安裝，請執行：pip install requests")
        return False

    try:
        resp = requests.post(
            LINE_NOTIFY_URL,
            headers={"Authorization": f"Bearer {token}"},
            data={"message": message},
            timeout=10,
        )
        if resp.status_code == 200:
            log.info("LINE Notify 推播成功")
            return True
        else:
            log.warning("LINE Notify 回傳 %d：%s", resp.status_code, resp.text)
            return False
    except Exception as e:
        log.error("LINE Notify 推播失敗：%s", e)
        return False


def send_image(message: str, image_path: str, token: str | None = None) -> bool:
    """發送含圖片的 LINE Notify（選用）。"""
    _load_dotenv()
    token = token or os.getenv("LINE_NOTIFY_TOKEN", "").strip()
    if not token:
        log.warning("LINE_NOTIFY_TOKEN 未設定，跳過推播")
        return False

    try:
        import requests
        with open(image_path, "rb") as img:
            resp = requests.post(
                LINE_NOTIFY_URL,
                headers={"Authorization": f"Bearer {token}"},
                data={"message": message},
                files={"imageFile": img},
                timeout=30,
            )
        return resp.status_code == 200
    except Exception as e:
        log.error("LINE Notify 圖片推播失敗：%s", e)
        return False
