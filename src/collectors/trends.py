"""
Google Trends 採集模組

注意事項：
  - pytrends 容易被限流（429），已加入 exponential backoff retry
  - 同一次請求抓完整時間段，避免分段查詢造成基準漂移
  - 若持續失敗，可把人工匯出的 CSV 放到 data/raw/trends/<topic>_manual.csv
    格式：date(YYYY-MM-DD), <keyword>, ...

執行方式：
  python -m src.collectors.trends --topic cowos
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path

import pandas as pd
import yaml
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "topics.yaml"
RAW_TRENDS_DIR = ROOT / "data" / "raw" / "trends"

MAX_RETRIES = 5
BASE_SLEEP = 60  # 秒，第一次 retry 的等待基準


def load_topic(topic: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if topic not in cfg:
        raise KeyError(f"主題 '{topic}' 不存在於 config/topics.yaml")
    return cfg[topic]


def _build_timeframe(start: str, end: str) -> str:
    """pytrends 的 timeframe 格式：'YYYY-MM-DD YYYY-MM-DD'"""
    return f"{start} {end}"


def _fetch_with_retry(pytrends: TrendReq, kw_list: list[str], timeframe: str) -> pd.DataFrame | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pytrends.build_payload(kw_list, timeframe=timeframe, geo="TW")
            df = pytrends.interest_over_time()
            if df is not None and not df.empty:
                return df
            log.warning("pytrends 回傳空資料（嘗試 %d/%d）", attempt, MAX_RETRIES)
        except TooManyRequestsError:
            wait = BASE_SLEEP * (2 ** (attempt - 1)) + random.uniform(0, 30)
            log.warning("Google Trends 限流（429），等待 %.0f 秒後重試 %d/%d", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
        except Exception as e:
            log.error("pytrends 錯誤：%s（嘗試 %d/%d）", e, attempt, MAX_RETRIES)
            time.sleep(BASE_SLEEP)

    log.error("pytrends 重試 %d 次仍失敗，放棄", MAX_RETRIES)
    return None


def _load_manual_csv(topic: str) -> pd.DataFrame | None:
    """讀取人工匯出的 Google Trends CSV（備援）"""
    manual_path = RAW_TRENDS_DIR / f"{topic}_manual.csv"
    if manual_path.exists():
        log.info("發現人工 CSV：%s，改用此檔", manual_path)
        df = pd.read_csv(manual_path, parse_dates=["date"], index_col="date")
        return df
    return None


def collect(topic: str, start: str | None = None, end: str | None = None) -> pd.DataFrame | None:
    topic_cfg = load_topic(topic)
    start = start or topic_cfg["time_range"]["start"]
    end = end or topic_cfg["time_range"]["end"]

    RAW_TRENDS_DIR.mkdir(parents=True, exist_ok=True)

    # 先試人工 CSV 備援
    manual = _load_manual_csv(topic)
    if manual is not None:
        out_path = RAW_TRENDS_DIR / f"{topic}_trends.parquet"
        manual.to_parquet(out_path)
        log.info("已存 %s（來源：人工 CSV）", out_path)
        return manual

    keywords = topic_cfg["keywords"]
    # 優先用英文關鍵字（Trends 對英文的一致性較好）
    kw_list = keywords.get("english", keywords.get("primary", []))

    # pytrends 單次最多 5 個關鍵字
    if len(kw_list) > 5:
        log.warning("關鍵字超過 5 個，僅取前 5 個：%s", kw_list[:5])
        kw_list = kw_list[:5]

    log.info("Google Trends 查詢：%s  %s ~ %s", kw_list, start, end)
    timeframe = _build_timeframe(start, end)

    # 不傳 retries/backoff_factor：pytrends 在 urllib3 v2+ 有相容性問題
    # retry 邏輯已由 _fetch_with_retry 自行處理
    pytrends = TrendReq(hl="zh-TW", tz=480, timeout=(10, 25))

    df = _fetch_with_retry(pytrends, kw_list, timeframe)
    if df is None:
        log.error(
            "Google Trends 採集失敗。備援方案：\n"
            "  1. 手動到 trends.google.com 匯出 CSV\n"
            "  2. 存到 data/raw/trends/%s_manual.csv\n"
            "  3. 重新執行此模組",
            topic,
        )
        return None

    # 移除 isPartial 欄位
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    # 統一欄位名稱為 keyword_trends_<name>
    df.columns = [f"trends_{c}" for c in df.columns]
    df.index.name = "date"

    out_path = RAW_TRENDS_DIR / f"{topic}_trends.parquet"
    df.to_parquet(out_path)
    log.info("已存 %s（%d 筆）", out_path, len(df))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Trends 採集")
    parser.add_argument("--topic", default="cowos")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    collect(args.topic, args.start, args.end)


if __name__ == "__main__":
    main()
