"""
股價採集模組

資料來源優先順序：
  1. yfinance（.TW / .TWO 均嘗試）
  2. twstock（台灣上市，備援）
  3. FinMind（需 token，P2）

執行方式：
  python -m src.collectors.prices --topic cowos
  python -m src.collectors.prices --topic cowos --start 2024-01-01 --end 2026-05-26
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "topics.yaml"
RAW_PRICES_DIR = ROOT / "data" / "raw" / "prices"


def load_topic(topic: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if topic not in cfg:
        raise KeyError(f"主題 '{topic}' 不存在於 config/topics.yaml")
    return cfg[topic]


def _all_tickers(topic_cfg: dict) -> list[dict]:
    stocks = topic_cfg["related_stocks"]
    tickers = list(stocks.get("primary", []))
    tickers += list(stocks.get("secondary", []))
    return tickers


def fetch_single(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    if ticker == "VERIFY.TW":
        log.warning("代號 VERIFY.TW 尚未確認，跳過")
        return None

    log.info("抓取 %s  %s ~ %s", ticker, start, end)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            log.warning("%s：yfinance 回傳空資料", ticker)
            return None

        # yfinance ≥0.2.x 回傳 MultiIndex columns，攤平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        df["ticker"] = ticker
        log.info("%s：取得 %d 筆", ticker, len(df))
        return df

    except Exception as e:
        log.error("%s：yfinance 失敗 — %s", ticker, e)
        return None


def _twstock_fallback(ticker_code: str, start: str, end: str) -> pd.DataFrame | None:
    """上市股票 twstock 備援（不含 .TW/.TWO 後綴）"""
    try:
        import twstock  # noqa: PLC0415
    except ImportError:
        log.warning("twstock 未安裝，無法備援")
        return None

    code = ticker_code.split(".")[0]
    log.info("%s：改用 twstock 備援", code)
    try:
        stock = twstock.Stock(code)
        start_y, start_m = int(start[:4]), int(start[5:7])
        end_y, end_m = int(end[:4]), int(end[5:7])

        records = []
        y, m = start_y, start_m
        while (y, m) <= (end_y, end_m):
            data = stock.fetch(y, m)
            records.extend(data)
            m += 1
            if m > 12:
                m = 1
                y += 1

        if not records:
            return None

        df = pd.DataFrame(
            [
                {
                    "date": r.date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.capacity,
                }
                for r in records
            ]
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        df["ticker"] = ticker_code
        log.info("%s：twstock 取得 %d 筆", code, len(df))
        return df
    except Exception as e:
        log.error("%s twstock 也失敗 — %s", code, e)
        return None


def collect(topic: str, start: str | None = None, end: str | None = None) -> dict[str, pd.DataFrame]:
    topic_cfg = load_topic(topic)
    start = start or topic_cfg["time_range"]["start"]
    end = end or topic_cfg["time_range"]["end"]

    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    tickers = _all_tickers(topic_cfg)

    # 也抓大盤基準
    benchmark = topic_cfg.get("benchmark")
    if benchmark:
        tickers = [{"ticker": benchmark, "name": "大盤基準", "role": "benchmark"}] + tickers

    for item in tickers:
        ticker = item["ticker"]
        df = fetch_single(ticker, start, end)

        # yfinance 失敗且為台股，改用 twstock
        if df is None and (".TW" in ticker or ".TWO" in ticker):
            df = _twstock_fallback(ticker, start, end)

        if df is None:
            log.warning("%s：所有資料源均失敗，略過", ticker)
            continue

        safe_name = ticker.replace(".", "_").replace("^", "")
        out_path = RAW_PRICES_DIR / f"{topic}_{safe_name}.parquet"
        df.to_parquet(out_path)
        log.info("已存 %s", out_path)
        results[ticker] = df

    log.info("股價採集完成：共 %d 支標的", len(results))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="股價採集")
    parser.add_argument("--topic", default="cowos")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    collect(args.topic, args.start, args.end)


if __name__ == "__main__":
    main()
