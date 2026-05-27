"""
時序對齊模組

對齊策略（見 decisions.md D003）：
  - 以股價交易日為時間軸基準
  - Trends（週頻）→ 日頻：線性內插，週末累積到下一個交易日（forward fill 補齊非交易日後 reindex）
  - 新聞數（日頻）→ 週末值累積到下一個交易日
  - 股價缺漏：forward fill
  - 訊息強度缺漏：填 0（無訊息 ≠ missing）

執行方式：
  python -m src.processors.alignment --topic cowos
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "topics.yaml"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_topic(topic: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if topic not in cfg:
        raise KeyError(f"主題 '{topic}' 不存在於 config/topics.yaml")
    return cfg[topic]


def _safe_name(ticker: str) -> str:
    return ticker.replace(".", "_").replace("^", "")


def load_prices(topic: str, tickers: list[str]) -> dict[str, pd.Series]:
    """載入各標的收盤價，回傳 dict[ticker -> Series]"""
    out = {}
    for ticker in tickers:
        path = RAW_DIR / "prices" / f"{topic}_{_safe_name(ticker)}.parquet"
        if not path.exists():
            log.warning("找不到股價檔：%s，略過", path)
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).normalize()
        out[ticker] = df["close"].rename(ticker)
    return out


def load_trends(topic: str) -> pd.Series:
    """
    載入 Google Trends，自動選擇最有訊號的欄位。

    選欄策略：
    1. 從 topics.yaml 讀取 primary 關鍵字，優先比對 trends_{primary_keyword}
    2. 若無 primary 欄，選均值最高的欄（排除零值欄）
    3. 最後 fallback：取第一欄
    """
    path = RAW_DIR / "trends" / f"{topic}_trends.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).normalize()

    # 1. 嘗試從 topics.yaml 取得 primary 關鍵字
    try:
        cfg = load_topic(topic)
        primary_kw = cfg.get("keywords", {}).get("primary", [None])[0]
        primary_col = f"trends_{primary_kw}" if primary_kw else None
    except Exception:
        primary_col = None

    if primary_col and primary_col in df.columns:
        col = primary_col
        log.info("Trends 主力欄位（primary keyword）：%s", col)
    else:
        # 2. 選均值最高的欄
        means = df.mean()
        best_col = means.idxmax() if means.max() > 0 else None
        if best_col:
            col = best_col
            log.info("Trends 主力欄位（最高均值 %.2f）：%s", means[col], col)
        else:
            # 3. Fallback：第一欄
            col = df.columns[0]
            log.warning("Trends 均值均為 0，fallback 第一欄：%s", col)

    return df[col].rename("trends_raw")


def load_news_daily(topic: str) -> pd.Series:
    """載入每日新聞總數"""
    path = RAW_DIR / "news" / f"{topic}_daily_count.parquet"
    if not path.exists():
        log.warning("找不到每日新聞數：%s，以全零代替", path)
        return pd.Series(dtype=float, name="news_count")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index).normalize()
    return df["total_count"].rename("news_count")


def _trading_calendar(price_series_list: list[pd.Series]) -> pd.DatetimeIndex:
    """取所有股價共同的交易日集合"""
    idx = price_series_list[0].index
    for s in price_series_list[1:]:
        idx = idx.union(s.index)
    return idx.sort_values()


def _accumulate_weekend_to_monday(series: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    把落在非交易日（週末/假日）的值累積到下一個交易日。
    週末新聞等開盤才反映，因此不是 forward-fill 而是加總。
    """
    # 先 reindex 到日曆日（包含週末），再 groupby 最近下一個交易日
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(full_idx, fill_value=0)

    # 找每個日曆日對應的「下一個交易日」
    trading_set = set(trading_days)
    next_td = {}
    td_list = sorted(trading_days)
    for date in full_idx:
        if date in trading_set:
            next_td[date] = date
        else:
            future = [d for d in td_list if d >= date]
            next_td[date] = future[0] if future else None

    daily_df = daily.to_frame("v")
    daily_df["next_td"] = daily_df.index.map(next_td)
    daily_df = daily_df.dropna(subset=["next_td"])
    grouped = daily_df.groupby("next_td")["v"].sum()
    return grouped.reindex(trading_days, fill_value=0)


def _interpolate_trends(trends_weekly: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    週頻 Trends → 日頻：
      1. 在完整日曆日上線性內插（pytrends 週資料代表該週的值）
      2. reindex 到交易日
    """
    full_idx = pd.date_range(trends_weekly.index.min(), trends_weekly.index.max(), freq="D")
    daily = trends_weekly.reindex(full_idx).interpolate(method="linear")
    # 超出 Trends 範圍的交易日：用最近一週的值 forward-fill
    result = daily.reindex(trading_days).ffill().bfill()
    result = result.clip(0, 100)  # Trends 值域 0-100
    return result.rename("trends_interp")


def align(topic: str) -> pd.DataFrame:
    """
    主函式：對齊所有時序資料，輸出一個 DataFrame：
      index = 交易日
      columns = [close_<ticker>, ..., trends_interp, news_count]
    """
    topic_cfg = load_topic(topic)
    stocks = topic_cfg["related_stocks"]
    all_items = ([topic_cfg.get("benchmark")] if topic_cfg.get("benchmark") else []) + \
                [s["ticker"] for s in stocks.get("primary", [])] + \
                [s["ticker"] for s in stocks.get("secondary", [])]

    # 載入
    price_dict = load_prices(topic, all_items)
    trends_raw = load_trends(topic)
    news_raw = load_news_daily(topic)

    if not price_dict:
        raise RuntimeError("沒有任何股價資料，請先執行 collectors/prices.py")

    # 建立交易日基準
    trading_days = _trading_calendar(list(price_dict.values()))
    log.info("交易日基準：%s ~ %s（%d 天）",
             trading_days.min().date(), trading_days.max().date(), len(trading_days))

    # 對齊股價（forward fill 補假日缺口）
    frames = {}
    for ticker, series in price_dict.items():
        aligned = series.reindex(trading_days).ffill()
        frames[f"close_{_safe_name(ticker)}"] = aligned

    # 對齊 Trends（內插）
    frames["trends_interp"] = _interpolate_trends(trends_raw, trading_days)

    # 對齊新聞數（週末累積到下一個交易日）
    if not news_raw.empty:
        frames["news_count"] = _accumulate_weekend_to_monday(news_raw, trading_days)
    else:
        frames["news_count"] = pd.Series(0, index=trading_days, name="news_count")

    aligned_df = pd.DataFrame(frames)
    aligned_df.index.name = "date"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{topic}_aligned.parquet"
    aligned_df.to_parquet(out_path)
    log.info("已存對齊資料：%s（%d 列 × %d 欄）", out_path, *aligned_df.shape)

    return aligned_df


def main() -> None:
    parser = argparse.ArgumentParser(description="時序對齊")
    parser.add_argument("--topic", default="cowos")
    args = parser.parse_args()
    df = align(args.topic)
    print(df.tail())


if __name__ == "__main__":
    main()
