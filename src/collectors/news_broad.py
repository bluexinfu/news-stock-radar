"""
廣域中文財經新聞採集器（不依賴預設題材關鍵字）

來源：
  1. Google News RSS × 廣域財經詞（台股、概念股、法人、漲停…）
  2. Yahoo 奇摩財經 RSS
  3. 經濟日報 RSS
  4. 聯合財經 RSS

目的：
  提供給 topic_discovery.py 做 BERTopic 自動題材發現。
  不限定特定主題，盡量涵蓋市場當前在討論的所有財經議題。

用法：
  from src.collectors.news_broad import collect_broad
  df = collect_broad(days=90)        # 最近 90 天的廣域新聞標題

  # 或 CLI
  python -m src.collectors.news_broad --days 90
"""

from __future__ import annotations

import argparse
import logging
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import feedparser
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BROAD_DIR = ROOT / "data" / "raw" / "news_broad"
BROAD_DIR.mkdir(parents=True, exist_ok=True)

# ── 資料來源設定 ─────────────────────────────────────────────────────

# Google News RSS 廣域財經詞（不針對任何特定題材）
GOOGLE_NEWS_BROAD_KW = [
    "台股概念股",
    "電子股 漲停",
    "半導體 題材",
    "台灣股市 法人",
    "法人買超 概念",
    "台股 新題材",
    "上市 上櫃 漲停板",
    "台積電 供應鏈",
    "AI概念股",
    "類股 輪動",
]

# 固定 RSS 源（不依賴關鍵字）
STATIC_RSS = {
    "yahoo_finance_tw": "https://tw.stock.yahoo.com/rss",
    "udn_economy":      "https://money.udn.com/rssfeed/news/1001/5590/5607?ch=money",
    "lianhe_finance":   "https://udn.com/rssfeed/news/2/6644?ch=news",
}


# ── 採集函式 ─────────────────────────────────────────────────────────

def _parse_date(entry: dict) -> datetime | None:
    """嘗試從 feedparser entry 解析發布日期。"""
    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                return datetime(*t[:6])
            except Exception:
                pass
    return None


def _fetch_rss(url: str, source_name: str, cutoff: datetime) -> list[dict]:
    """抓取單一 RSS，回傳 cutoff 之後的 articles。"""
    rows = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            pub = _parse_date(entry)
            if pub and pub < cutoff:
                continue          # 太舊
            title = entry.get("title", "").strip()
            link  = entry.get("link", "")
            if not title:
                continue
            rows.append({
                "title":     title,
                "published": pub or datetime.now(),
                "source":    source_name,
                "url":       link,
            })
    except Exception as e:
        log.warning("[%s] RSS 抓取失敗：%s", source_name, e)
    return rows


def _fetch_google_news(kw: str, cutoff: datetime, sleep_s: float = 2.0) -> list[dict]:
    """用廣域關鍵字抓 Google News RSS。"""
    encoded = urllib.parse.quote(kw)
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )
    rows = _fetch_rss(url, f"gnews:{kw}", cutoff)
    time.sleep(sleep_s)
    return rows


# ── 主函式 ───────────────────────────────────────────────────────────

def collect_broad(
    days: int = 90,
    save: bool = True,
) -> pd.DataFrame:
    """
    採集廣域中文財經新聞（最近 N 天）。

    Parameters
    ----------
    days : int
        往回採集幾天（預設 90 天）
    save : bool
        是否存到 data/raw/news_broad/

    Returns
    -------
    pd.DataFrame
        columns: title, published, source, url
    """
    cutoff = datetime.now() - timedelta(days=days)
    all_rows: list[dict] = []

    # 1. 固定 RSS 源
    for name, url in STATIC_RSS.items():
        log.info("RSS [%s]", name)
        rows = _fetch_rss(url, name, cutoff)
        log.info("  → %d 篇", len(rows))
        all_rows.extend(rows)
        time.sleep(1)

    # 2. Google News 廣域詞
    for kw in GOOGLE_NEWS_BROAD_KW:
        log.info("Google News [%s]", kw)
        rows = _fetch_google_news(kw, cutoff)
        log.info("  → %d 篇", len(rows))
        all_rows.extend(rows)

    if not all_rows:
        log.warning("無任何廣域新聞資料")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])

    # 去重（同標題）
    before = len(df)
    df = df.drop_duplicates(subset=["title"])
    df = df.sort_values("published").reset_index(drop=True)
    log.info("廣域新聞：去重前 %d → %d 篇（%s ~ %s）",
             before, len(df),
             df["published"].min().date(), df["published"].max().date())

    if save:
        out = BROAD_DIR / f"broad_{datetime.now().strftime('%Y%m%d')}.parquet"
        df.to_parquet(out, index=False)
        log.info("已存 %s", out)

    return df


def load_broad(days: int = 90) -> pd.DataFrame:
    """
    讀取 data/raw/news_broad/ 下的既有資料（合併多個日期的檔案）。
    若無資料則自動採集一次。
    """
    files = sorted(BROAD_DIR.glob("broad_*.parquet"))
    if not files:
        log.info("無既有廣域資料，執行採集…")
        return collect_broad(days=days, save=True)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["published"] = pd.to_datetime(df["published"])
        dfs.append(df[df["published"] >= cutoff])

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["title"])
    df = df.sort_values("published").reset_index(drop=True)
    log.info("載入廣域新聞：%d 篇（%d 個檔案）", len(df), len(files))
    return df


# ── CLI 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="廣域中文財經新聞採集")
    parser.add_argument("--days", type=int, default=90, help="往回採集天數（預設 90）")
    args = parser.parse_args()
    df = collect_broad(days=args.days, save=True)
    print(f"\n完成：{len(df)} 篇  來源分布：")
    print(df["source"].value_counts().to_string())
