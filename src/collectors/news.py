"""
新聞採集模組

資料來源（依優先順序）：
  P0 — Google News RSS（feedparser）：中英文均可，免費
  P1 — GDELT 2.0（gdeltdoc）：全球英文新聞事件密度，完全免費

輸出：
  data/raw/news/<topic>_googlenews.parquet   — 每筆一篇文章（標題、日期）
  data/raw/news/<topic>_gdelt.parquet        — 每日文章數（依關鍵字聚合）

執行方式：
  python -m src.collectors.news --topic cowos
  python -m src.collectors.news --topic cowos --source googlenews
  python -m src.collectors.news --topic cowos --source gdelt
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import feedparser
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "topics.yaml"
RAW_NEWS_DIR = ROOT / "data" / "raw" / "news"

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
GOOGLE_NEWS_RSS_EN = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def load_topic(topic: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if topic not in cfg:
        raise KeyError(f"主題 '{topic}' 不存在於 config/topics.yaml")
    return cfg[topic]


# ─── Google News RSS ──────────────────────────────────────────────────────────

def _parse_pub_date(entry) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        val = getattr(entry, attr, None)
        if val:
            return datetime(*val[:6])
    return None


def _fetch_rss(query: str, lang: str = "zh") -> list[dict]:
    url_tpl = GOOGLE_NEWS_RSS if lang == "zh" else GOOGLE_NEWS_RSS_EN
    url = url_tpl.format(query=quote(query))
    log.debug("RSS URL: %s", url)

    feed = feedparser.parse(url)
    if feed.bozo and not feed.entries:
        log.warning("RSS 解析有問題（bozo=%s），查詢：%s", feed.bozo_exception, query)

    rows = []
    for entry in feed.entries:
        pub = _parse_pub_date(entry)
        rows.append(
            {
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", ""),
                "url": entry.get("link", ""),
                "published": pub,
                "query": query,
                "lang": lang,
            }
        )
    return rows


def _dedup_titles(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """用簡單的標題長度+前 N 字元去重（避免引入 fuzzywuzzy 依賴造成安裝問題）"""
    seen: set[str] = set()
    mask = []
    for title in df["title"]:
        key = title[:50].strip().lower()
        if key in seen:
            mask.append(False)
        else:
            seen.add(key)
            mask.append(True)
    before = len(df)
    df = df[mask].reset_index(drop=True)
    log.info("去重：%d → %d 筆", before, len(df))
    return df


def collect_googlenews(topic: str, start: str, end: str) -> pd.DataFrame | None:
    topic_cfg = load_topic(topic)
    keywords = topic_cfg["keywords"]
    exclude = topic_cfg.get("exclude_terms", [])

    all_rows: list[dict] = []

    # 中文關鍵字
    for kw in keywords.get("chinese", []):
        query = kw
        if exclude:
            query += " -" + " -".join(exclude)
        rows = _fetch_rss(query, lang="zh")
        log.info("Google News RSS [zh] '%s'：%d 筆", kw, len(rows))
        all_rows.extend(rows)
        time.sleep(2)

    # 英文關鍵字（補充覆蓋率）
    for kw in keywords.get("english", []):
        query = kw
        if exclude:
            query += " -" + " -".join(exclude)
        rows = _fetch_rss(query, lang="en")
        log.info("Google News RSS [en] '%s'：%d 筆", kw, len(rows))
        all_rows.extend(rows)
        time.sleep(2)

    if not all_rows:
        log.error("Google News RSS 所有關鍵字均回傳空資料")
        return None

    df = pd.DataFrame(all_rows)
    df["published"] = pd.to_datetime(df["published"], errors="coerce")
    df = df.dropna(subset=["published"])

    # 過濾時間範圍
    df = df[(df["published"] >= pd.Timestamp(start)) & (df["published"] <= pd.Timestamp(end))]
    df = _dedup_titles(df)
    df = df.sort_values("published").reset_index(drop=True)

    out_path = RAW_NEWS_DIR / f"{topic}_googlenews.parquet"
    df.to_parquet(out_path)
    log.info("Google News 已存 %s（%d 筆）", out_path, len(df))
    return df


# ─── GDELT 2.0 ────────────────────────────────────────────────────────────────

def collect_gdelt(topic: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        from gdeltdoc import GdeltDoc, Filters  # noqa: PLC0415
    except ImportError:
        log.warning("gdeltdoc 未安裝（pip install gdeltdoc），跳過 GDELT")
        return None

    topic_cfg = load_topic(topic)
    keywords = topic_cfg["keywords"]
    # GDELT 只支援英文
    kw_list = keywords.get("english", keywords.get("primary", []))

    if not kw_list:
        log.warning("無英文關鍵字，跳過 GDELT")
        return None

    gd = GdeltDoc()
    all_rows: list[dict] = []

    # GDELT 單次查詢最多 3 個月，需要分段
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=90), end_dt)
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")

        for kw in kw_list[:3]:  # GDELT 建議不超過 3 個關鍵字
            try:
                f = Filters(
                    keyword=kw,
                    start_date=cs,
                    end_date=ce,
                )
                articles = gd.article_search(f)
                if articles is not None and not articles.empty:
                    articles["query"] = kw
                    all_rows.append(articles)
                    log.info("GDELT '%s' %s~%s：%d 筆", kw, cs, ce, len(articles))
                time.sleep(6)  # GDELT 要求每次請求間隔 ≥5 秒
            except Exception as e:
                log.warning("GDELT 查詢失敗 '%s' %s~%s：%s", kw, cs, ce, e)
                time.sleep(6)

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(10)  # chunk 之間額外等待，避免 GDELT 累計限流

    if not all_rows:
        log.warning("GDELT 所有查詢均回傳空資料")
        return None

    df = pd.concat(all_rows, ignore_index=True)

    # 統一日期欄位名稱
    if "seendate" in df.columns:
        df["published"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    elif "date" in df.columns:
        df["published"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["published"])
    df = df.sort_values("published").reset_index(drop=True)

    out_path = RAW_NEWS_DIR / f"{topic}_gdelt.parquet"
    df.to_parquet(out_path)
    log.info("GDELT 已存 %s（%d 筆）", out_path, len(df))
    return df


# ─── 每日新聞數彙整 ────────────────────────────────────────────────────────────

def build_daily_count(googlenews_df: pd.DataFrame | None, gdelt_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    把文章層級資料彙整成每日新聞數，供 NII 計算使用。
    回傳 DataFrame：index=date, columns=[googlenews_count, gdelt_count, total_count]
    """
    frames = {}

    if googlenews_df is not None and not googlenews_df.empty:
        gn = googlenews_df.copy()
        gn["date"] = gn["published"].dt.normalize()
        frames["googlenews_count"] = gn.groupby("date").size()

    if gdelt_df is not None and not gdelt_df.empty:
        gd = gdelt_df.copy()
        gd["date"] = gd["published"].dt.normalize()
        frames["gdelt_count"] = gd.groupby("date").size()

    if not frames:
        log.warning("無任何新聞資料，回傳空 DataFrame")
        return pd.DataFrame()

    daily = pd.DataFrame(frames).fillna(0).astype(int)
    daily["total_count"] = daily.sum(axis=1)
    daily.index.name = "date"
    return daily


# ─── 主函式 ────────────────────────────────────────────────────────────────────

def collect(
    topic: str,
    start: str | None = None,
    end: str | None = None,
    source: str = "all",
) -> pd.DataFrame | None:
    topic_cfg = load_topic(topic)
    start = start or topic_cfg["time_range"]["start"]
    end = end or topic_cfg["time_range"]["end"]

    RAW_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    gn_df = gdelt_df = None

    if source in ("all", "googlenews"):
        gn_df = collect_googlenews(topic, start, end)

    if source in ("all", "gdelt"):
        gdelt_df = collect_gdelt(topic, start, end)

    daily = build_daily_count(gn_df, gdelt_df)
    if not daily.empty:
        out_path = RAW_NEWS_DIR / f"{topic}_daily_count.parquet"
        daily.to_parquet(out_path)
        log.info("每日新聞數已存 %s", out_path)

    return daily if not daily.empty else None


def main() -> None:
    parser = argparse.ArgumentParser(description="新聞採集")
    parser.add_argument("--topic", default="cowos")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--source", default="all", choices=["all", "googlenews", "gdelt"])
    args = parser.parse_args()
    collect(args.topic, args.start, args.end, args.source)


if __name__ == "__main__":
    main()
