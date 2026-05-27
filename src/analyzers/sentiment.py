"""
Phase 3-D：中文情緒分析（NII 品質升級）

對每個主題的新聞標題進行正負面情緒評分，輸出每日情緒指數
（Sentiment Score, SS），可作為 NII 的輔助維度。

演算法：
  - 中文標題：SnowNLP（基於 NB 模型，0~1 分，> 0.5 偏正面）
  - 英文標題：關鍵詞加權（正面詞 +1，負面詞 -1，歸一化到 0~1）
  - 每日聚合：所有標題分數的加權平均（越多文章越穩定）

輸出：
  data/processed/{topic}_sentiment.parquet
    columns: [date, ss_mean, ss_std, ss_count, ss_norm]
    - ss_mean：當日平均情緒（0=完全負面，1=完全正面，0.5=中性）
    - ss_norm：全期 z-score 正規化後再 0~1 壓縮

解讀：
  ss_norm 接近 1.0 → 市場正面情緒高 → 題材可能已過熱
  ss_norm 接近 0.0 → 市場負面但 NII 升溫 → 可能是最佳逢低訊號

執行：
    python -m src.analyzers.sentiment --topic cowos
    python -m src.analyzers.sentiment --all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

# ── 英文正負面關鍵詞 ─────────────────────────────────────────────────
_POS_WORDS = {
    "surge", "soar", "rally", "boom", "record", "growth", "strong",
    "bullish", "breakthrough", "win", "profit", "beat", "upgrade",
    "demand", "expand", "launch", "partner", "deal", "high", "rise",
    "positive", "opportunity", "accelerate", "gain", "outperform",
}
_NEG_WORDS = {
    "fall", "drop", "decline", "slump", "concern", "risk", "delay",
    "miss", "loss", "cut", "reduce", "bearish", "warning", "weak",
    "cancel", "halt", "deficit", "crisis", "uncertainty", "downgrade",
    "slow", "struggle", "pressure", "headwind", "disappoint", "shortage",
}


def _score_english(title: str) -> float:
    """英文標題情緒評分（0=負面，0.5=中性，1=正面）。"""
    words = set(title.lower().split())
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.5
    return (pos / total + 1) / 2   # 壓縮到 (0.5, 1.0) 不含純負面的極端


def _score_chinese(title: str) -> float:
    """中文標題情緒評分（SnowNLP，0~1）。"""
    try:
        from snownlp import SnowNLP
        return float(SnowNLP(title).sentiments)
    except Exception:
        return 0.5


def score_title(title: str) -> float:
    """自動判斷語言並評分。"""
    if not title or not isinstance(title, str):
        return 0.5
    # 若含中文字元則用 SnowNLP
    has_cjk = any("一" <= c <= "鿿" for c in title)
    if has_cjk:
        return _score_chinese(title)
    return _score_english(title)


# ── 每日情緒聚合 ──────────────────────────────────────────────────────

def build_sentiment(topic: str, save: bool = True) -> pd.DataFrame:
    """
    計算主題的每日情緒指數。

    Parameters
    ----------
    topic : str
        topics.yaml 中的主題 key
    save : bool
        是否存 parquet

    Returns
    -------
    pd.DataFrame
        index=date, columns=[ss_mean, ss_std, ss_count, ss_norm]
    """
    news_dir = ROOT / "data" / "raw" / "news"

    # 讀取 googlenews + gdelt 標題
    dfs = []
    for src in ["googlenews", "gdelt"]:
        path = news_dir / f"{topic}_{src}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "title" in df.columns and "published" in df.columns:
                dfs.append(df[["title", "published"]].copy())

    if not dfs:
        log.warning("[%s] 無新聞資料可分析", topic)
        return pd.DataFrame()

    news = pd.concat(dfs, ignore_index=True).drop_duplicates("title")
    news["published"] = pd.to_datetime(news["published"], errors="coerce", utc=True)
    news = news.dropna(subset=["published", "title"])
    news["date"] = news["published"].dt.tz_localize(None).dt.normalize()

    log.info("[%s] 共 %d 篇新聞，開始情緒評分…", topic, len(news))

    # 逐篇評分（可能耗時，顯示進度）
    scores = []
    for i, title in enumerate(news["title"], 1):
        scores.append(score_title(str(title)))
        if i % 100 == 0:
            log.info("  評分進度 %d/%d", i, len(news))
    news["ss"] = scores

    # 每日聚合
    daily = (
        news.groupby("date")["ss"]
        .agg(ss_mean="mean", ss_std="std", ss_count="count")
        .reset_index()
        .set_index("date")
        .sort_index()
    )

    # ss_norm：z-score 後壓縮到 0~1
    mu, sigma = daily["ss_mean"].mean(), daily["ss_mean"].std()
    if sigma > 1e-6:
        z = (daily["ss_mean"] - mu) / sigma
        daily["ss_norm"] = 1 / (1 + np.exp(-z))   # sigmoid 壓縮
    else:
        daily["ss_norm"] = 0.5

    log.info(
        "[%s] 情緒分析完成：ss 均值=%.3f  ss_norm 範圍=[%.2f, %.2f]",
        topic,
        daily["ss_mean"].mean(),
        daily["ss_norm"].min(),
        daily["ss_norm"].max(),
    )

    if save:
        out = ROOT / "data" / "processed" / f"{topic}_sentiment.parquet"
        daily.to_parquet(out)
        log.info("[%s] 已存：%s", topic, out.name)

    return daily


def load_sentiment(topic: str) -> pd.DataFrame | None:
    """載入已計算的情緒資料。"""
    path = ROOT / "data" / "processed" / f"{topic}_sentiment.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    import yaml
    parser = argparse.ArgumentParser(description="Phase 3-D：情緒分析")
    parser.add_argument("--topic", default=None, help="單一主題 key")
    parser.add_argument("--all",   action="store_true", help="所有主題")
    args = parser.parse_args()

    if args.all or not args.topic:
        with open(ROOT / "config" / "topics.yaml", encoding="utf-8") as f:
            topics = list(yaml.safe_load(f).keys())
    else:
        topics = [args.topic]

    for t in topics:
        try:
            build_sentiment(t, save=True)
        except Exception as e:
            log.error("[%s] 情緒分析失敗：%s", t, e)
