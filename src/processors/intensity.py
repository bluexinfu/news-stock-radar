"""
訊息強度指數（News Intensity Index, NII）計算模組

公式（見 decisions.md D002）：
  NII_t = w_trends × Trends_norm(t) + w_news × NewsCount_zscore(t)

  - Trends_norm：已在 0-100 範圍，直接使用
  - NewsCount_zscore：全期間 z-score 正規化
  - 預設權重：w_trends = w_news = 0.5

輸出：
  data/processed/<topic>_nii.parquet
    columns: [nii, trends_norm, news_zscore, nii_w30, nii_w50, nii_w70]

執行方式：
  python -m src.processors.intensity --topic cowos
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def load_aligned(topic: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{topic}_aligned.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 {path}，請先執行：python -m src.processors.alignment --topic {topic}"
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def compute_nii(
    aligned_df: pd.DataFrame,
    w_trends: float = 0.5,
) -> pd.Series:
    """
    計算 NII。

    Args:
        aligned_df: alignment.py 的輸出
        w_trends:   Trends 的權重（news 權重 = 1 - w_trends）

    Returns:
        NII Series，index 與 aligned_df 相同
    """
    w_news = 1.0 - w_trends

    trends = aligned_df["trends_interp"].copy()
    news = aligned_df["news_count"].copy().astype(float)

    # Trends 已在 0-100，直接用（trends_norm）
    trends_norm = trends

    # NewsCount z-score（全期間）
    mu = news.mean()
    sigma = news.std(ddof=0)
    if sigma == 0:
        log.warning("新聞數標準差為 0（全為相同值），zscore 設為 0")
        news_zscore = pd.Series(0.0, index=news.index)
    else:
        news_zscore = (news - mu) / sigma

    nii = w_trends * trends_norm + w_news * news_zscore
    return nii


def build_nii_table(topic: str) -> pd.DataFrame:
    """
    計算並輸出完整 NII 表格，含三組權重的敏感度比較。
    """
    aligned = load_aligned(topic)

    trends_norm = aligned["trends_interp"]
    news = aligned["news_count"].astype(float)

    mu, sigma = news.mean(), news.std(ddof=0)
    if sigma == 0:
        news_zscore = pd.Series(0.0, index=news.index)
    else:
        news_zscore = (news - mu) / sigma

    result = pd.DataFrame(index=aligned.index)
    result["trends_norm"] = trends_norm
    result["news_count"] = news
    result["news_zscore"] = news_zscore

    # 三組權重（敏感度分析）
    for w in [0.3, 0.5, 0.7]:
        col = f"nii_w{int(w * 100)}"
        result[col] = w * trends_norm + (1 - w) * news_zscore

    # 預設（w=0.5）為主欄
    result["nii"] = result["nii_w50"]

    # 附加：NII 的 min-max 正規化版本（方便與股價疊圖）
    nii = result["nii"]
    nii_min, nii_max = nii.min(), nii.max()
    if nii_max > nii_min:
        result["nii_norm"] = (nii - nii_min) / (nii_max - nii_min) * 100
    else:
        result["nii_norm"] = 50.0

    out_path = PROCESSED_DIR / f"{topic}_nii.parquet"
    result.to_parquet(out_path)
    log.info("NII 計算完成，已存 %s（%d 列 × %d 欄）", out_path, *result.shape)

    # 摘要
    log.info("NII 統計（w=0.5）：mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
             result["nii"].mean(), result["nii"].std(),
             result["nii"].min(), result["nii"].max())

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="NII 訊息強度計算")
    parser.add_argument("--topic", default="cowos")
    args = parser.parse_args()

    df = build_nii_table(args.topic)
    print(df[["trends_norm", "news_count", "news_zscore", "nii"]].tail(10).round(3))


if __name__ == "__main__":
    main()
