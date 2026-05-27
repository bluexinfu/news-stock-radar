"""
概念股等權指數（Sector Momentum Index, SMI）

將同一主題下的概念股等權平均，形成一個「主題籃子指數」，
降低個股雜訊，讓 NII × 指數的相關係數更能反映主題整體動能。

使用方法：
    from src.processors.sector_index import build_smi
    smi = build_smi("cowos", aligned_df)  # 回傳 pd.Series（日報酬率）
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 大盤 benchmark 代號，排除在 SMI 成份之外
BENCHMARK_TICKERS = {"TWII", "^TWII"}


def _get_close_cols(aligned_df: pd.DataFrame) -> list[str]:
    """
    從 aligned_df 取出所有個股收盤價欄位（排除大盤）。

    欄位命名規則：close_2330_TW、close_3680_TWO 等
    （原始 ticker 中的 '.' 與 '^' 已被替換為 '_'）
    """
    benchmark_suffixes = {
        "close_" + t.replace(".", "_").replace("^", "")
        for t in BENCHMARK_TICKERS
    }
    return [
        c for c in aligned_df.columns
        if c.startswith("close_") and c not in benchmark_suffixes
    ]


def build_smi(
    topic: str,
    aligned_df: pd.DataFrame,
    method: str = "equal",
    save: bool = True,
) -> pd.Series:
    """
    建立概念股等權指數（Sector Momentum Index）。

    Parameters
    ----------
    topic : str
        主題名稱（用於輸出檔名）
    aligned_df : pd.DataFrame
        `align(topic)` 輸出的對齊 DataFrame，index 為交易日
    method : str
        "equal" = 等權平均（預設）
        目前僅支援等權；市值加權留待後續擴充
    save : bool
        是否將 SMI 存檔至 data/processed/<topic>_smi.parquet

    Returns
    -------
    pd.Series
        日報酬率序列，index 為交易日，名稱為 "smi_return"
    """
    if method != "equal":
        raise NotImplementedError(f"method='{method}' 尚未支援，目前僅支援 'equal'")

    close_cols = _get_close_cols(aligned_df)
    if not close_cols:
        raise ValueError("aligned_df 中找不到任何個股收盤價欄位（close_*）")

    log.info("SMI 成份（%d 檔）：%s", len(close_cols), close_cols)

    # 計算各股日報酬率，再取等權平均
    prices = aligned_df[close_cols]
    daily_returns = prices.pct_change()          # shape: (T, N)

    smi_return = daily_returns.mean(axis=1)      # 等權平均
    smi_return.name = "smi_return"

    # 計算等權指數水準（從 100 出發的累積指數，僅供視覺化）
    smi_level = (1 + smi_return.fillna(0)).cumprod() * 100
    smi_level.name = "smi_level"

    # 統計摘要
    n = len(close_cols)
    valid_days = smi_return.dropna()
    log.info(
        "SMI 日報酬率：mean=%.4f  std=%.4f  days=%d  stocks=%d",
        valid_days.mean(), valid_days.std(), len(valid_days), n,
    )

    if save:
        out_df = pd.DataFrame({"smi_return": smi_return, "smi_level": smi_level})
        out_path = PROCESSED_DIR / f"{topic}_smi.parquet"
        out_df.to_parquet(out_path)
        log.info("SMI 已存 %s", out_path)

    return smi_return


def smi_summary(smi_return: pd.Series) -> dict:
    """
    回傳 SMI 日報酬率的摘要統計，供 notebook 快速展示。

    Returns
    -------
    dict
        keys: n_days, mean_ret, std_ret, ann_vol, sharpe_est, max_dd
    """
    r = smi_return.dropna()
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    ann_vol = r.std() * np.sqrt(252)
    ann_ret = (1 + r.mean()) ** 252 - 1
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    return {
        "n_days":    len(r),
        "mean_ret":  r.mean(),
        "std_ret":   r.std(),
        "ann_vol":   ann_vol,
        "ann_ret":   ann_ret,
        "sharpe_est": sharpe,
        "max_dd":    max_dd,
    }
