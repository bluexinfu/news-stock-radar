"""
相關性分析：皮爾森相關 + 滾動相關（A2, A3）

所有相關係數計算均使用「日報酬率」而非原始價格，
以避免共同趨勢造成的虛假相關（spurious correlation）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def load_data(topic: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.read_parquet(PROCESSED_DIR / f"{topic}_aligned.parquet")
    nii = pd.read_parquet(PROCESSED_DIR / f"{topic}_nii.parquet")
    aligned.index = pd.to_datetime(aligned.index)
    nii.index = pd.to_datetime(nii.index)
    return aligned, nii


def _price_cols(aligned: pd.DataFrame) -> list[str]:
    return [c for c in aligned.columns if c.startswith("close_")]


def pearson_table(
    aligned: pd.DataFrame,
    nii: pd.DataFrame,
    nii_col: str = "nii",
    use_returns: bool = True,
) -> pd.DataFrame:
    """
    A2：計算 NII 與各標的的全期間皮爾森相關。

    Args:
        use_returns: True = 使用日報酬率（推薦）；False = 使用原始值

    Returns:
        DataFrame with columns [ticker, r, p_value, n, significant]
    """
    signal = nii[nii_col].copy()
    if use_returns:
        signal = signal.diff().dropna()

    rows = []
    for col in _price_cols(aligned):
        price = aligned[col]
        if use_returns:
            price = price.pct_change().dropna()

        common = signal.index.intersection(price.index)
        s, p = signal.loc[common], price.loc[common]
        valid = s.notna() & p.notna()
        s, p = s[valid], p[valid]

        if len(s) < 30:
            continue

        r, pval = stats.pearsonr(s, p)
        rows.append({
            "ticker": col.replace("close_", "").replace("_", "."),
            "r": round(r, 4),
            "p_value": round(pval, 5),
            "n": len(s),
            "significant": pval < 0.05,
        })

    return pd.DataFrame(rows).sort_values("r", ascending=False).reset_index(drop=True)


def rolling_correlation(
    aligned: pd.DataFrame,
    nii: pd.DataFrame,
    ticker_col: str,
    nii_col: str = "nii",
    windows: list[int] | None = None,
    use_returns: bool = True,
) -> pd.DataFrame:
    """
    A3：計算 NII × 特定標的 的滾動相關係數。

    Args:
        ticker_col: aligned 中的欄位名（如 'close_2330_TW'）
        windows:    滾動窗口天數清單，預設 [30, 60]

    Returns:
        DataFrame，index=date，columns=[rolling_corr_30, rolling_corr_60, ...]
    """
    if windows is None:
        windows = [30, 60]

    signal = nii[nii_col].copy()
    price = aligned[ticker_col].copy()

    if use_returns:
        signal = signal.diff()
        price = price.pct_change()

    common = signal.index.intersection(price.index)
    df = pd.DataFrame({"nii": signal.loc[common], "price": price.loc[common]}).dropna()

    result = pd.DataFrame(index=df.index)
    for w in windows:
        result[f"rolling_corr_{w}d"] = (
            df["nii"].rolling(w, min_periods=int(w * 0.6))
            .corr(df["price"])
        )

    return result


def rolling_correlation_all(
    aligned: pd.DataFrame,
    nii: pd.DataFrame,
    nii_col: str = "nii",
    windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    對所有標的計算滾動相關，回傳 dict[ticker -> DataFrame]。
    """
    if windows is None:
        windows = [30, 60]

    out = {}
    for col in _price_cols(aligned):
        ticker = col.replace("close_", "").replace("_", ".")
        out[ticker] = rolling_correlation(aligned, nii, col, nii_col, windows)
    return out
