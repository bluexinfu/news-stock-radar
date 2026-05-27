"""
領先/落後分析（A4）

計算 NII 在不同 lag 下與股價報酬的相關係數：
  - lag > 0：NII 領先股價（訊息先出現，股價後反應）
  - lag < 0：NII 落後股價（股價先動，訊息後跟上）
  - lag = 0：同步

Bootstrap 信賴區間用於確認最佳 lag 的統計顯著性。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_lag_correlation(
    signal: pd.Series,
    price: pd.Series,
    lags: list[int] | None = None,
    use_returns: bool = True,
) -> pd.DataFrame:
    """
    A4：計算不同 lag 下的皮爾森相關。

    Args:
        signal:      NII 序列（已對齊到交易日）
        price:       股價序列（已對齊到交易日）
        lags:        lag 天數清單，正數 = NII 領先，負數 = NII 落後
        use_returns: 是否用日報酬率（推薦）

    Returns:
        DataFrame with columns [lag, r, p_value, n]
    """
    if lags is None:
        lags = list(range(-10, 11))

    s = signal.copy()
    p = price.copy()

    if use_returns:
        s = s.diff()
        p = p.pct_change()

    common = s.index.intersection(p.index)
    s = s.loc[common].dropna()
    p = p.loc[common].dropna()
    common = s.index.intersection(p.index)
    s, p = s.loc[common], p.loc[common]

    rows = []
    for lag in lags:
        if lag > 0:
            # NII 領先：NII[t] vs price[t+lag]
            s_shifted = s.iloc[:-lag]
            p_shifted = p.iloc[lag:]
            p_shifted.index = s_shifted.index
        elif lag < 0:
            # NII 落後：NII[t] vs price[t+lag] => NII[t-|lag|] vs price[t]
            abs_lag = abs(lag)
            s_shifted = s.iloc[abs_lag:]
            p_shifted = p.iloc[:-abs_lag]
            s_shifted.index = p_shifted.index
        else:
            s_shifted, p_shifted = s, p

        valid = s_shifted.notna() & p_shifted.notna()
        sv, pv = s_shifted[valid], p_shifted[valid]

        if len(sv) < 20:
            rows.append({"lag": lag, "r": np.nan, "p_value": np.nan, "n": len(sv)})
            continue

        r, pval = stats.pearsonr(sv, pv)
        rows.append({"lag": lag, "r": round(r, 4), "p_value": round(pval, 5), "n": len(sv)})

    return pd.DataFrame(rows)


def bootstrap_best_lag(
    signal: pd.Series,
    price: pd.Series,
    lags: list[int] | None = None,
    n_boot: int = 500,
    ci: float = 0.95,
    use_returns: bool = True,
    random_state: int = 42,
) -> dict:
    """
    對最佳 lag 做 bootstrap 信賴區間。

    Returns:
        dict with keys: best_lag, r_at_best_lag, ci_lower, ci_upper, p_value
    """
    if lags is None:
        lags = list(range(-10, 11))

    rng = np.random.default_rng(random_state)
    lag_df = compute_lag_correlation(signal, price, lags, use_returns)
    lag_df = lag_df.dropna(subset=["r"])

    best_row = lag_df.loc[lag_df["r"].abs().idxmax()]
    best_lag = int(best_row["lag"])

    # 準備對應的對齊序列
    s = signal.diff() if use_returns else signal.copy()
    p = price.pct_change() if use_returns else price.copy()
    common = s.index.intersection(p.index)
    s, p = s.loc[common].dropna(), p.loc[common]
    common = s.index.intersection(p.index)
    s, p = s.loc[common], p.loc[common]

    lag = best_lag
    if lag > 0:
        sv = s.values[:-lag]
        pv = p.values[lag:]
    elif lag < 0:
        abs_lag = abs(lag)
        sv = s.values[abs_lag:]
        pv = p.values[:-abs_lag]
    else:
        sv, pv = s.values, p.values

    valid = ~(np.isnan(sv) | np.isnan(pv))
    sv, pv = sv[valid], pv[valid]

    boot_rs = []
    n = len(sv)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r_boot, _ = stats.pearsonr(sv[idx], pv[idx])
        boot_rs.append(r_boot)

    alpha = (1 - ci) / 2
    ci_lower = float(np.quantile(boot_rs, alpha))
    ci_upper = float(np.quantile(boot_rs, 1 - alpha))

    return {
        "best_lag": best_lag,
        "r_at_best_lag": float(best_row["r"]),
        "p_value": float(best_row["p_value"]),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "interpretation": (
            f"NII 領先股價 {best_lag} 天" if best_lag > 0 else
            f"NII 落後股價 {abs(best_lag)} 天" if best_lag < 0 else
            "NII 與股價同步"
        ),
    }
