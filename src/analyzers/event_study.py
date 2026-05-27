"""
事件研究（A5）

找出 NII 的歷史高峰日，觀察前後 ±N 天的平均股價變化（Cumulative Abnormal Return, CAR）。

方法：
  1. 定義「事件日」= NII 超過 (mean + k×std) 的日子，且相鄰事件間隔至少 window 天
  2. 對每個事件日，計算 [-N, +N] 天窗口內的累積報酬率
  3. 平均化後繪製事件研究圖
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_event_days(
    nii: pd.Series,
    k: float = 1.5,
    min_gap: int = 20,
) -> pd.DatetimeIndex:
    """
    找出 NII 超過 mean + k*std 的峰值日（非連續，間隔至少 min_gap 天）。
    """
    threshold = nii.mean() + k * nii.std()
    above = nii[nii > threshold].sort_values(ascending=False)

    events = []
    for date in above.index:
        if not events or all(abs((date - e).days) >= min_gap for e in events):
            events.append(date)

    return pd.DatetimeIndex(sorted(events))


def compute_event_returns(
    price: pd.Series,
    event_days: pd.DatetimeIndex,
    window: int = 10,
) -> pd.DataFrame:
    """
    對每個事件日計算 [-window, +window] 的累積報酬率（以事件日為基準 = 0%）。

    Returns:
        DataFrame，index = 相對天數（-window ~ +window），
        columns = event dates + "mean" + "median"
    """
    trading_days = price.index
    returns = price.pct_change()

    all_cars = {}
    for event in event_days:
        if event not in trading_days:
            continue

        pos = trading_days.get_loc(event)
        start = max(0, pos - window)
        end = min(len(trading_days) - 1, pos + window)

        window_ret = returns.iloc[start: end + 1].values
        cum_ret = np.cumprod(1 + np.nan_to_num(window_ret)) - 1

        # 對齊到相對天數
        rel_days = list(range(start - pos, end - pos + 1))
        s = pd.Series(cum_ret, index=rel_days)

        # 在事件日前將累積報酬歸零（相對基準）
        if 0 in s.index:
            base = s.loc[: 0].iloc[-1] if 0 in s.index else 0
            s = s - base

        all_cars[event] = s

    if not all_cars:
        return pd.DataFrame()

    df = pd.DataFrame(all_cars)
    df = df.reindex(range(-window, window + 1))
    df["mean"] = df.mean(axis=1)
    df["median"] = df.median(axis=1)
    return df


def run_event_study(
    nii: pd.Series,
    price: pd.Series,
    k: float = 1.5,
    min_gap: int = 20,
    window: int = 10,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """
    完整事件研究流程。

    Returns:
        (event_days, car_df)
    """
    event_days = find_event_days(nii, k=k, min_gap=min_gap)
    car_df = compute_event_returns(price, event_days, window=window)
    return event_days, car_df
