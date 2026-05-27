"""
多主題題材雷達（Theme Radar）

功能：
  1. compute_nii_slope()  — NII 熱度加速度（14天滾動線性斜率）
  2. detect_phase()       — 判斷主題目前處於「冷卻/預熱/發燒/降溫」哪個階段
  3. rank_themes()        — 多主題排行榜，依「熱度加速」排序

設計目標：
  偵測「下一波題材」在起漲前的熱度萌芽（預熱階段），
  作為籌碼面觀察的輔助工具。

⚠️ 所有輸出均為觀察性指標，相關不等於因果。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── 1. NII 熱度斜率（加速度） ────────────────────────────────────────

def compute_nii_slope(nii: pd.Series, window: int = 14) -> pd.Series:
    """
    計算 NII 的滾動線性斜率（熱度加速度）。

    以 `window` 天滾動窗口做線性回歸，取斜率係數（每天上升多少 NII 單位）。

    Parameters
    ----------
    nii : pd.Series
        NII 時序（index=交易日）
    window : int
        滾動窗口長度（預設 14 天）

    Returns
    -------
    pd.Series
        滾動斜率序列，正值=熱度上升，負值=降溫
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / x_var)

    slopes = nii.rolling(window).apply(_slope, raw=True)
    slopes.name = f"nii_slope_{window}d"
    return slopes


# ── 2. 階段判斷 ─────────────────────────────────────────────────────

_PHASE_ORDER = ["冷卻", "預熱", "發燒", "降溫"]


def detect_phase(nii: pd.Series, slope_window: int = 14) -> str:
    """
    判斷主題 NII 目前處於哪個階段。

    階段定義（基於全期均值與標準差）：
    ┌────────┬────────────────────────────────────────────────┐
    │ 冷卻   │ NII < mean，且近期斜率 ≤ 0                   │
    │ 預熱   │ NII < mean，但近期斜率 > 0（最值得關注！）  │
    │ 發燒   │ NII ≥ mean + 1σ                              │
    │ 降溫   │ NII ≥ mean，但近期斜率 < 0                   │
    └────────┴────────────────────────────────────────────────┘

    Parameters
    ----------
    nii : pd.Series
        NII 時序（含歷史資料）
    slope_window : int
        計算斜率使用的滾動窗口

    Returns
    -------
    str
        "冷卻" | "預熱" | "發燒" | "降溫"
    """
    valid = nii.dropna()
    if len(valid) < slope_window + 1:
        return "冷卻"

    mu = valid.mean()
    sigma = valid.std()
    latest = float(valid.iloc[-1])

    slopes = compute_nii_slope(valid, window=slope_window)
    latest_slope = float(slopes.dropna().iloc[-1]) if slopes.dropna().any() else 0.0

    if latest >= mu + sigma:
        return "發燒"
    elif latest >= mu:
        return "降溫" if latest_slope < 0 else "發燒"
    else:
        return "預熱" if latest_slope > 0 else "冷卻"


# ── 3. 多主題排行 ────────────────────────────────────────────────────

PHASE_EMOJI = {
    "冷卻": "❄️",
    "預熱": "🌡️",
    "發燒": "🔥",
    "降溫": "📉",
}

PHASE_PRIORITY = {
    "預熱": 1,   # 最高優先（早期信號）
    "發燒": 2,
    "降溫": 3,
    "冷卻": 4,
}


def rank_themes(
    topic_nii_map: dict[str, pd.Series],
    display_names: dict[str, str] | None = None,
    slope_window: int = 14,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    多主題熱度排行榜。

    Parameters
    ----------
    topic_nii_map : dict[str, pd.Series]
        {topic_key: NII Series}
    display_names : dict[str, str], optional
        {topic_key: "顯示名稱"}，未提供則用 topic_key
    slope_window : int
        斜率計算窗口（天）
    top_n : int
        回傳前 N 個主題（依「預熱優先 + 斜率」排序）

    Returns
    -------
    pd.DataFrame
        columns: topic, display_name, nii_latest, nii_mean, nii_zscore,
                 nii_7d_slope, nii_30d_pct_change, phase, phase_emoji, rank
    """
    rows = []
    for topic, nii in topic_nii_map.items():
        valid = nii.dropna()
        if len(valid) < 2:
            continue

        name = (display_names or {}).get(topic, topic)
        latest = float(valid.iloc[-1])
        mu = float(valid.mean())
        sigma = float(valid.std())
        zscore = (latest - mu) / sigma if sigma > 0 else 0.0

        # 7 天斜率（短期加速）
        slope_7d = compute_nii_slope(valid, window=min(7, len(valid) - 1))
        s7 = float(slope_7d.dropna().iloc[-1]) if not slope_7d.dropna().empty else 0.0

        # 30 天斜率（中期趨勢）
        slope_30d = compute_nii_slope(valid, window=min(30, len(valid) - 1))
        s30 = float(slope_30d.dropna().iloc[-1]) if not slope_30d.dropna().empty else 0.0

        # 30 天百分比變化
        pct_30d = float(
            (valid.iloc[-1] - valid.iloc[max(0, len(valid) - 30)]) /
            (valid.iloc[max(0, len(valid) - 30)] + 1e-9) * 100
        )

        phase = detect_phase(valid, slope_window=slope_window)

        rows.append({
            "topic":            topic,
            "display_name":     name,
            "nii_latest":       round(latest, 2),
            "nii_mean":         round(mu, 2),
            "nii_zscore":       round(zscore, 3),
            "nii_7d_slope":     round(s7, 4),
            "nii_30d_slope":    round(s30, 4),
            "nii_30d_pct_chg":  round(pct_30d, 1),
            "phase":            phase,
            "phase_emoji":      PHASE_EMOJI.get(phase, ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 排序：預熱 > 發燒 > 降溫 > 冷卻，同階段內依 7d 斜率排
    df["_phase_pri"] = df["phase"].map(PHASE_PRIORITY).fillna(99)
    df = df.sort_values(["_phase_pri", "nii_7d_slope"], ascending=[True, False])
    df["rank"] = range(1, len(df) + 1)
    df = df.drop(columns=["_phase_pri"]).reset_index(drop=True)

    return df.head(top_n)
