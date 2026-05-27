"""
籌碼訊號計算（Phase 4-B）

從 chips.py 採集的個股三大法人 + 融資資料，
聚合為「題材層級」的籌碼訊號時序，供與 NII 複合。

計算邏輯：
  - 外資訊號 (foreign_signal)：
      topic 內各股 7d 滾動外資買超（標準化後取均值）
      正 = 外資整體在買  負 = 外資整體在賣

  - 投信訊號 (trust_signal)：
      topic 內各股 7d 滾動投信買超（標準化後取均值）

  - 法人合力訊號 (institution_signal)：
      foreign + trust 的均值（自營商避險性質，排除）

  - 融資信號 (margin_signal)：
      topic 內各股 7d 融資餘額變化率均值
      正 = 散戶借錢追高（可能過熱）  負 = 散戶退場

  - 複合籌碼訊號 (composite_chips):
      institution_signal * 0.6 + (-margin_signal * 0.4)
      假設：法人買 + 散戶退 = 最強複合訊號

輸出：
  data/processed/{topic}_chips_signal.parquet
  columns: date, foreign_signal, trust_signal, institution_signal,
           margin_signal, composite_chips

用法：
  from src.processors.chips_signal import build_chips_signal
  df = build_chips_signal("optical_comm")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

ROOT          = Path(__file__).resolve().parents[2]
CHIPS_DIR     = ROOT / "data" / "raw"    / "chips"
PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_PATH   = ROOT / "config" / "topics.yaml"


def _zscore(s: pd.Series, min_periods: int = 10) -> pd.Series:
    """
    滾動 Z-score（30日視窗），分母加 ε 防零除。
    資料少於 min_periods 時自適應降低要求（最少 3 筆）。
    """
    n = s.notna().sum()
    effective_min = max(3, min(min_periods, n))
    mu  = s.rolling(30, min_periods=effective_min).mean()
    std = s.rolling(30, min_periods=effective_min).std().clip(lower=1e-6)
    return (s - mu) / std


def build_chips_signal(
    topic: str,
    window: int = 7,
    save:   bool = True,
) -> pd.DataFrame:
    """
    計算 topic 層級的每日籌碼訊號。

    Parameters
    ----------
    topic  : str  — topics.yaml 中的主題 key
    window : int  — 滾動天數（預設 7）
    save   : bool — 是否存到 data/processed/

    Returns
    -------
    DataFrame  index=date
      columns: foreign_signal, trust_signal, institution_signal,
               margin_signal, composite_chips
    若無籌碼資料則回傳空 DataFrame
    """
    inst_path   = CHIPS_DIR / f"{topic}_institutional.parquet"
    margin_path = CHIPS_DIR / f"{topic}_margin.parquet"

    if not inst_path.exists() and not margin_path.exists():
        log.warning("[%s] 無籌碼資料（請先執行 chips.py collect）", topic)
        return pd.DataFrame()

    # ── 載入資料 ──────────────────────────────────────────────────────
    inst_df   = pd.read_parquet(inst_path)   if inst_path.exists()   else pd.DataFrame()
    margin_df = pd.read_parquet(margin_path) if margin_path.exists() else pd.DataFrame()

    # 讀取所有監控股票清單
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)[topic]

    stocks = cfg.get("related_stocks", {})
    all_tickers = (
        [s["ticker"] for s in stocks.get("primary",   [])] +
        [s["ticker"] for s in stocks.get("secondary", [])]
    )
    ticker_codes = [t.split(".")[0] for t in all_tickers]

    # ── 三大法人訊號 ──────────────────────────────────────────────────
    if not inst_df.empty:
        inst_df["date"] = pd.to_datetime(inst_df["date"])
        inst_sub = inst_df[inst_df["ticker"].isin(ticker_codes)].copy()

        if inst_sub.empty:
            log.warning("[%s] inst_df 內無符合的 ticker", topic)
            foreign_signal = pd.Series(dtype=float, name="foreign_signal")
            trust_signal   = pd.Series(dtype=float, name="trust_signal")
        else:
            # 轉為寬表：每個 ticker 一欄
            foreign_wide = (
                inst_sub.pivot_table(index="date", columns="ticker", values="foreign_net")
                .sort_index()
            )
            trust_wide = (
                inst_sub.pivot_table(index="date", columns="ticker", values="trust_net")
                .sort_index()
            )

            # 滾動 Z-score 標準化後取各股均值
            foreign_zs = foreign_wide.apply(_zscore).rolling(window, min_periods=1).sum()
            trust_zs   = trust_wide.apply(_zscore).rolling(window, min_periods=1).sum()

            foreign_signal = foreign_zs.mean(axis=1).rename("foreign_signal")
            trust_signal   = trust_zs.mean(axis=1).rename("trust_signal")
    else:
        foreign_signal = pd.Series(dtype=float, name="foreign_signal")
        trust_signal   = pd.Series(dtype=float, name="trust_signal")

    # ── 融資訊號 ──────────────────────────────────────────────────────
    if not margin_df.empty:
        margin_df["date"] = pd.to_datetime(margin_df["date"])
        margin_sub = margin_df[margin_df["ticker"].isin(ticker_codes)].copy()

        if not margin_sub.empty:
            margin_wide = (
                margin_sub.pivot_table(index="date", columns="ticker", values="margin_balance")
                .sort_index()
            )
            # 融資 7d 變化率（pct_change）
            margin_chg = margin_wide.pct_change(window).fillna(0)
            margin_signal = margin_chg.mean(axis=1).rename("margin_signal")
        else:
            margin_signal = pd.Series(dtype=float, name="margin_signal")
    else:
        margin_signal = pd.Series(dtype=float, name="margin_signal")

    # ── 合併成日頻 DataFrame ──────────────────────────────────────────
    parts = [s for s in [foreign_signal, trust_signal, margin_signal] if not s.empty]
    if not parts:
        log.warning("[%s] 無任何籌碼訊號資料", topic)
        return pd.DataFrame()

    df = pd.concat(parts, axis=1).sort_index()

    # institution_signal = foreign + trust 平均
    inst_cols = [c for c in ["foreign_signal", "trust_signal"] if c in df.columns]
    if inst_cols:
        df["institution_signal"] = df[inst_cols].mean(axis=1)
    else:
        df["institution_signal"] = np.nan

    # composite_chips = 法人多頭 - 融資泡沫壓力
    # 正值 = 法人買且散戶不過熱 = 最佳複合訊號
    if "institution_signal" in df.columns and "margin_signal" in df.columns:
        df["composite_chips"] = (
            df["institution_signal"] * 0.6 +
            (-df["margin_signal"]) * 0.4
        )
    elif "institution_signal" in df.columns:
        df["composite_chips"] = df["institution_signal"]
    else:
        df["composite_chips"] = np.nan

    df.index.name = "date"

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out = PROCESSED_DIR / f"{topic}_chips_signal.parquet"
        df.to_parquet(out)
        log.info("[%s] 籌碼訊號已存：%s（%d 列）", topic, out.name, len(df))

    return df


def chips_phase(composite: float, threshold: float = 0.5) -> str:
    """
    依 composite_chips 值判斷籌碼面狀態。

    Returns
    -------
    "法人強買" | "法人買進" | "中性" | "法人賣出" | "法人強賣"
    """
    if composite >= threshold * 2:
        return "法人強買"
    elif composite >= threshold:
        return "法人買進"
    elif composite <= -threshold * 2:
        return "法人強賣"
    elif composite <= -threshold:
        return "法人賣出"
    return "中性"
