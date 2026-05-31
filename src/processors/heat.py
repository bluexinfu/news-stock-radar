"""
市場熱度指標 v2（Attention Heat Index）計算模組

取代有結構缺陷的舊 NII（舊 NII 約 99% 由 Google Trends 主導，且尺度錯配）。
本模組以「單篇內容層級加權合成」重新定義熱度，根治尺度問題。

設計（依與使用者討論定案）：
  - 來源：權威加權新聞（主）+ YouTube 觀看加權（輔）；股價僅作對照不入指標
  - 合成：各來源全域正規化到 0~100 後加權（預設 新聞 0.8 + YouTube 0.2）
  - 輸出：綜合熱度（水位）、兩個子分數、加速度、相位

輸出：
  data/processed/<topic>_heat.parquet
    columns: [news_level, youtube_level, news_norm, youtube_norm,
              composite, accel, phase]

設定檔：
  config/source_authority.yaml   媒體可信度權重
  config/youtube_channels.yaml    YouTube 頻道權重 + 訂閱門檻

執行方式：
  python -m src.processors.heat              # 全題材
  python -m src.processors.heat --topic cowos
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
NEWS_DIR = ROOT / "data" / "raw" / "news"
YT_DIR = ROOT / "data" / "raw" / "youtube"
CONFIG_DIR = ROOT / "config"

# 合成權重（設計選項 B：新聞主導，YouTube 輔助）
W_NEWS = 0.8
W_YOUTUBE = 0.2

PHASE_EMOJI = {"冷卻": "❄️", "預熱": "🌡️", "發燒": "🔥", "降溫": "📉"}


# ── 設定載入 ─────────────────────────────────────────────────────────

def load_source_authority() -> tuple[dict[str, float], float]:
    f = CONFIG_DIR / "source_authority.yaml"
    if not f.exists():
        return {}, 0.3
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    weights = {str(k).strip().lower(): float(v) for k, v in cfg.get("sources", {}).items()}
    return weights, float(cfg.get("default", 0.3))


def load_channel_config() -> tuple[dict[str, float], float, int]:
    f = CONFIG_DIR / "youtube_channels.yaml"
    if not f.exists():
        return {}, 0.5, 0
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    weights = {str(k).strip().lower(): float(v) for k, v in cfg.get("channels", {}).items()}
    return weights, float(cfg.get("default", 0.5)), int(cfg.get("min_subscribers", 0))


def get_calendar(topic: str) -> pd.DatetimeIndex | None:
    """以該題材既有的交易日索引為時間軸基準（沿用 NII 的日曆）。"""
    p = PROCESSED_DIR / f"{topic}_nii.parquet"
    if not p.exists():
        return None
    nii = pd.read_parquet(p)
    return pd.to_datetime(nii.index)


# ── 來源熱度（單篇內容層級加權）─────────────────────────────────────

def news_heat_daily(topic: str, calendar: pd.DatetimeIndex,
                    weights: dict, default: float, smooth: int = 7) -> pd.Series:
    """每日權威加權新聞量（去重後 Σ 來源權重），對齊交易日 + 7 日平滑。"""
    f = NEWS_DIR / f"{topic}_googlenews.parquet"
    if not f.exists():
        return pd.Series(0.0, index=calendar)
    df = pd.read_parquet(f).drop_duplicates(subset=["url"])
    df["w"] = df["source"].map(lambda s: weights.get(str(s).strip().lower(), default))
    df["date"] = pd.to_datetime(df["published"]).dt.normalize()
    daily = df.groupby("date")["w"].sum()
    # 非交易日併入下一交易日
    aligned = pd.Series(0.0, index=calendar)
    for d, v in daily.items():
        future = calendar[calendar >= d]
        if len(future):
            aligned.loc[future[0]] += v
    return aligned.rolling(smooth, min_periods=1).mean()


def youtube_heat_daily(topic: str, calendar: pd.DatetimeIndex, smooth: int = 7) -> pd.Series:
    """每日 YouTube 加權觸及（reach 已在採集時套用頻道權重+訂閱門檻），7 日滾動和。"""
    f = YT_DIR / f"{topic}_youtube.parquet"
    if not f.exists():
        return pd.Series(0.0, index=calendar)
    df = pd.read_parquet(f)
    df.index = pd.to_datetime(df.index)
    reach = df["reach"].groupby(df.index).sum().reindex(calendar, fill_value=0.0)
    return reach.rolling(smooth, min_periods=1).sum()


# ── 相位（算在新綜合熱度上）─────────────────────────────────────────

def rolling_slope(s: pd.Series, window: int = 7) -> pd.Series:
    x = np.arange(window, dtype=float)
    xm, xv = x.mean(), ((x - x.mean()) ** 2).sum()
    return s.rolling(window).apply(
        lambda y: np.nan if np.isnan(y).any() else ((x - xm) * (y - y.mean())).sum() / xv,
        raw=True)


def detect_phase(heat: pd.Series, slope_window: int = 14) -> pd.Series:
    """
    逐日相位（沿用舊系統四相位定義，但算在新綜合熱度上）：
      發燒 NII≥mean+1σ｜降溫 ≥mean且斜率<0｜預熱 <mean且斜率>0｜冷卻 其餘
    用擴張窗口（point-in-time）計算 mean/std，與上線時逐日運作一致。
    """
    valid = heat.dropna()
    slopes = rolling_slope(valid, slope_window)
    phases = pd.Series(index=valid.index, dtype=object)
    for i in range(len(valid)):
        if i < slope_window:
            phases.iloc[i] = "冷卻"; continue
        w = valid.iloc[: i + 1]
        mu, sigma, latest = w.mean(), w.std(), float(valid.iloc[i])
        slope = slopes.iloc[i]; slope = 0.0 if pd.isna(slope) else float(slope)
        if latest >= mu + sigma:
            phases.iloc[i] = "發燒"
        elif latest >= mu:
            phases.iloc[i] = "降溫" if slope < 0 else "發燒"
        else:
            phases.iloc[i] = "預熱" if slope > 0 else "冷卻"
    return phases.reindex(heat.index)


# ── 主計算（兩段式：全域正規化需跨題材）──────────────────────────────

def build_all_heat(topics: list[str], w_news: float = W_NEWS,
                   w_youtube: float = W_YOUTUBE, write: bool = True
                   ) -> dict[str, pd.DataFrame]:
    """
    計算所有題材的熱度。全域正規化（跨題材 min-max）以保留題材間可比性，
    避免舊 NII 的尺度錯配。回傳 {topic: DataFrame}，並可選擇寫入 parquet。
    """
    src_w, src_default = load_source_authority()

    # Pass 1：各題材的新聞/YouTube 原始水位
    news_raw, yt_raw, cals = {}, {}, {}
    for t in topics:
        cal = get_calendar(t)
        if cal is None:
            log.warning("[%s] 無 NII 日曆，跳過", t); continue
        cals[t] = cal
        news_raw[t] = news_heat_daily(t, cal, src_w, src_default)
        yt_raw[t] = youtube_heat_daily(t, cal)

    if not news_raw:
        return {}

    # 全域正規化邊界（跨題材池化）
    n_lo = min(s.min() for s in news_raw.values()); n_hi = max(s.max() for s in news_raw.values())
    y_lo = min(s.min() for s in yt_raw.values()); y_hi = max(s.max() for s in yt_raw.values())
    wsum = w_news + w_youtube
    wn, wy = w_news / wsum, w_youtube / wsum

    out = {}
    for t in news_raw:
        nn = (news_raw[t] - n_lo) / (n_hi - n_lo + 1e-9) * 100
        yy = (yt_raw[t] - y_lo) / (y_hi - y_lo + 1e-9) * 100
        comp = wn * nn + wy * yy
        df = pd.DataFrame({
            "news_level": news_raw[t],
            "youtube_level": yt_raw[t],
            "news_norm": nn,
            "youtube_norm": yy,
            "composite": comp,
            "accel": rolling_slope(comp, 7),
            "phase": detect_phase(comp),
        })
        out[t] = df
        if write:
            df.to_parquet(PROCESSED_DIR / f"{t}_heat.parquet")
            log.info("[%s] 熱度已輸出（綜合均值=%.1f，最新相位=%s）",
                     t, comp.mean(), df["phase"].iloc[-1])
    return out


def load_topics() -> list[str]:
    cfg = yaml.safe_load(open(CONFIG_DIR / "topics.yaml", encoding="utf-8"))
    return [t for t, c in cfg.items() if isinstance(c, dict)]


def main() -> None:
    ap = argparse.ArgumentParser(description="市場熱度指標 v2")
    ap.add_argument("--topic", help="只算單一題材（預設全部）")
    ap.add_argument("--w-news", type=float, default=W_NEWS)
    ap.add_argument("--w-youtube", type=float, default=W_YOUTUBE)
    args = ap.parse_args()

    topics = [args.topic] if args.topic else load_topics()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    build_all_heat(topics, args.w_news, args.w_youtube, write=True)


if __name__ == "__main__":
    main()
