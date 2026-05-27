"""
籌碼面資料採集器（Phase 4-B）

資料來源：
  上市股票（.TW）  ─ TWSE Open API
    ・三大法人買賣超：T86 (每日，外資/投信/自營商)
    ・融資融券餘額：MI_MARGN (每日)

  上櫃股票（.TWO）─ TPEX Web API
    ・三大法人買賣超：3itrade_hedge_result.php
    ・融資融券餘額：margin_bal_result.php

輸出：
  data/raw/chips/{topic}_institutional.parquet
    columns: date, ticker, foreign_net, trust_net, dealer_net, total_net
    單位：股（上市）或張（上櫃）——不統一，比較方向即可

  data/raw/chips/{topic}_margin.parquet
    columns: date, ticker, margin_balance, short_balance
    單位：同上

用法：
  from src.collectors.chips import collect as collect_chips
  collect_chips("cowos", "2024-05-01", "2026-05-26")

  # 或 CLI
  python -m src.collectors.chips --topic cowos
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
CHIPS_DIR  = ROOT / "data" / "raw" / "chips"
CHIPS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = ROOT / "config" / "topics.yaml"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
})


# ── 輔助函式 ─────────────────────────────────────────────────────────

def _to_int(s: str) -> int:
    """把逗號數字字串轉為 int；不可解析回傳 0。"""
    try:
        return int(str(s).replace(",", "").strip())
    except (ValueError, AttributeError):
        return 0


def _roc_date(d: date) -> str:
    """西元年轉民國年，格式 RR/MM/DD（TPEX 用）。"""
    roc_year = d.year - 1911
    return f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"


def _trading_dates(start: str, end: str) -> list[date]:
    """產生 start~end 之間所有週一到週五的日期清單（不過濾假日）。"""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    result = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:   # 0=Mon, 4=Fri
            result.append(cur)
        cur += timedelta(days=1)
    return result


# ── TWSE（上市）API ──────────────────────────────────────────────────

_TWSE_T86_URL   = "https://www.twse.com.tw/rwd/zh/fund/T86"
_TWSE_MARGN_URL = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"


def _fetch_twse_institutional(d: date, sleep_s: float = 1.5) -> pd.DataFrame:
    """
    抓取 TWSE 三大法人單日資料。

    Returns
    -------
    DataFrame: ticker, foreign_net, trust_net, dealer_net, total_net
    """
    params = {
        "date":       d.strftime("%Y%m%d"),
        "selectType": "ALLBUT0999",
        "response":   "json",
    }
    try:
        r = _SESSION.get(_TWSE_T86_URL, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        log.warning("TWSE T86 %s 失敗：%s", d, e)
        return pd.DataFrame()
    finally:
        time.sleep(sleep_s)

    if j.get("stat") != "OK" or not j.get("data"):
        return pd.DataFrame()

    rows = []
    for rec in j["data"]:
        # field index: 0=代號, 4=外資超, 10=投信超, 11=自營超, 18=三大合計
        try:
            rows.append({
                "ticker":      rec[0].strip(),
                "foreign_net": _to_int(rec[4]),
                "trust_net":   _to_int(rec[10]),
                "dealer_net":  _to_int(rec[11]),
                "total_net":   _to_int(rec[18]),
            })
        except (IndexError, Exception):
            continue

    return pd.DataFrame(rows)


def _fetch_twse_margin(d: date, sleep_s: float = 1.5) -> pd.DataFrame:
    """
    抓取 TWSE 融資融券單日餘額。

    Returns
    -------
    DataFrame: ticker, margin_balance, short_balance
    """
    params = {
        "date":       d.strftime("%Y%m%d"),
        "selectType": "ALL",
        "response":   "json",
    }
    try:
        r = _SESSION.get(_TWSE_MARGN_URL, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        log.warning("TWSE MI_MARGN %s 失敗：%s", d, e)
        return pd.DataFrame()
    finally:
        time.sleep(sleep_s)

    if j.get("stat") != "OK" or not j.get("tables"):
        return pd.DataFrame()

    # tables[0]=彙總統計  tables[1]=個股明細
    if len(j["tables"]) < 2:
        return pd.DataFrame()

    detail = j["tables"][1]
    rows = []
    for rec in detail.get("data", []):
        # 0=代號, 1=名稱, 6=融資餘額, 12=融券餘額
        try:
            rows.append({
                "ticker":         rec[0].strip(),
                "margin_balance": _to_int(rec[6]),
                "short_balance":  _to_int(rec[12]),
            })
        except (IndexError, Exception):
            continue

    return pd.DataFrame(rows)


# ── TPEX（上櫃）API ──────────────────────────────────────────────────

_TPEX_3I_URL    = ("https://www.tpex.org.tw/web/stock/3insti/daily_trade/"
                   "3itrade_hedge_result.php")
_TPEX_MARGN_URL = ("https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/"
                   "margin_bal_result.php")


def _fetch_tpex_institutional(d: date, sleep_s: float = 1.5) -> pd.DataFrame:
    """抓取 TPEX 三大法人單日資料。"""
    params = {
        "l":  "zh-tw",
        "se": "EW",
        "t":  "D",
        "d":  _roc_date(d),
        "s":  "0,asc",
    }
    try:
        r = _SESSION.get(_TPEX_3I_URL, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        log.warning("TPEX 3I %s 失敗：%s", d, e)
        return pd.DataFrame()
    finally:
        time.sleep(sleep_s)

    tables = j.get("tables", [])
    if not tables:
        return pd.DataFrame()

    rows = []
    for rec in tables[0].get("data", []):
        # 0=代號, 4=外資超, 13=投信超, 22=自營超合計, 23=三大合計
        try:
            rows.append({
                "ticker":      rec[0].strip(),
                "foreign_net": _to_int(rec[4]),
                "trust_net":   _to_int(rec[13]),
                "dealer_net":  _to_int(rec[22]),
                "total_net":   _to_int(rec[23]),
            })
        except (IndexError, Exception):
            continue

    return pd.DataFrame(rows)


def _fetch_tpex_margin(d: date, sleep_s: float = 1.5) -> pd.DataFrame:
    """抓取 TPEX 融資融券單日餘額。"""
    params = {
        "l": "zh-tw",
        "d": _roc_date(d),
        "s": "0,asc",
    }
    try:
        r = _SESSION.get(_TPEX_MARGN_URL, params=params, timeout=15)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        log.warning("TPEX 融資 %s 失敗：%s", d, e)
        return pd.DataFrame()
    finally:
        time.sleep(sleep_s)

    tables = j.get("tables", [])
    if not tables:
        return pd.DataFrame()

    rows = []
    for rec in tables[0].get("data", []):
        # 0=代號, 6=資餘額(張), 14=券餘額(張)
        try:
            rows.append({
                "ticker":         rec[0].strip(),
                "margin_balance": _to_int(rec[6]),
                "short_balance":  _to_int(rec[14]),
            })
        except (IndexError, Exception):
            continue

    return pd.DataFrame(rows)


# ── 主採集函式 ───────────────────────────────────────────────────────

def _classify_ticker(ticker: str) -> str:
    """依 suffix 判斷是 twse 或 tpex。"""
    if ticker.endswith(".TWO"):
        return "tpex"
    return "twse"   # .TW 或 ^TWII


def fetch_chips_one_day(
    d: date,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    取得單一日期的三大法人 + 融資，篩選出 tickers 的資料。

    Returns
    -------
    (inst_df, margin_df) — 若無資料則為空 DataFrame
    """
    # 分類 TWSE / TPEX
    twse_codes = {t.split(".")[0] for t in tickers if _classify_ticker(t) == "twse"}
    tpex_codes = {t.split(".")[0] for t in tickers if _classify_ticker(t) == "tpex"}

    inst_parts   = []
    margin_parts = []

    if twse_codes:
        df_i = _fetch_twse_institutional(d)
        df_m = _fetch_twse_margin(d)
        if not df_i.empty:
            inst_parts.append(df_i[df_i["ticker"].isin(twse_codes)])
        if not df_m.empty:
            margin_parts.append(df_m[df_m["ticker"].isin(twse_codes)])

    if tpex_codes:
        df_i = _fetch_tpex_institutional(d)
        df_m = _fetch_tpex_margin(d)
        if not df_i.empty:
            inst_parts.append(df_i[df_i["ticker"].isin(tpex_codes)])
        if not df_m.empty:
            margin_parts.append(df_m[df_m["ticker"].isin(tpex_codes)])

    inst_df   = pd.concat(inst_parts,   ignore_index=True) if inst_parts   else pd.DataFrame()
    margin_df = pd.concat(margin_parts, ignore_index=True) if margin_parts else pd.DataFrame()

    return inst_df, margin_df


def collect(
    topic: str,
    start: Optional[str] = None,
    end:   Optional[str] = None,
    save:  bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    對指定主題的所有相關股票，採集每日三大法人 + 融資資料。

    Parameters
    ----------
    topic : str  — topics.yaml 中的主題 key
    start : str  — "YYYY-MM-DD"（預設從 topics.yaml 讀取）
    end   : str  — "YYYY-MM-DD"（預設今天）
    save  : bool — 是否存到 data/raw/chips/

    Returns
    -------
    (institutional_df, margin_df)
    institutional_df columns: date, ticker, foreign_net, trust_net, dealer_net, total_net
    margin_df        columns: date, ticker, margin_balance, short_balance
    """
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)[topic]

    start = start or cfg.get("time_range", {}).get("start", "2024-05-01")
    end   = end   or cfg.get("time_range", {}).get("end",
                                                    date.today().strftime("%Y-%m-%d"))

    # 收集所有相關股票（benchmark 不需要籌碼）
    stocks = cfg.get("related_stocks", {})
    tickers = (
        [s["ticker"] for s in stocks.get("primary",   [])] +
        [s["ticker"] for s in stocks.get("secondary", [])]
    )
    if not tickers:
        log.warning("[%s] 無相關股票設定，跳過籌碼採集", topic)
        return pd.DataFrame(), pd.DataFrame()

    log.info("[%s] 採集籌碼面：%s ~ %s  股票 %s", topic, start, end, tickers)

    all_inst   = []
    all_margin = []

    trading_dates = _trading_dates(start, end)
    log.info("  → %d 個交易日（含假日需過濾）", len(trading_dates))

    for d in trading_dates:
        df_i, df_m = fetch_chips_one_day(d, tickers)
        if not df_i.empty:
            df_i["date"] = d
            all_inst.append(df_i)
        if not df_m.empty:
            df_m["date"] = d
            all_margin.append(df_m)

    inst_df   = pd.concat(all_inst,   ignore_index=True) if all_inst   else pd.DataFrame()
    margin_df = pd.concat(all_margin, ignore_index=True) if all_margin else pd.DataFrame()

    # 格式化
    for df in (inst_df, margin_df):
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    if save:
        if not inst_df.empty:
            p = CHIPS_DIR / f"{topic}_institutional.parquet"
            inst_df.to_parquet(p, index=False)
            log.info("  存：%s（%d 列）", p.name, len(inst_df))
        if not margin_df.empty:
            p = CHIPS_DIR / f"{topic}_margin.parquet"
            margin_df.to_parquet(p, index=False)
            log.info("  存：%s（%d 列）", p.name, len(margin_df))

    log.info("[%s] 籌碼採集完成：三大法人 %d 列，融資 %d 列",
             topic, len(inst_df), len(margin_df))
    return inst_df, margin_df


def load_chips(topic: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入已採集的籌碼資料。

    Returns
    -------
    (institutional_df, margin_df)  — 若檔案不存在則為空 DataFrame
    """
    inst_path   = CHIPS_DIR / f"{topic}_institutional.parquet"
    margin_path = CHIPS_DIR / f"{topic}_margin.parquet"

    inst_df   = pd.read_parquet(inst_path)   if inst_path.exists()   else pd.DataFrame()
    margin_df = pd.read_parquet(margin_path) if margin_path.exists() else pd.DataFrame()

    return inst_df, margin_df


# ── 摘要統計 ─────────────────────────────────────────────────────────

def chips_summary(
    inst_df:   pd.DataFrame,
    margin_df: pd.DataFrame,
    ticker:    str,
) -> dict:
    """
    對單一股票計算籌碼摘要統計。

    Returns
    -------
    dict with keys:
      ticker, n_days,
      foreign_net_sum, trust_net_sum, total_net_sum,   # 外資/投信/三大合計
      margin_balance_latest, margin_balance_30d_chg,   # 融資餘額
      short_balance_latest, short_balance_30d_chg,     # 融券餘額
    """
    code = ticker.split(".")[0]
    result: dict = {"ticker": ticker}

    if not inst_df.empty:
        sub = inst_df[inst_df["ticker"] == code].sort_values("date")
        result["n_days"]          = len(sub)
        result["foreign_net_sum"] = int(sub["foreign_net"].sum())
        result["trust_net_sum"]   = int(sub["trust_net"].sum())
        result["total_net_sum"]   = int(sub["total_net"].sum())
    else:
        result.update(n_days=0, foreign_net_sum=0, trust_net_sum=0, total_net_sum=0)

    if not margin_df.empty:
        sub = margin_df[margin_df["ticker"] == code].sort_values("date")
        if not sub.empty:
            result["margin_balance_latest"]  = int(sub["margin_balance"].iloc[-1])
            result["short_balance_latest"]   = int(sub["short_balance"].iloc[-1])
            # 近 30 天變化
            if len(sub) >= 30:
                result["margin_balance_30d_chg"] = int(
                    sub["margin_balance"].iloc[-1] - sub["margin_balance"].iloc[-30])
                result["short_balance_30d_chg"]  = int(
                    sub["short_balance"].iloc[-1]  - sub["short_balance"].iloc[-30])
            else:
                result["margin_balance_30d_chg"] = None
                result["short_balance_30d_chg"]  = None
    else:
        result.update(margin_balance_latest=None, short_balance_latest=None,
                      margin_balance_30d_chg=None, short_balance_30d_chg=None)

    return result


# ── CLI 入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="籌碼面資料採集")
    parser.add_argument("--topic", required=True, help="topics.yaml 中的主題 key")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="YYYY-MM-DD")
    args = parser.parse_args()

    inst_df, margin_df = collect(args.topic, args.start, args.end)

    if not inst_df.empty:
        print(f"\n三大法人（{len(inst_df)} 列）：")
        print(inst_df.sort_values("date").tail())
    if not margin_df.empty:
        print(f"\n融資融券（{len(margin_df)} 列）：")
        print(margin_df.sort_values("date").tail())
