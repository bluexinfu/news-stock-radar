"""
Phase 4-C：自動題材→成份股匹配

從廣域新聞（news_broad）的標題中，找出哪些台股
在特定題材討論中頻繁出現，並用 NII 報酬相關性輔助排名，
輸出候選 topics.yaml 條目供人工確認。

演算法：
  1. 字串比對（公司名 → ticker）掃描新聞標題
  2. 按照 source（關鍵字分類）分組計算提及頻率
  3. 若有 NII 與股價資料，計算 NII-報酬相關性作為第二分數
  4. 輸出 Markdown 報告 + YAML 草稿

執行：
    python -m src.analyzers.stock_matcher
    python -m src.analyzers.stock_matcher --topic cowos --top 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

# ── 台股名稱→代碼對照表（常見個股，可自行擴充）────────────────────
TW_STOCK_MAP: dict[str, tuple[str, str]] = {
    # 台積電生態系
    "台積電": ("2330.TW",  "台積電"),
    "TSMC":   ("2330.TW",  "台積電"),
    # CoWoS / 先進封裝
    "京鼎":   ("3413.TW",  "京鼎"),
    "辛耘":   ("3583.TW",  "辛耘"),
    "家登":   ("3680.TWO", "家登"),
    "弘塑":   ("3131.TWO", "弘塑"),
    # HBM / ABF
    "欣興":   ("3037.TW",  "欣興"),
    "南電":   ("8046.TW",  "南電"),
    "景碩":   ("3189.TW",  "景碩"),
    "力成":   ("6239.TW",  "力成"),
    # 矽光子 / 光通訊
    "聯亞":   ("3081.TWO", "聯亞"),
    "嘉澤":   ("3533.TW",  "嘉澤"),
    "光聖":   ("6245.TWO", "光聖"),
    "揚明光": ("3504.TW",  "揚明光"),
    "源傑":   ("6492.TWO", "源傑"),
    "上詮":   ("3363.TWO", "上詮"),
    "華星光通":("4979.TWO","華星光通"),
    # AI 伺服器
    "緯穎":   ("6669.TW",  "緯穎"),
    "廣達":   ("2382.TW",  "廣達"),
    "技嘉":   ("2376.TW",  "技嘉"),
    "英業達": ("2356.TW",  "英業達"),
    "鴻海":   ("2317.TW",  "鴻海"),
    "緯創":   ("3231.TW",  "緯創"),
    "仁寶":   ("2324.TW",  "仁寶"),
    "和碩":   ("4938.TW",  "和碩"),
    # 電源管理
    "台達電": ("2308.TW",  "台達電"),
    "漢磊":   ("3707.TWO", "漢磊"),
    "飛宏":   ("2457.TW",  "飛宏"),
    "強茂":   ("2481.TW",  "強茂"),
    "士電":   ("1503.TW",  "士電"),
    # 被動元件
    "國巨":   ("2327.TW",  "國巨"),
    "禾伸堂": ("3026.TW",  "禾伸堂"),
    "華新科": ("2492.TW",  "華新科"),
    "信昌電": ("6173.TWO", "信昌電"),
    "立隆電": ("2472.TW",  "立隆電"),
    "大毅":   ("2478.TW",  "大毅"),
    # 其他常見大型股
    "聯發科": ("2454.TW",  "聯發科"),
    "聯電":   ("2303.TW",  "聯電"),
    "瑞昱":   ("2379.TW",  "瑞昱"),
    "日月光": ("3711.TW",  "日月光"),
    "矽品":   ("2325.TW",  "矽品"),
    "群聯":   ("8299.TW",  "群聯"),
    "旺宏":   ("2337.TW",  "旺宏"),
    "華邦電": ("2344.TW",  "華邦電"),
    "南亞科": ("2408.TW",  "南亞科"),
    "鴻準":   ("2354.TW",  "鴻準"),
    "光寶科": ("2301.TW",  "光寶科"),
    "大立光": ("3008.TW",  "大立光"),
    "台光電": ("2383.TW",  "台光電"),
    "金像電": ("2368.TW",  "金像電"),
    "華通":   ("2313.TW",  "華通"),
    "耀華":   ("2367.TW",  "耀華"),
}


# ── 新聞載入 ─────────────────────────────────────────────────────────

def load_broad_news() -> pd.DataFrame:
    news_dir = ROOT / "data" / "raw" / "news_broad"
    files = sorted(news_dir.glob("broad_*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到廣域新聞資料：{news_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).drop_duplicates("title")
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
    return df


# ── 提及次數計算 ──────────────────────────────────────────────────────

def count_mentions(news_df: pd.DataFrame, source_filter: str | None = None) -> pd.DataFrame:
    """
    掃描新聞標題，計算各公司被提及次數。

    Parameters
    ----------
    news_df : pd.DataFrame
        廣域新聞 DataFrame
    source_filter : str, optional
        只看特定 source 類別（e.g. "AI概念股"）

    Returns
    -------
    pd.DataFrame
        columns: [ticker, name, mention_count, source_list]
    """
    if source_filter:
        mask = news_df["source"].str.contains(source_filter, na=False, case=False)
        subset = news_df[mask]
    else:
        subset = news_df

    rows = []
    for company, (ticker, name) in TW_STOCK_MAP.items():
        count = subset["title"].str.contains(company, na=False).sum()
        if count > 0:
            rows.append({
                "ticker": ticker,
                "name": name,
                "mention_count": int(count),
                "company_keyword": company,
            })

    if not rows:
        return pd.DataFrame(columns=["ticker", "name", "mention_count", "company_keyword"])

    df = pd.DataFrame(rows)
    # 相同 ticker 的多個關鍵字合併
    df = (
        df.groupby(["ticker", "name"])
        .agg(mention_count=("mention_count", "sum"), keywords=("company_keyword", list))
        .reset_index()
        .sort_values("mention_count", ascending=False)
    )
    return df


# ── NII 相關性計算 ────────────────────────────────────────────────────

def compute_nii_correlation(ticker: str, nii: pd.Series) -> float | None:
    """計算個股報酬與 NII 的 Pearson 相關（lag=0）。"""
    prices_dir = ROOT / "data" / "raw" / "prices"
    # 找對應的 prices 檔案
    ticker_key = ticker.replace(".", "_")
    matches = list(prices_dir.glob(f"*{ticker_key}*.parquet"))
    if not matches:
        return None
    try:
        price_df = pd.read_parquet(matches[0])
        if "close" not in price_df.columns:
            return None
        returns = price_df["close"].pct_change().dropna()
        returns.index = pd.to_datetime(returns.index)
        nii_idx = pd.to_datetime(nii.index)
        common = returns.index.intersection(nii_idx)
        if len(common) < 20:
            return None
        return float(returns.loc[common].corr(nii.loc[common]))
    except Exception:
        return None


# ── 主函式 ───────────────────────────────────────────────────────────

def match_stocks(
    topic: str | None = None,
    top_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """
    對每個主題的關鍵字，找出新聞中提及最多的股票，
    並附上 NII 相關性分數。

    Parameters
    ----------
    topic : str, optional
        指定單一主題 key；None = 所有主題
    top_n : int
        每個主題取前幾名

    Returns
    -------
    dict[str, pd.DataFrame]
        {topic_key: 候選股 DataFrame}
    """
    with open(ROOT / "config" / "topics.yaml", encoding="utf-8") as f:
        topics_cfg = yaml.safe_load(f)

    if topic:
        topics_cfg = {k: v for k, v in topics_cfg.items() if k == topic}
    if not topics_cfg:
        log.warning("找不到主題：%s", topic)
        return {}

    news_df = load_broad_news()
    results: dict[str, pd.DataFrame] = {}

    for topic_key, cfg in topics_cfg.items():
        display = cfg.get("display_name", topic_key)
        keywords = (
            cfg.get("keywords", {}).get("chinese", []) +
            cfg.get("keywords", {}).get("primary", [])
        )
        # 用主題關鍵字過濾相關新聞
        if keywords:
            pattern = "|".join(keywords)
            mask = news_df["title"].str.contains(pattern, na=False, case=False)
            topic_news = news_df[mask]
        else:
            topic_news = news_df

        if topic_news.empty:
            log.info("[%s] 無相關新聞，跳過", topic_key)
            continue

        log.info("[%s] %s — 相關新聞 %d 篇", topic_key, display, len(topic_news))

        # 提及次數
        mentions = count_mentions(topic_news)
        if mentions.empty:
            log.info("[%s] 無識別到已知股票", topic_key)
            continue

        # 載入 NII 計算相關性
        nii_path = ROOT / "data" / "processed" / f"{topic_key}_nii.parquet"
        nii = None
        if nii_path.exists():
            nii_df = pd.read_parquet(nii_path)
            nii = nii_df["nii"].dropna() if "nii" in nii_df.columns else None

        if nii is not None:
            mentions["nii_corr"] = mentions["ticker"].apply(
                lambda t: compute_nii_correlation(t, nii)
            )
        else:
            mentions["nii_corr"] = None

        # 排除已在 topics.yaml 的主要概念股（避免重複建議）
        existing_tickers = set()
        for stock in cfg.get("related_stocks", {}).get("primary", []):
            existing_tickers.add(stock.get("ticker", ""))
        for stock in cfg.get("related_stocks", {}).get("secondary", []):
            existing_tickers.add(stock.get("ticker", ""))

        new_candidates = mentions[~mentions["ticker"].isin(existing_tickers)]
        results[topic_key] = new_candidates.head(top_n)

    return results


def print_report(results: dict[str, pd.DataFrame]) -> None:
    """列印 Markdown 格式報告。"""
    if not results:
        print("無候選股票")
        return

    print("\n" + "=" * 60)
    print("Phase 4-C：自動題材→候選成份股匹配報告")
    print("=" * 60)
    print("⚠️  以下為系統自動建議，請人工確認後再加入 topics.yaml\n")

    for topic_key, df in results.items():
        if df.empty:
            continue
        print(f"\n### {topic_key}")
        print(f"{'排名':<4} {'代碼':<12} {'公司':<10} {'提及次數':>6} {'NII相關':>8}")
        print("-" * 45)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            corr_str = f"{row['nii_corr']:.3f}" if pd.notna(row.get("nii_corr")) else "  N/A"
            print(f"{i:<4} {row['ticker']:<12} {row['name']:<10} {row['mention_count']:>6} {corr_str:>8}")

    print("\n" + "=" * 60)
    print("📋 YAML 草稿（複製到 topics.yaml 修改後使用）")
    print("=" * 60)
    for topic_key, df in results.items():
        if df.empty:
            continue
        print(f"\n# ── {topic_key} 候選新增股票 ────")
        for _, row in df.iterrows():
            print(f"      - {{ ticker: \"{row['ticker']}\",  "
                  f"name: \"{row['name']}\",  role: \"待確認（提及{row['mention_count']}次）\" }}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Phase 4-C：自動題材→成份股匹配")
    parser.add_argument("--topic", default=None, help="指定主題 key（預設：全部）")
    parser.add_argument("--top", type=int, default=8, help="每主題取前幾名（預設：8）")
    args = parser.parse_args()

    results = match_stocks(topic=args.topic, top_n=args.top)
    print_report(results)
