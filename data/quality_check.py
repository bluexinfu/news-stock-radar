"""
資料品質檢查腳本（Phase 5-C）

每次跑完管線後執行，自動驗證：
  1. 必要檔案是否存在
  2. 資料時間範圍是否符合 topics.yaml 設定
  3. NII 缺漏率（> 5% 則警告）
  4. 股價資料異常值（單日漲跌超過 ±20%）
  5. Trends 資料是否全為 0（可能限流）

用法：
  python data/quality_check.py
  python data/quality_check.py --topic cowos  （只檢查特定主題）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── ANSI 顏色 ───────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✅{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠️ {RESET} {msg}")
def error(msg): print(f"  {RED}❌{RESET} {msg}")


# ── 主要檢查函式 ─────────────────────────────────────────────────────

def check_topic(topic: str, cfg: dict) -> int:
    """
    檢查單一主題的資料品質。
    回傳錯誤數。
    """
    print(f"\n{'─'*50}")
    print(f"  {topic}: {cfg.get('display_name','')}")
    print(f"{'─'*50}")

    errors = 0
    start  = cfg.get("time_range", {}).get("start", "")
    end    = cfg.get("time_range", {}).get("end",   "")

    raw_dir       = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"

    # ── 1. 必要的 processed 檔案 ──────────────────────────────────
    required = [
        processed_dir / f"{topic}_aligned.parquet",
        processed_dir / f"{topic}_nii.parquet",
        processed_dir / f"{topic}_smi.parquet",
    ]
    for path in required:
        if path.exists():
            ok(f"{path.name} 存在")
        else:
            error(f"{path.name} 不存在（請先執行 run_pipeline.py）")
            errors += 1

    # ── 2. NII 時間範圍 + 缺漏率 ──────────────────────────────────
    nii_path = processed_dir / f"{topic}_nii.parquet"
    if nii_path.exists():
        nii_df = pd.read_parquet(nii_path)
        nii_df.index = pd.to_datetime(nii_df.index)
        nii = nii_df["nii"] if "nii" in nii_df else nii_df.iloc[:, 0]

        # 時間範圍檢查
        if start and nii_df.index.min().strftime("%Y-%m-%d") > start:
            warn(f"NII 起始日 {nii_df.index.min().date()} > 設定 {start}")
        else:
            ok(f"NII 時間：{nii_df.index.min().date()} ~ {nii_df.index.max().date()}（{len(nii_df)} 天）")

        # 缺漏率
        miss_rate = nii.isna().mean()
        if miss_rate > 0.05:
            warn(f"NII 缺漏率 {miss_rate*100:.1f}% > 5%")
        else:
            ok(f"NII 缺漏率 {miss_rate*100:.1f}%")

        # 全零檢查
        if nii.dropna().mean() < 0.1:
            error(f"NII 均值 {nii.dropna().mean():.3f} ≈ 0（可能 Trends 資料異常）")
            errors += 1
        else:
            ok(f"NII 均值 {nii.dropna().mean():.2f}，最大 {nii.dropna().max():.2f}")

    # ── 3. Trends 資料 ────────────────────────────────────────────
    trends_path = raw_dir / "trends" / f"{topic}_trends.parquet"
    if trends_path.exists():
        t_df = pd.read_parquet(trends_path)
        if t_df.max().max() < 1:
            error(f"Trends 全部為 0 或接近 0（可能被 429 限流，請重新採集）")
            errors += 1
        else:
            best_col = t_df.mean().idxmax()
            ok(f"Trends: {len(t_df)} 週，最佳欄 '{best_col}' 均值={t_df[best_col].mean():.1f}")
    else:
        warn(f"Trends 資料不存在：{trends_path.name}")

    # ── 4. 股價異常值 ─────────────────────────────────────────────
    prices_dir = raw_dir / "prices"
    price_files = list(prices_dir.glob(f"{topic}_*.parquet"))
    if not price_files:
        error(f"找不到股價資料（{topic}_*.parquet）")
        errors += 1
    else:
        outlier_count = 0
        for pf in price_files:
            try:
                df = pd.read_parquet(pf)
                if "close" in df.columns:
                    daily_ret = df["close"].pct_change().dropna()
                    outliers  = (daily_ret.abs() > 0.20).sum()
                    if outliers > 0:
                        outlier_count += 1
                        warn(f"{pf.name}: {outliers} 天單日漲跌 > ±20%")
            except Exception as e:
                warn(f"{pf.name}: 讀取失敗 {e}")
        if outlier_count == 0:
            ok(f"股價資料：{len(price_files)} 個檔案，無異常漲跌")

    # ── 5. Google News 資料 ───────────────────────────────────────
    news_path = raw_dir / "news" / f"{topic}_daily_count.parquet"
    if news_path.exists():
        cnt = pd.read_parquet(news_path)
        total = int(cnt.iloc[:, 0].sum())
        ok(f"新聞資料：共 {total} 篇")
    else:
        warn(f"新聞計數資料不存在：{news_path.name}")

    # ── 6. 籌碼資料（選用） ─────────────────────────────────────
    chips_dir = ROOT / "data" / "raw" / "chips"
    inst_path = chips_dir / f"{topic}_institutional.parquet"
    if inst_path.exists():
        inst = pd.read_parquet(inst_path)
        ok(f"籌碼資料：三大法人 {len(inst)} 列")
    else:
        warn(f"無籌碼資料（可執行 python -m src.collectors.chips --topic {topic}）")

    return errors


def run_check(topics: list[str] | None = None) -> None:
    cfg_path = ROOT / "config" / "topics.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f)

    if topics:
        cfg = {k: v for k, v in all_cfg.items() if k in topics}
        missing = set(topics) - set(cfg)
        if missing:
            print(f"{RED}找不到主題：{missing}{RESET}")
    else:
        cfg = all_cfg

    print(f"\n{'='*50}")
    print(f"  資料品質檢查：{len(cfg)} 個主題")
    print(f"{'='*50}")

    total_errors = 0
    for topic, topic_cfg in cfg.items():
        total_errors += check_topic(topic, topic_cfg)

    print(f"\n{'='*50}")
    if total_errors == 0:
        print(f"  {GREEN}✅ 全部通過，無錯誤{RESET}")
    else:
        print(f"  {RED}❌ 共 {total_errors} 個問題需要修正{RESET}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="資料品質檢查")
    parser.add_argument("--topic", nargs="+", default=None, help="只檢查特定主題")
    args = parser.parse_args()
    run_check(args.topic)
