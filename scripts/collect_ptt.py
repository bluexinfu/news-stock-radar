#!/usr/bin/env python3
"""
PTT Stock 板 投資人討論熱度採集器（P2 原型）

範圍：近 1 個月 + 題材關鍵字匹配 + 只 Stock 板

方法：
  1. 從 Stock 板最新頁往回翻，直到超過 N 天前（跳過置底公告）
  2. 每篇貼文解析：日期、推噓數、標題
  3. 標題若含某題材關鍵字（CoWoS、HBM、光通訊…）→ 歸屬該題材
     （改用題材關鍵字而非個股代號：避免台積電等共用股洗掉題材差異）
  4. 每題材每日熱度貢獻 = Σ 匹配貼文：log(1 + 互動分)

⚠️ 原型結論：PTT 標題層級的題材訊號「太稀疏」（整月個位數貼文），
   因投資人多以個股發文、題材詞藏在內文。經評估 PTT 為弱來源，
   熱度合成（heat_composite.py）暫不納入。保留本採集器供未來
   以「讀內文」方式增強時參考。

輸出：data/raw/ptt/{topic}_ptt.parquet（index=date, 欄位 post_count, reach）

用法：
  python scripts/collect_ptt.py --days 30
  python scripts/collect_ptt.py --days 7 --max-pages 40   # 快速驗證
"""

from __future__ import annotations

import argparse
import math
import re
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "ptt"
BASE = "https://www.ptt.cc"
BOARD = "Stock"
UA = {"User-Agent": "Mozilla/5.0 (research; news-stock-radar)"}

PUSH_RE = re.compile(r'<div class="nrec">(?:<span[^>]*>)?([^<]*)')
TITLE_RE = re.compile(r'<div class="title">\s*(?:<a[^>]*>)?([^<]+)')
DATE_RE = re.compile(r'<div class="date">\s*([0-9/]+)\s*</div>')
PREV_RE = re.compile(r'href="(/bbs/Stock/index(\d+)\.html)">\s*&lsaquo;')


def split_entries(html: str) -> list[str]:
    """以 r-ent 切分，每段為一篇貼文的完整 HTML（含 meta/date）。"""
    parts = html.split('<div class="r-ent">')
    return parts[1:]  # 第一段是頁首，丟棄


def fetch(url: str, delay: float) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode("utf-8", errors="ignore")
    time.sleep(delay)
    return html


def parse_push(s: str) -> int:
    s = s.strip()
    if s == "爆":
        return 100
    if s.startswith("X"):
        rest = s[1:]
        if rest == "X":
            return 100
        return int(rest) * 10 if rest.isdigit() else 50
    return int(s) if s.isdigit() else 0


def load_topic_keywords() -> dict[str, set[str]]:
    """
    回傳 {題材關鍵字(小寫): {topic,...}}。

    PTT 投資人以「題材名」討論（CoWoS 缺貨、HBM 概念、光通訊噴），
    故用題材關鍵字匹配，而非個股代號（避免台積電等共用股洗掉題材差異）。
    """
    cfg = yaml.safe_load(open(ROOT / "config" / "topics.yaml", encoding="utf-8"))
    lookup: dict[str, set[str]] = {}
    for topic, c in cfg.items():
        if not isinstance(c, dict):
            continue
        kw = c.get("keywords", {})
        # 中文 + primary 縮寫（CoWoS/HBM/CPO 等會出現在中文標題裡）
        terms = list(kw.get("chinese", [])) + list(kw.get("primary", []))
        for t in terms:
            t = str(t).strip().lower()
            if len(t) >= 2:                # 過濾過短的字
                lookup.setdefault(t, set()).add(topic)
    return lookup


def match_topics(title: str, lookup: dict[str, set[str]]) -> set[str]:
    low = title.lower()
    hit: set[str] = set()
    for key, topics in lookup.items():
        if key in low:
            hit |= topics
    return hit


def main() -> None:
    ap = argparse.ArgumentParser(description="PTT Stock 板熱度採集（原型）")
    ap.add_argument("--days", type=int, default=30, help="回溯天數")
    ap.add_argument("--max-pages", type=int, default=300, help="頁數上限（保護）")
    ap.add_argument("--delay", type=float, default=0.6, help="每頁延遲秒數")
    args = ap.parse_args()

    cutoff = (datetime.now() - timedelta(days=args.days)).date()
    lookup = load_topic_keywords()
    print(f"題材個股關鍵字數：{len(lookup)}　回溯至：{cutoff}")

    # 取得最新頁碼
    idx = fetch(f"{BASE}/bbs/{BOARD}/index.html", args.delay)
    m = PREV_RE.search(idx)
    if not m:
        print("✗ 找不到上頁連結，PTT 結構可能改變"); return
    cur = int(m.group(2)) + 1   # 上頁是 latest-1，故 latest = +1

    # 逐日累積：{(topic,date): [post_count, reach_sum]}
    agg: dict[tuple[str, str], list] = {}
    pages = 0
    stop = False
    now_year = datetime.now().year

    while pages < args.max_pages and not stop:
        url = f"{BASE}/bbs/{BOARD}/index{cur}.html"
        try:
            html = fetch(url, args.delay)
        except Exception as e:
            print(f"  頁 {cur} 讀取失敗：{e}"); break
        pages += 1
        page_oldest = None
        for ent in split_entries(html):
            t = TITLE_RE.search(ent)
            d = DATE_RE.search(ent)
            if not t or not d:
                continue
            title = t.group(1).strip()
            if "刪除" in title:                  # 刪除文
                continue
            if title.startswith("[公告]"):       # 置底公告（舊日期會污染停止判斷）
                continue
            mm_dd = d.group(1).strip()
            try:
                mo, da = [int(x) for x in mm_dd.split("/")]
            except ValueError:
                continue
            # 年份推斷（跨年處理）
            yr = now_year if mo <= datetime.now().month else now_year - 1
            try:
                pdate = datetime(yr, mo, da).date()
            except ValueError:
                continue
            page_oldest = pdate if page_oldest is None else min(page_oldest, pdate)
            if pdate < cutoff:
                continue
            topics = match_topics(title, lookup)
            if not topics:
                continue
            push = parse_push(PUSH_RE.search(ent).group(1) if PUSH_RE.search(ent) else "")
            reach = math.log1p(push)
            for tp in topics:
                k = (tp, pdate.isoformat())
                a = agg.setdefault(k, [0, 0.0])
                a[0] += 1
                a[1] += reach
        if page_oldest and page_oldest < cutoff:
            stop = True
        cur -= 1

    print(f"共爬 {pages} 頁，匹配到 {len(agg)} 個 (題材,日期) 組合")

    # 整理成每題材的 parquet
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_topic: dict[str, list] = {}
    for (tp, ds), (cnt, reach) in agg.items():
        by_topic.setdefault(tp, []).append({"date": ds, "post_count": cnt, "reach": reach})

    cfg = yaml.safe_load(open(ROOT / "config" / "topics.yaml", encoding="utf-8"))
    print("\n各題材 PTT 討論熱度（近 %d 天彙整）：" % args.days)
    print(f"{'題材':<16}{'總貼文':>8}{'總觸及':>10}{'有討論天數':>12}")
    print("-" * 48)
    for tp in cfg:
        if not isinstance(cfg[tp], dict) or tp not in by_topic:
            continue
        df = pd.DataFrame(by_topic[tp])
        df["date"] = pd.to_datetime(df["date"])
        df = df.groupby("date").sum().sort_index()
        df.to_parquet(OUT_DIR / f"{tp}_ptt.parquet")
        dn = cfg[tp]["display_name"]
        print(f"{dn:<16}{int(df['post_count'].sum()):>8}{df['reach'].sum():>10.1f}{len(df):>12}")
    print(f"\n✅ 已輸出至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
