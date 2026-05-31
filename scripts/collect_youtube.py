#!/usr/bin/env python3
"""
YouTube 財經影片 觀看熱度採集器（P2）

投資人觸及訊號的核心：用「影片觀看數」衡量題材的真實關注度。

方法：
  1. 對每個題材，用中文關鍵字搜尋 YouTube 影片（限定時間窗、台灣地區）
  2. 取得每部影片的 發布日、頻道、觀看數
  3. 每題材每日熱度貢獻 = Σ 當日發布影片：頻道權重 × log(1 + 觀看數)
     （觀看數為「目前累積」，作為該影片帶起之關注度的代理）

頻道權重：config/youtube_channels.yaml（嚴肅分析 vs 明牌台分級）；
          未列出頻道套用 default。首次執行先用預設權重，並列出出現的頻道
          供使用者分級。

環境變數：YOUTUBE_API_KEY（存於 .env，勿寫進程式碼）

⚠️ search.list 每次耗 100 quota（每日上限 10,000）。本程式有分頁上限保護。

用法：
  python scripts/collect_youtube.py --days 116        # 對齊現有資料窗
  python scripts/collect_youtube.py --days 30 --max-pages 1   # 快速驗證
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "youtube"
API = "https://www.googleapis.com/youtube/v3"


def load_api_key() -> str:
    # 優先讀環境變數（CI 用 GitHub Secret）；本機則回退讀 .env
    import os
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("YOUTUBE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("❌ 找不到 YOUTUBE_API_KEY（環境變數或 .env）")


# 部分題材的中文關鍵字對 YouTube 搜尋效果差，用更貼近財經影片的查詢覆蓋
QUERY_OVERRIDE = {
    "power_management": "台達電 儲能 電源管理",
    "passive_components": "被動元件 國巨 MLCC",
}


def load_channel_weights() -> tuple[dict, float, int]:
    f = ROOT / "config" / "youtube_channels.yaml"
    if not f.exists():
        return {}, 0.5, 0
    cfg = yaml.safe_load(open(f, encoding="utf-8"))
    weights = {str(k).strip().lower(): float(v) for k, v in cfg.get("channels", {}).items()}
    return weights, float(cfg.get("default", 0.5)), int(cfg.get("min_subscribers", 0))


def load_topic_queries() -> dict[str, tuple[str, str]]:
    """回傳 {topic: (display_name, 搜尋字串)}。用中文關鍵字組搜尋。"""
    cfg = yaml.safe_load(open(ROOT / "config" / "topics.yaml", encoding="utf-8"))
    out = {}
    for topic, c in cfg.items():
        if not isinstance(c, dict):
            continue
        zh = c.get("keywords", {}).get("chinese", [])
        query = QUERY_OVERRIDE.get(topic) or (" ".join(zh[:3]) if zh else c.get("display_name", topic))
        out[topic] = (c.get("display_name", topic), query)
    return out


def api_get(endpoint: str, params: dict) -> dict:
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    return json.loads(urllib.request.urlopen(url, timeout=20).read())


ISO = "%Y-%m-%dT%H:%M:%SZ"


def month_slices(days: int, chunk: int = 30) -> list[tuple[str, str]]:
    """把回溯窗切成 ~每月切片，確保整段時間均勻覆蓋（避免熱門題材偏近期）。"""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    out, cur = [], start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk), end)
        out.append((cur.strftime(ISO), nxt.strftime(ISO)))
        cur = nxt
    return out


def search_videos(query: str, key: str, after: str, before: str,
                  max_pages: int) -> list[dict]:
    """search.list 分頁取得影片基本資料（含 videoId、頻道、發布日）。"""
    vids, token, pages = [], None, 0
    while pages < max_pages:
        params = {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": 50, "order": "date",
            "publishedAfter": after, "publishedBefore": before,
            "regionCode": "TW", "relevanceLanguage": "zh-Hant", "key": key,
        }
        if token:
            params["pageToken"] = token
        data = api_get("search", params)
        for it in data.get("items", []):
            vids.append({
                "video_id": it["id"]["videoId"],
                "channel": it["snippet"]["channelTitle"],
                "channel_id": it["snippet"]["channelId"],
                "published": it["snippet"]["publishedAt"],
                "title": it["snippet"]["title"],
            })
        token = data.get("nextPageToken")
        pages += 1
        if not token:
            break
    return vids


def fetch_view_counts(video_ids: list[str], key: str) -> dict[str, int]:
    """videos.list 批次（每批 50）取得觀看數。每批僅耗 1 quota。"""
    views = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        data = api_get("videos", {
            "part": "statistics", "id": ",".join(batch), "key": key,
        })
        for it in data.get("items", []):
            views[it["id"]] = int(it.get("statistics", {}).get("viewCount", 0))
    return views


def fetch_subscribers(channel_ids: list[str], key: str) -> dict[str, int]:
    """channels.list 批次取得頻道訂閱數（每批 50，1 quota）。"""
    subs = {}
    uniq = list(dict.fromkeys(channel_ids))
    for i in range(0, len(uniq), 50):
        batch = uniq[i:i + 50]
        data = api_get("channels", {
            "part": "statistics", "id": ",".join(batch), "key": key,
        })
        for it in data.get("items", []):
            st = it.get("statistics", {})
            subs[it["id"]] = int(st.get("subscriberCount", 0)) if not st.get("hiddenSubscriberCount") else -1
    return subs


def main() -> None:
    ap = argparse.ArgumentParser(description="YouTube 觀看熱度採集（P2）")
    ap.add_argument("--days", type=int, default=190, help="回溯天數（涵蓋現有資料窗）")
    ap.add_argument("--max-pages", type=int, default=1, help="每題材每月切片分頁上限")
    args = ap.parse_args()

    key = load_api_key()
    ch_w, ch_default, min_subs = load_channel_weights()
    queries = load_topic_queries()
    slices = month_slices(args.days)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"回溯 {args.days} 天｜{len(slices)} 個月切片｜每切片 {args.max_pages} 頁"
          f"｜訂閱門檻 {min_subs:,}")
    print(f"切片範圍：{slices[0][0][:10]} ~ {slices[-1][1][:10]}")
    print("=" * 64)

    # 逐月切片搜尋（確保均勻覆蓋），以 video_id 去重
    topic_vids: dict[str, list] = {}
    all_chids: list[str] = []
    for topic, (dn, query) in queries.items():
        vids, seen = [], set()
        for after_iso, before_iso in slices:
            for v in search_videos(query, key, after_iso, before_iso, args.max_pages):
                if v["video_id"] not in seen:
                    seen.add(v["video_id"]); vids.append(v)
        topic_vids[topic] = vids
        all_chids += [v["channel_id"] for v in vids]
    subs = fetch_subscribers(all_chids, key)

    def channel_weight(title: str, chid: str) -> float:
        """套用：訂閱門檻 → 明確權重 → 預設。"""
        s = subs.get(chid, 0)
        explicit = ch_w.get(title.strip().lower())
        if explicit == 0.0:                    # 明確排除
            return 0.0
        if s >= 0 and s < min_subs and explicit is None:
            return 0.0                          # 訂閱不足且未特別指定 → 排除
        return explicit if explicit is not None else ch_default

    channel_stat: dict[str, list] = {}          # title -> [videos, subs, weight]
    summary = []

    for topic, (dn, query) in queries.items():
        vids = topic_vids[topic]
        if not vids:
            print(f"{dn:<16} 查無影片（query='{query}'）"); continue
        views = fetch_view_counts([v["video_id"] for v in vids], key)

        rows = []
        for v in vids:
            vw = views.get(v["video_id"], 0)
            w = channel_weight(v["channel"], v["channel_id"])
            reach = w * math.log1p(vw)
            date = pd.to_datetime(v["published"]).tz_localize(None).normalize()
            rows.append({"date": date, "videos": 1, "views": vw, "reach": reach})
            st = channel_stat.setdefault(v["channel"], [0, subs.get(v["channel_id"], 0), w])
            st[0] += 1

        df = pd.DataFrame(rows).groupby("date").sum().sort_index()
        df.to_parquet(OUT_DIR / f"{topic}_youtube.parquet")
        kept = df[df["reach"] > 0] if "reach" in df else df
        summary.append({
            "題材": dn, "影片數": int(df["videos"].sum()),
            "計入觀看": int(df["views"].sum()), "加權觸及": round(float(df["reach"].sum()), 1),
        })

    print("\n各題材 YouTube 觀看熱度（已套用訂閱門檻+權重）：")
    sdf = pd.DataFrame(summary)
    pd.set_option("display.unicode.east_asian_width", True); pd.set_option("display.width", 200)
    print(sdf.to_string(index=False))

    print(f"\n出現的頻道（訂閱數 / 採用權重；標 ✗ 為被過濾）：")
    ranked = sorted(channel_stat.items(), key=lambda kv: -kv[1][0])
    for ch, (cnt, s, w) in ranked[:25]:
        mark = "✗排除" if w == 0 else f"權重{w}"
        ssub = f"{s:,}" if s >= 0 else "隱藏"
        print(f"  {cnt:3d}部  訂閱{ssub:>9}  {mark:<8} {ch}")
    print(f"\n✅ 已輸出至 {OUT_DIR}/")


if __name__ == "__main__":
    main()
