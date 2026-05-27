#!/usr/bin/env python3
"""
更新 config/topics.yaml 的時間範圍

在 GitHub Actions 每次執行前呼叫，把：
  end   → 今天（確保採集最新資料）
  start → 保持不變（固定起點，不縮短歷史）

用法：
    python scripts/update_dates.py
    python scripts/update_dates.py --end 2026-06-30   （指定 end）
    python scripts/update_dates.py --dry-run           （只印不寫）
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def update_topics_yaml(new_end: str, dry_run: bool = False) -> None:
    path = ROOT / "config" / "topics.yaml"
    text = path.read_text(encoding="utf-8")

    # 找出目前的 end 值並替換（只換 end: 那一行）
    import re
    pattern = r'(end:\s*")[^"]+(")'
    new_text = re.sub(pattern, rf'\g<1>{new_end}\g<2>', text)

    # 也處理無引號格式  end: 2026-05-26
    pattern2 = r'(end:\s*)(\d{4}-\d{2}-\d{2})'
    new_text = re.sub(pattern2, rf'\g<1>{new_end}', new_text)

    if new_text == text:
        print(f"topics.yaml end 日期已是最新（{new_end}），無需更新")
        return

    if dry_run:
        print(f"[dry-run] 將把 end 更新為 {new_end}")
        return

    path.write_text(new_text, encoding="utf-8")
    print(f"✅ topics.yaml 已更新：end → {new_end}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="更新 topics.yaml 時間範圍")
    parser.add_argument("--end", default=date.today().isoformat(),
                        help="結束日期 YYYY-MM-DD（預設：今天）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    update_topics_yaml(args.end, dry_run=args.dry_run)
