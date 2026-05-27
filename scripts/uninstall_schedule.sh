#!/bin/bash
# 停用每日自動管線排程
# 用法：bash scripts/uninstall_schedule.sh

PLIST_DST="$HOME/Library/LaunchAgents/com.newsstock.dailypipeline.plist"
LABEL="com.newsstock.dailypipeline"

echo "🛑 停用每日管線排程…"
launchctl unload "$PLIST_DST" 2>/dev/null && rm -f "$PLIST_DST"
echo "✅ 排程已停用。"
