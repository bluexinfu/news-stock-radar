#!/bin/bash
# 安裝每日自動管線排程（macOS launchd）
# 用法：bash scripts/install_schedule.sh

PLIST_SRC="$(cd "$(dirname "$0")" && pwd)/com.newsstock.dailypipeline.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.newsstock.dailypipeline.plist"
LABEL="com.newsstock.dailypipeline"

echo "📅 安裝每日管線排程…"
echo "   來源：$PLIST_SRC"
echo "   目標：$PLIST_DST"

# 若已安裝，先 unload
if launchctl list "$LABEL" &>/dev/null; then
    echo "   已有舊排程，先停用…"
    launchctl unload "$PLIST_DST" 2>/dev/null
fi

cp "$PLIST_SRC" "$PLIST_DST"
launchctl load "$PLIST_DST"

if launchctl list "$LABEL" &>/dev/null; then
    echo "✅ 排程安裝成功！每天 18:30 自動執行管線。"
    echo ""
    echo "   手動觸發測試：launchctl start $LABEL"
    echo "   查看 log：    tail -f logs/pipeline.log"
    echo "   停用排程：    bash scripts/uninstall_schedule.sh"
else
    echo "❌ 排程安裝失敗，請確認 plist 路徑正確。"
    exit 1
fi
