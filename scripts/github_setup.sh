#!/bin/bash
# ================================================================
# GitHub 一鍵上架腳本
# 執行前請先完成：gh auth login
# 用法：bash scripts/github_setup.sh
# ================================================================

set -e  # 任何錯誤就停止

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "=================================================="
echo "  📡 消息面×股價題材雷達 — GitHub 上架設定"
echo "=================================================="
echo ""

# ── 確認已登入 ──────────────────────────────────────────────────
if ! gh auth status &>/dev/null; then
  echo "❌ 尚未登入 GitHub，請先執行："
  echo "   gh auth login"
  exit 1
fi

GH_USER=$(gh api user --jq '.login')
echo "✅ 已登入為：$GH_USER"
echo ""

# ── 詢問 Repo 名稱 ───────────────────────────────────────────────
read -p "📝 Repository 名稱（預設：news-stock-radar）： " REPO_NAME
REPO_NAME="${REPO_NAME:-news-stock-radar}"

# ── Step 1：初始化 Git ───────────────────────────────────────────
echo ""
echo "【Step 1/6】初始化 Git repository…"
if [ ! -d ".git" ]; then
  git init
  git branch -M main
  echo "✅ Git 初始化完成"
else
  echo "ℹ️  .git 已存在，跳過"
fi

# ── Step 2：建立初始 Commit ──────────────────────────────────────
echo ""
echo "【Step 2/6】建立初始 commit…"
git add .
git commit -m "feat: 消息面×股價題材雷達 初始提交" --allow-empty 2>/dev/null || true
echo "✅ Commit 完成"

# ── Step 3：在 GitHub 建立 Repo 並推送 ──────────────────────────
echo ""
echo "【Step 3/6】在 GitHub 建立 Public Repository 並推送…"
gh repo create "$REPO_NAME" \
  --public \
  --description "消息面×股價題材雷達：NII × 籌碼面多主題監控系統" \
  --source . \
  --remote origin \
  --push
echo "✅ 程式碼已推送到 https://github.com/$GH_USER/$REPO_NAME"

# ── Step 4：建立 data branch（存放原始資料快取）────────────────
echo ""
echo "【Step 4/6】建立 data branch（原始資料快取）…"

# 建立暫存目錄
TMP_DATA=$(mktemp -d)
mkdir -p "$TMP_DATA/data/raw"

# 複製原始資料
if [ -d "data/raw" ]; then
  cp -r data/raw/. "$TMP_DATA/data/raw/"
  FILE_COUNT=$(find "$TMP_DATA/data/raw" -name "*.parquet" | wc -l | tr -d ' ')
  echo "  → 複製 $FILE_COUNT 個 parquet 檔案"
else
  echo "  ⚠️  找不到 data/raw/，建立空的 data branch"
fi

# 使用 worktree 建立 data branch
git worktree add -B data "$TMP_DATA/worktree" --orphan 2>/dev/null || \
  git worktree add "$TMP_DATA/worktree" --orphan -b data 2>/dev/null || true

if [ -d "$TMP_DATA/worktree" ]; then
  cp -r "$TMP_DATA/data" "$TMP_DATA/worktree/" 2>/dev/null || true
  cd "$TMP_DATA/worktree"
  git add data/ 2>/dev/null || true
  git commit -m "chore: 初始原始資料快取" --allow-empty 2>/dev/null || true
  git push origin data
  cd "$PROJECT_ROOT"
  git worktree remove "$TMP_DATA/worktree" --force 2>/dev/null || true
  echo "✅ data branch 建立完成"
else
  # Fallback：直接建立 orphan branch
  git checkout --orphan data
  mkdir -p data/raw
  [ -d "$PROJECT_ROOT/data/raw" ] && cp -r "$PROJECT_ROOT/data/raw/." data/raw/ 2>/dev/null || true
  git add data/ 2>/dev/null || true
  git commit -m "chore: 初始原始資料快取" --allow-empty
  git push origin data
  git checkout main
  echo "✅ data branch 建立完成"
fi
rm -rf "$TMP_DATA"

# ── Step 5：設定 GitHub Secrets ──────────────────────────────────
echo ""
echo "【Step 5/6】設定 GitHub Secrets（Telegram 憑證）…"

# 從 .env 讀取
if [ -f ".env" ]; then
  source <(grep -E "^TELEGRAM_" .env | sed 's/^/export /')
fi

if [ -n "$TELEGRAM_BOT_TOKEN" ] && [[ "$TELEGRAM_BOT_TOKEN" != your_* ]]; then
  gh secret set TELEGRAM_BOT_TOKEN --body "$TELEGRAM_BOT_TOKEN" --repo "$GH_USER/$REPO_NAME"
  echo "✅ TELEGRAM_BOT_TOKEN 已設定"
else
  echo "⚠️  未找到 TELEGRAM_BOT_TOKEN，請手動設定："
  echo "   gh secret set TELEGRAM_BOT_TOKEN --repo $GH_USER/$REPO_NAME"
fi

if [ -n "$TELEGRAM_CHAT_ID" ] && [[ "$TELEGRAM_CHAT_ID" != your_* ]]; then
  gh secret set TELEGRAM_CHAT_ID --body "$TELEGRAM_CHAT_ID" --repo "$GH_USER/$REPO_NAME"
  echo "✅ TELEGRAM_CHAT_ID 已設定"
else
  echo "⚠️  未找到 TELEGRAM_CHAT_ID，請手動設定："
  echo "   gh secret set TELEGRAM_CHAT_ID --repo $GH_USER/$REPO_NAME"
fi

# ── Step 6：啟用 GitHub Pages ────────────────────────────────────
echo ""
echo "【Step 6/6】啟用 GitHub Pages…"
gh api \
  --method POST \
  "/repos/$GH_USER/$REPO_NAME/pages" \
  --field source='{"branch":"gh-pages","path":"/"}' \
  2>/dev/null && echo "✅ GitHub Pages 已啟用" || \
  echo "ℹ️  GitHub Pages 需手動啟用（首次 gh-pages branch 推送後自動生效）"

# ── 完成 ─────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  🎉 設定完成！"
echo "=================================================="
echo ""
echo "  📂 Repository:   https://github.com/$GH_USER/$REPO_NAME"
echo "  🌐 網頁報告:      https://$GH_USER.github.io/$REPO_NAME/"
echo "  ⚙️  Actions:      https://github.com/$GH_USER/$REPO_NAME/actions"
echo ""
echo "  ⏰ 每天台灣時間 18:30 自動執行"
echo "  📱 執行後 Telegram 會收到通知＋報告連結"
echo ""
echo "  💡 手動觸發（測試用）："
echo "     gh workflow run daily_pipeline.yml --repo $GH_USER/$REPO_NAME"
echo ""
