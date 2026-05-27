/**
 * 題材雷達 — Telegram Bot Webhook Handler
 * 部署於 Cloudflare Workers（免費方案）
 *
 * 環境變數（在 Cloudflare 後台設定為 Secrets）：
 *   TELEGRAM_BOT_TOKEN  Bot Token
 *   TELEGRAM_CHAT_ID    授權的 Chat ID
 *   GITHUB_PAT          GitHub Personal Access Token（Actions: write）
 *   GITHUB_REPO         例如 bluexinfu/news-stock-radar
 *
 * 支援指令：
 *   /run    — 立即觸發日報管線
 *   /status — 查詢管線執行狀態
 *   /help   — 顯示可用指令
 */

export default {
  async fetch(request, env) {
    // 只接受 POST（Telegram Webhook 使用 POST）
    if (request.method !== "POST") {
      return new Response("OK", { status: 200 });
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return new Response("Bad Request", { status: 400 });
    }

    const message = body.message;
    if (!message) return new Response("OK", { status: 200 });

    const chatId = String(message.chat?.id ?? "");
    const rawText = (message.text ?? "").trim();
    const cmd = rawText.split(/\s+/)[0].toLowerCase();

    // 安全性：只回應授權的 Chat
    if (chatId !== String(env.TELEGRAM_CHAT_ID)) {
      console.log(`Ignored message from unauthorized chat: ${chatId}`);
      return new Response("OK", { status: 200 });
    }

    // 指令路由
    if (cmd === "/run") {
      await handleRun(env, chatId);
    } else if (cmd === "/status") {
      await handleStatus(env, chatId);
    } else if (cmd === "/help" || cmd === "/start") {
      await handleHelp(env, chatId);
    } else {
      await tgSend(env, chatId,
        `❓ 不認識的指令：<code>${escapeHtml(rawText)}</code>\n` +
        `輸入 /help 查看可用指令。`
      );
    }

    return new Response("OK", { status: 200 });
  },
};

// ── 指令處理 ──────────────────────────────────────────────────────────

async function handleRun(env, chatId) {
  const ok = await ghPost(
    env,
    `/repos/${env.GITHUB_REPO}/actions/workflows/daily_pipeline.yml/dispatches`,
    { ref: "main" }
  );

  if (ok) {
    await tgSend(env, chatId,
      `🚀 <b>日報管線已啟動！</b>\n` +
      `⏳ 約 20 分鐘後會收到完整報告通知。\n\n` +
      `🔍 <a href="https://github.com/${env.GITHUB_REPO}/actions">查看執行進度</a>`
    );
  } else {
    await tgSend(env, chatId,
      `❌ <b>啟動失敗</b>\n` +
      `請至 <a href="https://github.com/${env.GITHUB_REPO}/actions">GitHub Actions</a> 手動觸發。`
    );
  }
}

async function handleStatus(env, chatId) {
  try {
    const data = await ghGet(
      env,
      `/repos/${env.GITHUB_REPO}/actions/workflows/daily_pipeline.yml/runs?per_page=1`
    );
    const run = data.workflow_runs?.[0];
    if (!run) {
      await tgSend(env, chatId, "ℹ️ 尚無執行紀錄。");
      return;
    }

    const created = run.created_at.slice(0, 16).replace("T", " ");
    let msg;
    if (run.status === "in_progress") {
      msg = `🔄 <b>管線執行中</b>\n⏱ 開始：${created} UTC\n🔍 <a href="${run.html_url}">查看進度</a>`;
    } else if (run.status === "queued") {
      msg = `⏳ <b>管線排隊中</b>（${created} UTC）`;
    } else if (run.conclusion === "success") {
      msg = `✅ <b>上次執行成功</b>\n📅 ${created} UTC\n🔍 <a href="${run.html_url}">查看結果</a>`;
    } else if (run.conclusion === "failure") {
      msg = `❌ <b>上次執行失敗</b>\n📅 ${created} UTC\n🔍 <a href="${run.html_url}">查看錯誤</a>`;
    } else {
      msg = `ℹ️ 狀態：${run.status}/${run.conclusion}（${created} UTC）`;
    }
    await tgSend(env, chatId, msg);
  } catch (e) {
    await tgSend(env, chatId, `⚠️ 無法查詢狀態：${e.message}`);
  }
}

async function handleHelp(env, chatId) {
  await tgSend(env, chatId,
    `📋 <b>可用指令</b>\n\n` +
    `/run — 立即執行日報（約 20 分鐘後收到報告）\n` +
    `/status — 查詢管線執行狀態\n` +
    `/help — 顯示此說明\n\n` +
    `⚠️ <i>僅限授權使用者操作</i>`
  );
}

// ── Telegram API ──────────────────────────────────────────────────────

async function tgSend(env, chatId, text) {
  const resp = await fetch(
    `https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chat_id: chatId,
        text,
        parse_mode: "HTML",
        disable_web_page_preview: true,
      }),
    }
  );
  if (!resp.ok) {
    console.error("tgSend failed:", await resp.text());
  }
}

// ── GitHub API ────────────────────────────────────────────────────────

const GH_BASE = "https://api.github.com";

function ghHeaders(env) {
  return {
    Authorization: `Bearer ${env.GITHUB_PAT}`,
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "Content-Type": "application/json",
    "User-Agent": "news-stock-radar-bot/1.0",
  };
}

async function ghGet(env, path) {
  const resp = await fetch(`${GH_BASE}${path}`, { headers: ghHeaders(env) });
  return resp.json();
}

async function ghPost(env, path, body) {
  const resp = await fetch(`${GH_BASE}${path}`, {
    method: "POST",
    headers: ghHeaders(env),
    body: JSON.stringify(body),
  });
  return resp.ok;
}

// ── 工具 ──────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
