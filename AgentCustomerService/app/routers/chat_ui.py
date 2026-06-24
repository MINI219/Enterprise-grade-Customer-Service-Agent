"""
聊天界面路由 —— 在 / 路径提供 Web 聊天界面
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>智能客服 — 小智</title>
<style>
  :root {
    --bg:           #f0f2f5;
    --header-bg:    #1a6fb5;
    --header-text:  #fff;
    --bubble-user:  #95ec69;
    --bubble-agent: #fff;
    --input-bg:     #fff;
    --border:       #e0e0e0;
    --text-primary: #1a1a1a;
    --text-secondary: #888;
    --shadow:       0 1px 3px rgba(0,0,0,.08);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial,
                 "PingFang SC", "Microsoft YaHei", sans-serif;
    background: var(--bg);
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  /* ── 聊天容器 ── */
  .chat-container {
    width: 100%;
    max-width: 800px;
    height: 95vh;
    background: #f7f7f7;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,.12);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── 顶部栏 ── */
  .chat-header {
    background: var(--header-bg);
    color: var(--header-text);
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-shrink: 0;
  }
  .chat-header .avatar {
    width: 42px; height: 42px;
    border-radius: 50%;
    background: rgba(255,255,255,.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
  }
  .chat-header .info h3 { font-size: 17px; font-weight: 600; }
  .chat-header .info span { font-size: 12px; opacity: .85; }

  /* ── 消息区域 ── */
  .chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 16px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: #ededed;
  }
  .chat-messages::-webkit-scrollbar { width: 6px; }
  .chat-messages::-webkit-scrollbar-thumb { background: #ccc; border-radius: 3px; }

  /* ── 气泡 ── */
  .msg-row { display: flex; gap: 8px; max-width: 85%; animation: fadeUp .3s ease; }
  .msg-row.user  { align-self: flex-end; flex-direction: row-reverse; }
  .msg-row.agent { align-self: flex-start; }

  .msg-row .avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .msg-row.agent .avatar { background: #d9e9fa; }
  .msg-row.user  .avatar { background: #b7e6a0; }

  .bubble {
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14.5px;
    line-height: 1.65;
    word-break: break-word;
    white-space: pre-wrap;
    position: relative;
  }
  .msg-row.agent .bubble {
    background: var(--bubble-agent);
    border-top-left-radius: 2px;
    box-shadow: var(--shadow);
  }
  .msg-row.user .bubble {
    background: var(--bubble-user);
    border-top-right-radius: 2px;
    box-shadow: var(--shadow);
  }

  .msg-row .time {
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }

  /* ── 加载动画 ── */
  .typing-indicator {
    align-self: flex-start;
    display: flex; gap: 6px; align-items: center;
    padding: 10px 14px;
    background: var(--bubble-agent);
    border-radius: 8px;
    box-shadow: var(--shadow);
  }
  .typing-indicator span {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #aaa;
    animation: bounce 1.2s infinite ease-in-out;
  }
  .typing-indicator span:nth-child(2) { animation-delay: .2s; }
  .typing-indicator span:nth-child(3) { animation-delay: .4s; }

  /* ── 输入区域 ── */
  .chat-input-area {
    background: var(--input-bg);
    border-top: 1px solid var(--border);
    padding: 12px 16px;
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-shrink: 0;
  }
  .chat-input-area textarea {
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    outline: none;
    max-height: 100px;
    line-height: 1.5;
    transition: border-color .2s;
  }
  .chat-input-area textarea:focus { border-color: var(--header-bg); }
  .chat-input-area button {
    width: 44px; height: 44px;
    border: none; border-radius: 8px;
    background: var(--header-bg);
    color: #fff;
    font-size: 20px;
    cursor: pointer;
    flex-shrink: 0;
    transition: background .2s, transform .1s;
    display: flex; align-items: center; justify-content: center;
  }
  .chat-input-area button:hover  { background: #155a8a; }
  .chat-input-area button:active { transform: scale(.95); }
  .chat-input-area button:disabled { opacity: .5; cursor: not-allowed; }

  /* ── 快捷按钮 ── */
  .quick-actions {
    display: flex; gap: 8px; flex-wrap: wrap; padding: 8px 16px 0;
  }
  .quick-actions button {
    border: 1px solid #ccc; border-radius: 16px;
    background: #fff; color: #555; font-size: 12px;
    padding: 6px 14px; cursor: pointer;
    transition: all .15s;
  }
  .quick-actions button:hover {
    background: var(--header-bg); color: #fff; border-color: var(--header-bg);
  }

  /* ── 动画 ── */
  @keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes bounce { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }

  /* ── 响应式 ── */
  @media (max-width: 600px) {
    .chat-container { height: 100vh; border-radius: 0; }
    .msg-row { max-width: 92%; }
  }
</style>
</head>
<body>

<div class="chat-container">
  <!-- 顶部栏 -->
  <div class="chat-header">
    <div class="avatar">🤖</div>
    <div class="info">
      <h3>小智 · 智能客服</h3>
      <span id="statusDot">●</span> <span id="statusText">在线</span>
    </div>
  </div>

  <!-- 消息列表 -->
  <div class="chat-messages" id="messages">
    <!-- 欢迎消息 -->
    <div class="msg-row agent">
      <div class="avatar">🤖</div>
      <div>
        <div class="bubble">👋 您好！我是智能客服 <b>小智</b>，很高兴为您服务。

您可以尝试问我：
• 「帮我查订单 ORD-20240623-001」
• 「快递 SF1234567890 到哪了？」
• 「我的投诉处理得怎么样了」

请问有什么可以帮您的？</div>
        <div class="time" style="padding-left:4px">刚刚</div>
      </div>
    </div>
  </div>

  <!-- 快捷操作 -->
  <div class="quick-actions" id="quickActions">
    <button onclick="sendQuick('帮我查一下订单 ORD-20240623-001 的状态')">📦 查订单</button>
    <button onclick="sendQuick('查一下快递 SF1234567890 的物流')">🚚 查物流</button>
    <button onclick="sendQuick('我的投诉处理得怎么样了？')">📋 查投诉</button>
  </div>

  <!-- 输入区域 -->
  <div class="chat-input-area">
    <textarea id="userInput" rows="1" placeholder="输入您的问题…" onkeydown="onKeyDown(event)"></textarea>
    <button id="sendBtn" onclick="sendMessage()" title="发送">➤</button>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════
// 状态管理
// ═══════════════════════════════════════════════════════
let conversationId = null;   // 当前会话 ID
let isWaiting = false;      // 是否正在等待回复

const messagesEl = document.getElementById('messages');
const inputEl    = document.getElementById('userInput');
const sendBtn    = document.getElementById('sendBtn');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

// 随机 user_id（模拟登录用户）
const USER_ID = 'U' + Math.floor(10000 + Math.random() * 90000);

// ═══════════════════════════════════════════════════════
// 工具函数
// ═══════════════════════════════════════════════════════
function now() {
  const d = new Date();
  return d.getHours().toString().padStart(2,'0') + ':' +
         d.getMinutes().toString().padStart(2,'0');
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setStatus(online, text) {
  statusDot.style.color = online ? '#4fde7e' : '#f5a623';
  statusText.textContent = text || (online ? '在线' : '处理中…');
}

// ═══════════════════════════════════════════════════════
// 渲染消息
// ═══════════════════════════════════════════════════════
function addMessage(role, text) {
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'agent' ? '🤖' : '👤';

  const body = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  const time = document.createElement('div');
  time.className = 'time';
  time.textContent = now();

  body.appendChild(bubble);
  body.appendChild(time);
  row.appendChild(avatar);
  row.appendChild(body);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function showTyping() {
  const div = document.createElement('div');
  div.className = 'typing-indicator';
  div.id = 'typing-el';
  div.innerHTML = '<span></span><span></span><span></span>';
  messagesEl.appendChild(div);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById('typing-el');
  if (el) el.remove();
}

// ═══════════════════════════════════════════════════════
// 发送消息
// ═══════════════════════════════════════════════════════
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isWaiting) return;

  isWaiting = true;
  sendBtn.disabled = true;
  inputEl.value = '';
  inputEl.style.height = 'auto';
  setStatus(false, '处理中…');

  // 显示用户消息
  addMessage('user', text);
  showTyping();

  try {
    const body = { user_id: USER_ID, message: text };
    if (conversationId) body.conversation_id = conversationId;

    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || '网络请求失败');
    }

    const data = await resp.json();
    conversationId = data.conversation_id;

    removeTyping();

    // 如果是兜底回复（code=500），给个视觉提示
    if (data.code === 500) {
      addMessage('agent', data.reply + '\n\n（⚠️ 当前为离线模式，请检查 API Key 配置）');
    } else {
      addMessage('agent', data.reply);
    }

    setStatus(true, '在线');

  } catch (err) {
    removeTyping();
    addMessage('agent', '😞 抱歉，消息发送失败：' + err.message + '\n\n请稍后重试或联系人工客服。');
    console.error('Chat error:', err);
    setStatus(true, '在线');
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// 快捷发送
function sendQuick(text) {
  inputEl.value = text;
  sendMessage();
}

// ═══════════════════════════════════════════════════════
// 键盘事件
// ═══════════════════════════════════════════════════════
function onKeyDown(e) {
  // Enter 发送（Shift+Enter 换行）
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

// 自动调整输入框高度
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + 'px';
});

// 初始聚焦
inputEl.focus();
scrollToBottom();
</script>
</body>
</html>"""


@router.get("/", response_class=HTMLResponse)
async def chat_ui():
    """返回 Web 聊天界面"""
    return HTMLResponse(content=CHAT_HTML)
