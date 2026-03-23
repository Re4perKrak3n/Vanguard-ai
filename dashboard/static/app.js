/**
 * Sentinel-Brain PWA — Client-Side Logic
 * WebSocket connection, alert rendering, notifications.
 */

// ── State ──────────────────────────────────────────────────────────
let ws = null;
let alertCount = 0;
let reconnectTimer = null;
let pingTimer = null;
const RECONNECT_DELAY = 3000;

// ── DOM refs ───────────────────────────────────────────────────────
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const clientCount = document.getElementById('clientCount');
const currentTime = document.getElementById('currentTime');
const alertCountEl = document.getElementById('alertCount');
const lastThreat = document.getElementById('lastThreat');
const uptime = document.getElementById('uptime');
const alertFeed = document.getElementById('alertFeed');
const emptyState = document.getElementById('emptyState');
const threatCard = document.querySelector('.threat-card');

// ── Clock ──────────────────────────────────────────────────────────
function updateClock() {
    const now = new Date();
    currentTime.textContent = now.toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit'
    });
}
setInterval(updateClock, 1000);
updateClock();

// ── WebSocket ──────────────────────────────────────────────────────
function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;

    statusText.textContent = 'Connecting...';
    statusDot.className = 'pulse-dot';

    ws = new WebSocket(url);

    ws.onopen = () => {
        statusDot.className = 'pulse-dot connected';
        statusText.textContent = 'Connected';
        if (pingTimer) {
            clearInterval(pingTimer);
            pingTimer = null;
        }
        pingTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    ws.onclose = () => {
        statusDot.className = 'pulse-dot';
        statusText.textContent = 'Disconnected';
        if (pingTimer) {
            clearInterval(pingTimer);
            pingTimer = null;
        }
        scheduleReconnect();
    };

    ws.onerror = () => {
        statusDot.className = 'pulse-dot';
        statusText.textContent = 'Connection error';
    };

}

function scheduleReconnect() {
    if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connect();
        }, RECONNECT_DELAY);
    }
}

function handleMessage(data) {
    switch (data.type) {
        case 'connected':
            clientCount.textContent = `${data.alerts_total || 0} logged`;
            break;

        case 'alert':
            handleAlert(data);
            break;

        case 'log':
            handleLog(data);
            break;

        case 'pong':
            break;

        default:
            console.log('Unknown message type:', data.type);
    }
}

function handleLog(data) {
    const logs = document.getElementById('liveLogs');
    const entry = document.createElement('div');
    entry.className = `log-entry ${data.level || 'info'}`;
    entry.innerHTML = `<span class="log-time">[${data.timestamp}]</span> ${escapeHtml(data.message)}`;

    logs.appendChild(entry);

    // Auto-scroll
    logs.scrollTop = logs.scrollHeight;

    // Keep max 100 logs
    while (logs.children.length > 100) {
        logs.removeChild(logs.firstChild);
    }
}

// ── Alert Handler ──────────────────────────────────────────────────
function handleAlert(data) {
    alertCount++;
    alertCountEl.textContent = alertCount;

    // Update threat display
    const score = data.threat_score || 0;
    const level = getThreatLevel(score);
    lastThreat.textContent = `${Math.round(score * 100)}%`;

    // Update threat card color
    threatCard.className = `stat-card threat-card ${level}`;

    // Flash the status dot
    statusDot.className = 'pulse-dot alert';
    setTimeout(() => {
        statusDot.className = 'pulse-dot connected';
    }, 3000);

    // Hide empty state
    emptyState.style.display = 'none';

    // Render alert card
    const card = createAlertCard(data, level);
    alertFeed.insertBefore(card, alertFeed.firstChild);

    // Limit displayed alerts
    while (alertFeed.children.length > 50) {
        alertFeed.removeChild(alertFeed.lastChild);
    }

    // Browser notification
    sendNotification(data, level);

    // Vibrate on high+ threats
    if (score > 0.6 && navigator.vibrate) {
        navigator.vibrate([200, 100, 200]);
    }
}

// ── Create Alert Card ──────────────────────────────────────────────
function createAlertCard(data, level) {
    const card = document.createElement('div');
    card.className = `alert-card threat-${level}`;

    const time = new Date(data.timestamp).toLocaleTimeString([], {
        hour: '2-digit', minute: '2-digit', second: '2-digit'
    });

    // Find speak action
    const speakAction = (data.actions || []).find(a => a.function === 'speak');
    const speakMsg = speakAction ? speakAction.params.message : null;

    // Find alert action
    const alertAction = (data.actions || []).find(a => a.function === 'alert');
    const alertMsg = alertAction ? alertAction.params.message : '';
    const priority = alertAction ? alertAction.params.priority : level;

    let html = '';

    // Frame image
    if (data.frame) {
        html += `<img class="alert-frame" src="data:image/jpeg;base64,${data.frame}" alt="Captured frame">`;
    }

    html += `<div class="alert-body">`;

    // Top bar
    html += `
        <div class="alert-top">
            <span class="threat-badge ${level}">${Math.round(data.threat_score * 100)}% ${level}</span>
            <span class="alert-time">${time}</span>
        </div>
    `;

    // Alert message
    if (alertMsg) {
        html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:10px;">${escapeHtml(alertMsg)}</p>`;
    }

    // Speak bubble
    if (speakMsg) {
        html += `
            <div class="speak-bubble">
                <span class="speak-label">🔊 Sentinel Said</span>
                <p class="speak-text">"${escapeHtml(speakMsg)}"</p>
            </div>
        `;
    }

    // Chain of thought (collapsible)
    if (data.chain_of_thought) {
        const cotId = `cot-${Date.now()}`;
        html += `
            <div class="cot-section">
                <button class="cot-toggle" onclick="toggleCot('${cotId}')">
                    ▸ Chain of Thought
                </button>
                <div class="cot-content" id="${cotId}">
                    ${escapeHtml(data.chain_of_thought)}
                </div>
            </div>
        `;
    }

    html += `</div>`;
    card.innerHTML = html;
    return card;
}

// ── Helpers ────────────────────────────────────────────────────────
function getThreatLevel(score) {
    if (score >= 0.8) return 'critical';
    if (score >= 0.6) return 'high';
    if (score >= 0.3) return 'medium';
    return 'low';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function toggleCot(id) {
    const el = document.getElementById(id);
    if (el) {
        el.classList.toggle('expanded');
        const btn = el.previousElementSibling;
        if (btn) {
            btn.textContent = el.classList.contains('expanded')
                ? '▾ Chain of Thought'
                : '▸ Chain of Thought';
        }
    }
}

function clearAlerts() {
    alertFeed.innerHTML = '';
    alertCount = 0;
    alertCountEl.textContent = '0';
    lastThreat.textContent = '--';
    threatCard.className = 'stat-card threat-card';
    emptyState.style.display = '';
}

// ── Notifications ──────────────────────────────────────────────────
async function initNotifications() {
    if ('Notification' in window && Notification.permission === 'default') {
        await Notification.requestPermission();
    }
}

function sendNotification(data, level) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    if (data.threat_score < 0.3) return; // Don't notify for low threats

    const alertAction = (data.actions || []).find(a => a.function === 'alert');
    const body = alertAction
        ? alertAction.params.message
        : `Threat: ${Math.round(data.threat_score * 100)}%`;

    const n = new Notification('🚨 Sentinel Alert', {
        body: body,
        icon: '/static/icon-192.png',
        badge: '/static/icon-192.png',
        tag: 'sentinel-alert',
        renotify: true,
        vibrate: [200, 100, 200],
    });

    n.onclick = () => {
        window.focus();
        n.close();
    };
}

// ── Service Worker ─────────────────────────────────────────────────
async function registerSW() {
    if ('serviceWorker' in navigator) {
        try {
            const reg = await navigator.serviceWorker.register('/sw.js');
            console.log('Service Worker registered:', reg.scope);
        } catch (err) {
            console.warn('SW registration failed:', err);
        }
    }
}

// ── Desktop Command Center ─────────────────────────────────────────
let statusTimer = null;

async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/system_status');
        const status = await res.json();

        document.getElementById('sysStatusText').textContent = status.status;
        document.getElementById('sysYolo').textContent = status.yolo_model.split('.')[0] || 'N/A';
        document.getElementById('sysBrain').textContent = status.brain_model.split('/')[1] || status.brain_model || 'N/A';
        document.getElementById('sysTts').textContent = status.tts_model.split('/')[1] || status.tts_model || 'N/A';
        document.getElementById('sysGpu').textContent = status.device.toUpperCase();

    } catch (err) {
        console.warn('System offline');
    }
}

async function initCommandCenter() {
    const landing = document.getElementById('desktopLanding');
    if (!landing || window.innerWidth < 768 || sessionStorage.getItem('hideLanding')) {
        if (landing) landing.style.display = 'none';
        return;
    }

    // 1. Connect Live Feed Action
    const feed = document.getElementById('liveFeed');
    feed.src = '/api/video_feed';

    // 2. Fetch Initial System Status & Start polling
    fetchSystemStatus();
    statusTimer = setInterval(fetchSystemStatus, 5000);

    // 3. Generate Mobile Dashboard QR Code
    try {
        const res = await fetch('/api/network');
        const data = await res.json();
        const port = window.location.port || '80';
        const url = `${window.location.protocol}//${data.ip}:${port}`;

        new QRCode(document.getElementById('qrcode'), {
            text: url,
            width: 120,
            height: 120,
            colorDark: "#000000",
            colorLight: "#ffffff",
            correctLevel: QRCode.CorrectLevel.L
        });
    } catch (err) {
        console.error('Failed to get network info', err);
    }
}

function hideLanding() {
    const landing = document.getElementById('desktopLanding');
    if (landing) {
        landing.style.display = 'none';
        sessionStorage.setItem('hideLanding', 'true');
        if (statusTimer) clearInterval(statusTimer);
    }
}

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initCommandCenter();
    registerSW();
    initNotifications();
    connect();
    initChat();
});

// ── Chat Console ───────────────────────────────────────────────────
let chatWs = null;

function initChat() {
    const input = document.getElementById('chatInput');
    if (!input) return;

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') sendChat();
    });

    connectChat();
}

function connectChat() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/chat`;

    chatWs = new WebSocket(url);

    chatWs.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'chat_response') {
            removeChatTyping();
            appendChatMessage('ai', data.message);
        }
    };

    chatWs.onclose = () => {
        setTimeout(connectChat, 3000);
    };
}

function sendChat() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text || !chatWs || chatWs.readyState !== WebSocket.OPEN) return;

    appendChatMessage('user', text);
    chatWs.send(JSON.stringify({ message: text }));
    input.value = '';
    appendChatTyping();
}

function appendChatMessage(role, text) {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function appendChatTyping() {
    const container = document.getElementById('chatMessages');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'chat-bubble ai typing';
    el.id = 'chatTyping';
    el.textContent = 'Thinking...';
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
}

function removeChatTyping() {
    const el = document.getElementById('chatTyping');
    if (el) el.remove();
}
