let ws = null;
let alertCount = 0;
let reconnectTimer = null;
let pingTimer = null;
let historyTimer = null;
let logsTimer = null;
let renderedAlertIds = new Set();
let recognitionSession = null;

const RECONNECT_DELAY = 3000;
const CHAT_LISTEN_MS = 7000;

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
const speechState = document.getElementById('speechState');
const chatMicButton = document.getElementById('chatMic');
const chatInput = document.getElementById('chatInput');
const chatSendButton = document.getElementById('chatSend');

const startedAt = Date.now();

function getSpeechRecognitionCtor() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function updateClock() {
    if (currentTime) {
        const now = new Date();
        currentTime.textContent = now.toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
        });
    }

    if (uptime) {
        const uptimeMinutes = Math.max(0, Math.floor((Date.now() - startedAt) / 60000));
        uptime.textContent = `${uptimeMinutes}m`;
    }
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value ?? '';
    return div.innerHTML;
}

function getThreatLevel(score) {
    if (score >= 0.8) return 'critical';
    if (score >= 0.6) return 'high';
    if (score >= 0.3) return 'medium';
    return 'low';
}

function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
    }, RECONNECT_DELAY);
}

function ensurePollingFallback() {
    if (!historyTimer) {
        historyTimer = setInterval(fetchHistory, 2000);
    }
    if (!logsTimer) {
        logsTimer = setInterval(fetchLogs, 1500);
    }
    fetchHistory();
    fetchLogs();
}

function clearFallbackTimers() {
    if (historyTimer) {
        clearInterval(historyTimer);
        historyTimer = null;
    }
    if (logsTimer) {
        clearInterval(logsTimer);
        logsTimer = null;
    }
}

async function cleanupLegacyBrowserCache() {
    try {
        if ('serviceWorker' in navigator) {
            const regs = await navigator.serviceWorker.getRegistrations();
            for (const reg of regs) {
                await reg.unregister();
            }
        }
        if ('caches' in window) {
            const names = await caches.keys();
            await Promise.all(names.map((name) => caches.delete(name)));
        }
    } catch (err) {
        console.warn('Failed to clean old browser cache state');
    }
}

function setSpeechState(label, tone = 'idle') {
    if (speechState) {
        speechState.textContent = label;
        speechState.className = `speech-state ${tone}`;
    }

    if (chatMicButton) {
        chatMicButton.classList.toggle('listening', tone === 'listening');
        chatMicButton.disabled = tone === 'busy';
        chatMicButton.textContent = tone === 'listening' ? 'Stop' : 'Mic';
    }
}

function normalizeTranscript(text) {
    const cleaned = String(text || '').replace(/\s+/g, ' ').trim();
    if (!cleaned) return '';
    if (/[.!?]$/.test(cleaned)) {
        return cleaned;
    }
    return `${cleaned}.`;
}

function appendLocalLog(message, level = 'info') {
    appendLogEntry({
        level,
        message,
        timestamp: new Date().toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        }),
    });
}

function stopRecognitionSession() {
    if (!recognitionSession) return;
    try {
        recognitionSession.instance.stop();
    } catch (err) {
        console.warn('Failed to stop browser speech recognition');
    }
}

function recognizeSpeech({ mode, timeoutMs }) {
    return new Promise((resolve) => {
        const SpeechRecognition = getSpeechRecognitionCtor();
        if (!SpeechRecognition) {
            setSpeechState('Browser mic unsupported', 'error');
            resolve('');
            return;
        }

        if (recognitionSession) {
            resolve('');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.continuous = true;
        recognition.interimResults = true;

        let finalTranscript = '';
        let stopQueued = false;
        let finished = false;
        let timeoutHandle = null;

        const finish = (text = '', tone = 'ready') => {
            if (finished) return;
            finished = true;
            if (timeoutHandle) {
                clearTimeout(timeoutHandle);
            }
            recognitionSession = null;
            setSpeechState(
                tone === 'error'
                    ? 'Browser hearing unavailable'
                    : 'Browser hearing ready',
                tone,
            );
            resolve(normalizeTranscript(text));
        };

        const queueStop = () => {
            if (stopQueued) return;
            stopQueued = true;
            window.setTimeout(() => {
                try {
                    recognition.stop();
                } catch (err) {
                    finish(finalTranscript, 'ready');
                }
            }, 450);
        };

        recognition.onstart = () => {
            const label = mode === 'chat' ? 'Listening for chat...' : 'Listening for nearby reply...';
            setSpeechState(label, 'listening');
        };

        recognition.onresult = (event) => {
            for (let index = event.resultIndex; index < event.results.length; index += 1) {
                const segment = String(event.results[index][0].transcript || '').trim();
                if (event.results[index].isFinal && segment) {
                    finalTranscript = `${finalTranscript} ${segment}`.trim();
                }
            }

            if (finalTranscript) {
                queueStop();
            }
        };

        recognition.onerror = (event) => {
            if (event.error === 'aborted') {
                finish(finalTranscript, 'ready');
                return;
            }
            if (event.error === 'no-speech') {
                finish('', 'ready');
                return;
            }
            console.warn('Speech recognition error', event.error);
            finish('', 'error');
        };

        recognition.onend = () => {
            finish(finalTranscript, 'ready');
        };

        timeoutHandle = window.setTimeout(() => {
            try {
                recognition.stop();
            } catch (err) {
                finish(finalTranscript, 'ready');
            }
        }, timeoutMs);

        recognitionSession = { instance: recognition, mode };

        try {
            recognition.start();
        } catch (err) {
            console.warn('Speech recognition start failed', err);
            finish('', 'error');
        }
    });
}

async function submitLiveReply(requestId, transcript) {
    try {
        await fetch('/api/live_reply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                request_id: requestId,
                transcript,
            }),
        });
    } catch (err) {
        console.warn('Failed to submit live reply');
    }
}

async function handleListenRequest(data) {
    if (!data.request_id) return;

    if (recognitionSession) {
        appendLocalLog('Browser mic was busy, so one live reply request was skipped.', 'warning');
        await submitLiveReply(data.request_id, '');
        return;
    }

    const transcript = await recognizeSpeech({
        mode: 'live',
        timeoutMs: Math.max(1000, Number(data.timeout || 2) * 1000),
    });

    if (transcript) {
        appendLocalLog(`Browser heard: "${transcript}"`, 'success');
    }

    await submitLiveReply(data.request_id, transcript);
}

function connect() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws`;

    if (statusText) statusText.textContent = 'Connecting...';
    if (statusDot) statusDot.className = 'pulse-dot';

    ws = new WebSocket(url);

    ws.onopen = () => {
        if (statusDot) statusDot.className = 'pulse-dot connected';
        if (statusText) statusText.textContent = 'Connected';
        clearFallbackTimers();

        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }

        if (pingTimer) clearInterval(pingTimer);
        pingTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleSocketMessage(data);
    };

    ws.onerror = () => {
        if (statusText) statusText.textContent = 'Connection error';
        if (statusDot) statusDot.className = 'pulse-dot';
        ensurePollingFallback();
    };

    ws.onclose = () => {
        if (statusText) statusText.textContent = 'Disconnected';
        if (statusDot) statusDot.className = 'pulse-dot';
        if (pingTimer) {
            clearInterval(pingTimer);
            pingTimer = null;
        }
        ensurePollingFallback();
        scheduleReconnect();
    };
}

function handleSocketMessage(data) {
    if (data.type === 'connected') {
        if (clientCount) clientCount.textContent = `${data.alerts_total || 0} events`;
        return;
    }
    if (data.type === 'alert') {
        ingestAlert(data);
        return;
    }
    if (data.type === 'log') {
        appendLogEntry(data);
        return;
    }
    if (data.type === 'listen_request') {
        handleListenRequest(data);
    }
}

function appendLogEntry(entry) {
    const logs = document.getElementById('liveLogs');
    if (!logs) return;

    const row = document.createElement('div');
    row.className = `log-entry ${entry.level || 'info'}`;
    row.innerHTML = `<span class="log-time">[${entry.timestamp || '--:--:--'}]</span> ${escapeHtml(entry.message || '')}`;
    logs.appendChild(row);
    logs.scrollTop = logs.scrollHeight;

    while (logs.children.length > 100) {
        logs.removeChild(logs.firstChild);
    }
}

function ingestAlert(data) {
    const key = `${data.timestamp}-${data.threat_score}-${JSON.stringify(data.actions || [])}`;
    if (renderedAlertIds.has(key)) return;
    renderedAlertIds.add(key);

    alertCount += 1;
    if (alertCountEl) alertCountEl.textContent = String(alertCount);
    if (clientCount) clientCount.textContent = `${alertCount} events`;

    const score = data.threat_score || 0;
    const level = getThreatLevel(score);
    if (lastThreat) lastThreat.textContent = `${Math.round(score * 100)}%`;
    if (threatCard) threatCard.className = `stat-card threat-card ${level}`;

    if (statusDot) {
        statusDot.className = 'pulse-dot alert';
        setTimeout(() => {
            if (statusText && statusText.textContent === 'Connected') {
                statusDot.className = 'pulse-dot connected';
            }
        }, 1500);
    }

    if (emptyState) {
        emptyState.style.display = 'none';
    }

    if (alertFeed) {
        const card = createAlertCard(data, level);
        alertFeed.insertBefore(card, alertFeed.firstChild);

        while (alertFeed.children.length > 50) {
            alertFeed.removeChild(alertFeed.lastChild);
        }
    }
}

function createAlertCard(data, level) {
    const card = document.createElement('div');
    card.className = `alert-card threat-${level}`;

    const time = data.timestamp
        ? new Date(data.timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
        })
        : '--:--:--';

    const speakAction = (data.actions || []).find((action) => action.function === 'speak');
    const alertAction = (data.actions || []).find((action) => action.function === 'alert');
    const speakMsg = speakAction ? speakAction.params.message : '';
    const alertMsg = alertAction ? alertAction.params.message : '';

    let html = '';
    if (data.frame) {
        html += `<img class="alert-frame" src="data:image/jpeg;base64,${data.frame}" alt="Captured frame">`;
    }

    html += '<div class="alert-body">';
    html += `
        <div class="alert-top">
            <span class="threat-badge ${level}">${Math.round((data.threat_score || 0) * 100)}% ${level}</span>
            <span class="alert-time">${time}</span>
        </div>
    `;

    if (alertMsg) {
        html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:10px;">${escapeHtml(alertMsg)}</p>`;
    }

    if (speakMsg) {
        html += `
            <div class="speak-bubble">
                <span class="speak-label">Voice Output</span>
                <p class="speak-text">"${escapeHtml(speakMsg)}"</p>
            </div>
        `;
    }

    if (data.chain_of_thought) {
        const cotId = `cot-${Math.random().toString(36).slice(2)}`;
        html += `
            <div class="cot-section">
                <button class="cot-toggle" onclick="toggleCot('${cotId}')">Show reasoning</button>
                <div class="cot-content" id="${cotId}">${escapeHtml(data.chain_of_thought)}</div>
            </div>
        `;
    }

    html += '</div>';
    card.innerHTML = html;
    return card;
}

function toggleCot(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('expanded');
}

function clearAlerts() {
    if (alertFeed) {
        alertFeed.innerHTML = '';
    }
    alertCount = 0;
    renderedAlertIds = new Set();
    if (alertCountEl) alertCountEl.textContent = '0';
    if (lastThreat) lastThreat.textContent = '--';
    if (clientCount) clientCount.textContent = '0 events';
    if (threatCard) threatCard.className = 'stat-card threat-card';
    if (emptyState) {
        emptyState.style.display = '';
    }
}

async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/system_status');
        if (!res.ok) return;
        const status = await res.json();

        const sysStatusText = document.getElementById('sysStatusText');
        const sysYolo = document.getElementById('sysYolo');
        const sysBrain = document.getElementById('sysBrain');
        const sysTts = document.getElementById('sysTts');
        const sysGpu = document.getElementById('sysGpu');

        if (sysStatusText) sysStatusText.textContent = status.status || 'Unknown';
        if (sysYolo) sysYolo.textContent = (status.yolo_model || 'N/A').split('.').slice(0, -1).join('.') || status.yolo_model || 'N/A';
        if (sysBrain) sysBrain.textContent = (status.brain_model || 'N/A').split('/').pop();
        if (sysTts) sysTts.textContent = (status.tts_model || 'N/A').split('/').pop();
        if (sysGpu) sysGpu.textContent = (status.device || 'cpu').toUpperCase();
    } catch (err) {
        console.warn('Failed to fetch system status');
    }
}

async function fetchLogs() {
    try {
        const res = await fetch('/api/logs');
        if (!res.ok) return;
        const entries = await res.json();
        const logs = document.getElementById('liveLogs');
        if (!logs) return;

        logs.innerHTML = '';
        for (const entry of entries) {
            appendLogEntry(entry);
        }
    } catch (err) {
        console.warn('Failed to fetch logs');
    }
}

async function fetchHistory() {
    try {
        const res = await fetch('/api/history');
        if (!res.ok) return;
        const history = await res.json();
        for (const item of history) {
            ingestAlert(item);
        }
    } catch (err) {
        console.warn('Failed to fetch history');
    }
}

async function initCommandCenter() {
    const landing = document.getElementById('desktopLanding');
    if (!landing || window.innerWidth < 768) {
        if (landing) landing.style.display = 'none';
        return;
    }

    const feed = document.getElementById('liveFeed');
    if (feed) {
        feed.src = '/api/video_feed';
    }

    fetchSystemStatus();
    setInterval(fetchSystemStatus, 5000);
}

function initChat() {
    if (!chatInput) return;
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            sendChat();
        }
    });
}

function initSpeechControls() {
    if (!speechState) return;
    if (getSpeechRecognitionCtor()) {
        setSpeechState('Browser hearing ready', 'ready');
    } else {
        setSpeechState('Browser mic unsupported', 'error');
    }
}

async function toggleChatMic() {
    if (recognitionSession) {
        stopRecognitionSession();
        return;
    }

    const transcript = await recognizeSpeech({
        mode: 'chat',
        timeoutMs: CHAT_LISTEN_MS,
    });

    if (!transcript || !chatInput) {
        return;
    }

    chatInput.value = transcript;
    sendChat();
}

async function sendChat() {
    if (!chatInput) return;

    const text = chatInput.value.trim();
    if (!text) return;

    appendChatMessage('user', text);
    chatInput.value = '';
    appendChatTyping();
    if (chatSendButton) chatSendButton.disabled = true;
    if (chatMicButton) chatMicButton.disabled = true;

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });
        const data = await res.json();
        removeChatTyping();

        if (!res.ok) {
            appendChatMessage('ai', `Chat error: ${data.error || 'request failed'}`);
            return;
        }

        appendChatMessage('ai', data.message || 'No response.');
    } catch (err) {
        removeChatTyping();
        appendChatMessage('ai', 'Chat error: unable to reach Vanguard.');
    } finally {
        if (chatSendButton) chatSendButton.disabled = false;
        if (chatMicButton) chatMicButton.disabled = false;
    }
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
    const bubble = document.createElement('div');
    bubble.id = 'chatTyping';
    bubble.className = 'chat-bubble ai typing';
    bubble.textContent = 'Thinking...';
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function removeChatTyping() {
    const bubble = document.getElementById('chatTyping');
    if (bubble) bubble.remove();
}

window.toggleCot = toggleCot;
window.clearAlerts = clearAlerts;
window.sendChat = sendChat;
window.toggleChatMic = toggleChatMic;

document.addEventListener('DOMContentLoaded', () => {
    cleanupLegacyBrowserCache();
    updateClock();
    setInterval(updateClock, 1000);
    initCommandCenter();
    initChat();
    initSpeechControls();
    connect();
    ensurePollingFallback();
});
