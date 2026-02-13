/**
 * MTI-EVO Control Center - Main Application
 * Professional UI with tab routing and backend integration.
 */
import { createBrainView } from './brainView.js?v=2.8';
import { createLatentView } from './latentView.js?v=2.8';
import { PlaygroundView } from './playgroundView.js?v=2.8';
import { ProjectsView } from './projectsView.js';
// import { uiEffects } from './uiEffects.js';

const API_BASE = 'http://127.0.0.1:8800';

const state = {
    connected: false,
    startTime: Date.now(),
    neurons: 0,
    latency: 0
};

let t = 0;
// Global State (Top Level for Hoisting)
let brainView = null;
let latentView = null;
let projectsView = null;

// Helper function definitions if missing
function updateStatus(data) {
    const apiStatus = document.getElementById('apiStatus');
    const brainStatus = document.getElementById('brainStatus');
    const llmStatus = document.getElementById('llmStatus');

    if (apiStatus) apiStatus.className = 'indicator-dot online';
    if (brainStatus) brainStatus.className = 'indicator-dot ' + ((data.status === 'online' || data.brain_active) ? 'online' : '');
    // LLM status is often separate, but simplistically:
    if (llmStatus) llmStatus.className = 'indicator-dot online';

    document.getElementById('headerNeurons').textContent = data.neurons || 0;
    document.getElementById('infoNeurons').textContent = data.neurons || 0;
    document.getElementById('infoMode').textContent = data.mode || 'ACTIVE';
    document.getElementById('infoVersion').textContent = data.version || 'v2.2.0';
}

function updateLatency(ms) {
    state.latency = ms;
    document.getElementById('headerLatency').textContent = `${ms}ms`;
    document.getElementById('hudPing').textContent = `${ms}ms`;
    document.getElementById('metricLatency').textContent = `${ms}ms`;
}

function updateHUD({ drift, vram, fps, latency }) {
    // 1. Drift Gauge
    const needle = document.getElementById('driftNeedle');
    const driftVal = document.getElementById('driftValue');
    if (needle && driftVal) {
        const pos = 10 + (drift * 80);
        needle.style.left = `${pos}%`;
        driftVal.textContent = drift.toFixed(3);

        // Glitch effect on high drift
        /*
        if (drift > 0.8) {
             // uiEffects.glitchText(document.querySelector('.tech-header span'), 'WARNING');
        }
        */
    }

    // Threat Level (New Data Binding)
    const threatBars = document.querySelectorAll('.threat-bar');
    const threatLevel = Math.min(8, Math.floor(drift * 8)); // Map drift to threat
    threatBars.forEach((bar, i) => {
        if (i < threatLevel) {
            bar.classList.add('active');
            if (threatLevel > 6) bar.classList.add('critical');
        } else {
            bar.classList.remove('active', 'critical');
        }
    });

    // 2. Complex Reactor (Dual Rings)
    const outC = document.getElementById('vramCircleOuter');
    const inC = document.getElementById('vramCircleInner');
    const vramText = document.getElementById('vramValue');

    if (outC && inC && vramText) {
        // Outer 2*pi*45 = 283
        const offOut = 283 - (283 * vram / 100);
        outC.style.strokeDasharray = 283;
        outC.style.strokeDashoffset = offOut;

        // Inner 2*pi*35 = 220
        // Inner shows inverse logical capacity or just animation
        const offIn = 220 - (220 * (vram * 0.8) / 100);
        inC.style.strokeDasharray = 220;
        inC.style.strokeDashoffset = offIn;

        vramText.textContent = `${Math.round(vram)}%`;

        // Color shift
        const color = vram > 80 ? 'var(--accent-red)' : (vram > 40 ? 'var(--accent-yellow)' : 'var(--accent-cyan)');
        outC.style.stroke = color;
        inC.style.stroke = color;
    }

    // 3. System
    const fpsEl = document.getElementById('hudFps');
    if (fpsEl) fpsEl.textContent = fps;

    // 4. Flux Update (affects waveform)
    targetFlux = latency > 0 ? (1000 / latency) : 1;
    const fluxEl = document.getElementById('tensorValue');
    if (fluxEl) fluxEl.textContent = `${Math.round(targetFlux * 10)} t/s`;
}

// Waveform Animation Loop
let targetFlux = 1;
let wavePhase = 0;
function animateWaveform() {
    const cvs = document.getElementById('waveform-canvas');
    if (!cvs) return requestAnimationFrame(animateWaveform);

    const ctx = cvs.getContext('2d');
    const w = cvs.width;
    const h = cvs.height;

    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const amp = Math.min(25, 5 + targetFlux * 2); // Amplitude based on flux
    const freq = 0.05 + targetFlux * 0.01;

    for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin(x * freq + wavePhase) * amp * Math.sin(x * 0.02); // Modulate
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    wavePhase += 0.1 + targetFlux * 0.05;
    requestAnimationFrame(animateWaveform);
}
requestAnimationFrame(animateWaveform);

// Draggable System
document.querySelectorAll('.draggable').forEach(makeDraggable);

// Helper logging function
function log(msg) {
    console.log(`[MTI-EVO] ${msg}`);
    const consoleLogs = document.getElementById('consoleLogs');
    if (consoleLogs) {
        const entry = document.createElement('div');
        entry.className = 'log-entry system';
        const time = new Date().toLocaleTimeString();
        entry.innerHTML = `<span class="timestamp">${time}</span><span class="type">[SYSTEM]</span> ${msg}`;
        consoleLogs.appendChild(entry);
        consoleLogs.scrollTop = consoleLogs.scrollHeight;
    }
}

function makeDraggable(el) {
    let isDown = false;
    let startX, startY, initialLeft, initialTop;

    // Header drag handler
    const onMouseDown = (e) => {
        // [Logic] Only drag via Header if it exists
        const header = el.querySelector('.chat-header, .console-header, .tech-header, .panel-header, .threat-label');
        if (header && !header.contains(e.target)) return;

        // [Logic] Ignore inputs/buttons
        if (['INPUT', 'BUTTON', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) return;

        isDown = true;
        startX = e.clientX;
        startY = e.clientY;

        // Use offsetLeft/Top for relative positioning correctness
        initialLeft = el.offsetLeft;
        initialTop = el.offsetTop;

        // Bring to front
        el.style.zIndex = 1001;

        // Prevent selection
        e.preventDefault();
    };

    const onMouseUp = () => {
        isDown = false;
        // Optional: reduce z-index or keep it high
        if (el.style.zIndex === '1001') el.style.zIndex = '';
    };

    const onMouseMove = (e) => {
        if (isDown) {
            e.preventDefault();

            // Calculate delta
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;

            // Apply new position relative to parent
            el.style.left = (initialLeft + dx) + 'px';
            el.style.top = (initialTop + dy) + 'px';
            el.style.bottom = 'auto';
            el.style.right = 'auto';
            el.style.transform = 'none'; // Clear any transform
        }
    };

    el.addEventListener('mousedown', onMouseDown, true);
    document.addEventListener('mouseup', onMouseUp, true);
    document.addEventListener('mousemove', onMouseMove, true);
}


function updateUptime() {
    const secs = Math.floor((Date.now() - state.startTime) / 1000);
    const mins = Math.floor(secs / 60);
    const s = secs % 60;
    document.getElementById('infoUptime').textContent = `${mins}m ${s}s`;
}

// ============== TABS & ROUTING ==============
function initTabs() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanels = document.querySelectorAll('.tab-panel');

    // Default active tab
    // document.getElementById('tab-brain').classList.add('active');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Deactivate all
            navItems.forEach(n => n.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            // Activate current
            item.classList.add('active');
            const tabName = item.dataset.tab;
            const targetPanel = document.getElementById(`tab-${tabName}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            // Specific init logic per tab
            if (tabName === 'brain') {
                if (!brainView) initBrainView();
                // brainView?.start();
            } else if (tabName === 'latent') {
                // if (!latentView) renderLatent();
                // else latentView.start();
                renderLatent(); // Refresh auto
            } else if (tabName === 'activity') {
                generateHeatmap(); // [NEW] Auto-generate
            } else if (tabName === 'metrics') {
                updateMetrics();
            } else if (tabName === 'playground') {
                // Init Playground on demand
                if (!window.playground) {
                    // Init Playground
                    playground = new PlaygroundView();

                    // Start Polling Loops
                    startPoll();
                }
            } else if (tabName === 'settings') {
                // Init Settings on demand
                if (!settings) {
                    settings = new SettingsView();
                }
            } else if (tabName === 'hive') {
                // Hive logic if separate
            } else if (tabName === 'projects') {
                if (!projectsView) {
                    projectsView = new ProjectsView();
                    projectsView.init();
                }
            }
        });
    });
}

function bindListControls() {
    const btn = document.getElementById('btnShowList');
    const panel = document.getElementById('attractorListPanel');
    const close = document.getElementById('btnCloseList');

    if (btn && panel) {
        btn.addEventListener('click', () => {
            panel.style.display = 'flex';
        });
    }

    if (close && panel) {
        close.addEventListener('click', () => {
            panel.style.display = 'none';
        });
    }
}

// ============== BRAIN VIEW ==============
function initBrainView() {
    const canvas = document.getElementById('brain-canvas');
    if (!canvas) return;


    if (!brainView) {
        brainView = createBrainView({ canvas, log });
        brainView.start();
    }
}

async function loadGraph() {
    log('Loading brain topology...');
    try {
        const res = await fetch(`${API_BASE}/api/graph`);
        if (!res.ok) {
            log(`Graph error: ${res.status}`);
            return;
        }

        const data = await res.json();
        log(`Graph data: ${JSON.stringify(data).slice(0, 100)}...`);

        if (data.nodes && data.nodes.length > 0) {
            brainView?.loadGraph(data.nodes, data.edges || []);
            document.getElementById('nodeCount').textContent = `Nodes: ${data.nodes.length}`;
            document.getElementById('edgeCount').textContent = `Edges: ${data.edges?.length || 0}`;
            log(`Loaded ${data.nodes.length} nodes, ${data.edges?.length || 0} edges`);
        } else if (data.error) {
            log(`Graph error: ${data.error}`);
        } else {
            log('No nodes in graph');
        }
    } catch (e) {
        log(`Graph fetch failed: ${e.message}`);
    }
}

// ============== CONTROLS ==============
function bindControls() {
    bindListControls(); // [NEW] Bind List Panel

    // Dream button
    document.getElementById('btnDream')?.addEventListener('click', async () => {
        const seed = document.getElementById('brainSeed')?.value || 'sun';
        const steps = parseInt(document.getElementById('brainSteps')?.value) || 10;

        log(`Starting dream: ${seed}, ${steps} steps`);

        try {
            const res = await fetch(`${API_BASE}/control/dream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seed, steps })
            });

            if (res.ok) {
                const data = await res.json();
                document.getElementById('dreamResult').textContent = data.path?.join(' → ') || '—';
                log(`Dream complete: ${data.path?.length || 0} steps`);

                // Highlight path
                if (brainView && data.path) {
                    data.path.forEach((id, i) => {
                        setTimeout(() => brainView.setActivation(id, 1.0 - i * 0.1), i * 200);
                    });
                }
            }
        } catch (e) {
            log(`Dream failed: ${e.message}`);
        }
    });

    // Interview button
    document.getElementById('btnInterview')?.addEventListener('click', async () => {
        const target = document.getElementById('brainSeed')?.value || 'consciousness';

        log(`Starting interview: ${target}`);

        try {
            const res = await fetch(`${API_BASE}/control/interview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target })
            });

            if (res.ok) {
                const data = await res.json();
                document.getElementById('interviewResult').innerHTML =
                    `<strong>Associations:</strong> ${data.associations?.join(', ') || '—'}<br>` +
                    `<strong>Explanation:</strong> ${data.explanation || '—'}`;
                log(`Interview complete`);
            }
        } catch (e) {
            log(`Interview failed: ${e.message}`);
        }
    });

    // Refresh button
    document.getElementById('btnRefresh')?.addEventListener('click', loadGraph);

    // Heatmap generate
    document.getElementById('generateHeatmap')?.addEventListener('click', async () => {
        const bins = parseInt(document.getElementById('heatmapBins')?.value) || 20;
        await generateHeatmap(bins);
    });

    // Latent refresh [UPDATED]
    document.getElementById('btnDeepScan')?.addEventListener('click', async () => {
        await renderLatent();
    });

    // Probe button [UPDATED]
    document.getElementById('btnProbeLatent')?.addEventListener('click', () => {
        const seed = prompt("Enter Seed ID to Probe (e.g. 7245):", "7245");
        if (seed) window.probeNeuron(seed);
    });

    // Initial Metrics Load
    if (document.getElementById('tab-metrics')) {
        updateMetrics();
        setInterval(() => {
            if (document.getElementById('autoRefreshMetrics')?.checked &&
                document.getElementById('tab-metrics').classList.contains('active')) {
                updateMetrics();
            }
        }, 5000);
    }
}

// ============== METRICS VISUALIZATION ==============
async function updateMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        const history = data.history || [];

        if (history.length === 0) return;

        // Group by name
        const traces = {};
        history.forEach(pt => {
            if (!traces[pt.name]) traces[pt.name] = { x: [], y: [], type: 'scatter', mode: 'lines', name: pt.name };
            traces[pt.name].x.push(new Date(pt.timestamp * 1000));
            traces[pt.name].y.push(pt.value);
        });

        const plotData = Object.values(traces);

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8' },
            margin: { t: 20, r: 20, b: 40, l: 40 },
            xaxis: { showgrid: false, color: '#475569' },
            yaxis: { showgrid: true, gridcolor: '#334155', color: '#475569' },
            showlegend: true,
            legend: { x: 0, y: 1 }
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.newPlot('metricsChart', plotData, layout, config);

        // Update Act P95 if available
        if (traces['neurons']) {
            const vals = traces['neurons'].y;
            const last = vals[vals.length - 1];
            document.getElementById('metricSeeds').textContent = last;
        }
        if (traces['latency']) {
            const vals = traces['latency'].y;
            const last = vals[vals.length - 1];
            document.getElementById('metricLatency').textContent = `${Math.round(last)}ms`;
        }

    } catch (e) {
        log(`Metrics error: ${e.message}`);
    }
}

// [PHASE 50] Oneiric Archetypes
async function fetchArchetypes() {
    try {
        const panel = document.getElementById('archetypesPanel');
        if (!panel || panel.style.display === 'none') return; // Only fetch if visible (perf)

        const res = await fetch(`${API_BASE}/api/dreams/archetypes`);
        if (!res.ok) return;

        const data = await res.json();
        const archetypes = data.archetypes || [];

        if (archetypes.length === 0) {
            panel.innerHTML = '<div style="color:#64748b; font-style:italic;">No archetypes detected yet. System dreaming...</div>';
            return;
        }

        panel.innerHTML = archetypes.map(arch => {
            const anxietyPct = arch.avg_anxiety * 100;
            const vividPct = (arch.avg_vividness / 10) * 100;

            // Color code anxiety
            const anxietyColor = anxietyPct > 70 ? '#ef4444' : (anxietyPct > 40 ? '#f59e0b' : '#10b981');

            return `
            <div class="archetype-card">
                <div style="display:flex; justify-content:space-between; align-items:center; mb-2">
                    <strong style="color:var(--accent-cyan)">${arch.name}</strong>
                    <span style="font-size:10px; background:#334155; padding:2px 6px; border-radius:4px;">${arch.count} Dreams</span>
                </div>
                <div style="font-size:11px; margin: 8px 0; font-style:italic; color:#94a3b8; height:32px; overflow:hidden;">
                    "${arch.sample_text}"
                </div>
                
                <div class="stat-row">
                    <span>ANXIETY</span>
                    <div>
                        ${Math.round(anxietyPct)}%
                        <div class="meter-bg"><div class="meter-fill" style="width:${anxietyPct}%; background:${anxietyColor}"></div></div>
                    </div>
                </div>
                <div class="stat-row">
                    <span>VIVIDNESS</span>
                    <div>
                        ${arch.avg_vividness}
                        <div class="meter-bg"><div class="meter-fill" style="width:${vividPct}%; background:var(--accent-magenta)"></div></div>
                    </div>
                </div>
                
                <div style="margin-top:10px; font-size:10px; color:#64748b;">
                    MOODS: ${arch.dominant_moods.slice(0, 3).join(', ')}
                </div>
            </div>
            `;
        }).join('');

    } catch (e) {
        // Silent fail
    }
}

// ============== CHAT & FILES ==============
function bindChatControls() {
    const panel = document.getElementById('chat-panel');
    const toggle = document.getElementById('chatToggle');
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('chatSend');
    const messages = document.getElementById('chatMessages');
    const fileInput = document.getElementById('fileInput');
    const attachBtn = document.getElementById('attachBtn');
    const attachmentsDiv = document.getElementById('chatAttachments');

    let attachments = [];

    // Toggle minimize
    toggle?.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.toggle('minimized');
        toggle.textContent = panel.classList.contains('minimized') ? '□' : '—';
    });

    // Header click to maximize
    document.querySelector('.chat-header')?.addEventListener('click', () => {
        if (panel.classList.contains('minimized')) {
            panel.classList.remove('minimized');
            toggle.textContent = '—';
        }
    });

    // Attach file button
    attachBtn?.addEventListener('click', () => fileInput.click());

    // Handle file selection
    fileInput?.addEventListener('change', () => {
        if (fileInput.files) {
            handleFiles(Array.from(fileInput.files));
        }
    });

    // Drag and drop
    panel?.addEventListener('dragover', (e) => {
        e.preventDefault();
        panel.style.borderColor = '#3b82f6';
    });

    panel?.addEventListener('dragleave', (e) => {
        e.preventDefault();
        panel.style.borderColor = '';
    });

    panel?.addEventListener('drop', (e) => {
        e.preventDefault();
        panel.style.borderColor = '';
        if (e.dataTransfer.files) {
            handleFiles(Array.from(e.dataTransfer.files));
        }
    });

    function handleFiles(files) {
        if (!files || !files.length) return;

        files.forEach(file => {
            attachments.push(file);

            const div = document.createElement('div');
            div.className = 'attachment-item';

            let preview = '';
            if (file.type.startsWith('image/')) {
                const url = URL.createObjectURL(file);
                preview = `<img src="${url}">`;
            } else {
                preview = `<span>📄</span>`;
            }

            div.innerHTML = `
        ${preview}
        <span>${file.name}</span>
        <button class="attachment-remove">×</button>
      `;

            div.querySelector('.attachment-remove').onclick = () => {
                attachments = attachments.filter(f => f !== file);
                div.remove();
            };

            attachmentsDiv.appendChild(div);
        });

        // Auto-expand if minimized
        if (panel.classList.contains('minimized')) {
            panel.classList.remove('minimized');
            toggle.textContent = '—';
        }
    }

    // Send message
    async function sendMessage() {
        const text = input.value.trim();
        if (!text && attachments.length === 0) return;

        // Add user message
        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg user';
        let content = `<div>${text}</div>`;

        // Add attachment previews to message
        if (attachments.length > 0) {
            attachments.forEach(file => {
                if (file.type.startsWith('image/')) {
                    const url = URL.createObjectURL(file);
                    content += `<img src="${url}">`;
                } else if (file.type.startsWith('video/')) {
                    const url = URL.createObjectURL(file);
                    content += `<video src="${url}" controls></video>`;
                } else {
                    content += `<div class="file-attachment">📄 ${file.name}</div>`;
                }
            });
        }

        msgDiv.innerHTML = content;
        messages.appendChild(msgDiv);
        messages.scrollTop = messages.scrollHeight;

        // Clear input
        input.value = '';
        attachments = [];
        attachmentsDiv.innerHTML = '';
        fileInput.value = '';

        // Handle commands
        if (text.startsWith('/dream')) {
            input.disabled = true;
            try {
                const seed = text.split(' ')[1] || 'random';
                addSystemMessage(`Dreaming about "${seed}"...`);

                const res = await fetch(`${API_BASE}/control/dream`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ seed, steps: 10 })
                });

                const data = await res.json();
                addSystemMessage(`Dream complete. Path: ${data.path?.join(' → ')}`);

                // Visualize path if brain view is active
                if (brainView && data.path) {
                    data.path.forEach((id, i) => {
                        setTimeout(() => brainView.setActivation(id, 1.0 - i * 0.1), i * 200);
                    });
                }
            } catch (e) {
                addSystemMessage(`Error: ${e.message}`, 'error');
            }
            input.disabled = false;
            input.focus();
        }
        else if (text.startsWith('/interview')) {
            const target = text.split(' ').slice(1).join(' ') || 'self';
            addSystemMessage(`Interviewing "${target}"...`);
            // Simulate interview for now or call backend
            try {
                const res = await fetch(`${API_BASE}/control/interview`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target })
                });
                const data = await res.json();
                addBotMessage(`<b>Analysis:</b> ${data.explanation}<br><b>Associations:</b> ${data.associations?.join(', ')}`);
            } catch (e) {
                addSystemMessage(`Error: ${e.message}`, 'error');
            }
        }
        else {
            // General chat (Telepathy)
            input.disabled = true;
            try {
                const res = await fetch(`${API_BASE}/v1/local/reflex`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        action: 'telepathy',
                        prompt: text,
                        max_tokens: 1024
                    })
                });

                if (res.ok) {
                    const data = await res.json();
                    addBotMessage(data.response || 'No response from brain.');
                } else {
                    addSystemMessage(`Error: ${res.statusText}`, 'error');
                }
            } catch (e) {
                addSystemMessage(`Connection error: ${e.message}`, 'error');
            }
            input.disabled = false;
            input.focus();
        }
    }

    function addSystemMessage(text, type = 'info') {
        const div = document.createElement('div');
        div.className = 'msg system';
        div.innerHTML = text;
        if (type === 'error') div.style.color = '#ef4444';
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    function addBotMessage(html) {
        const div = document.createElement('div');
        div.className = 'msg response';
        div.innerHTML = html;
        messages.appendChild(div);
        messages.scrollTop = messages.scrollHeight;
    }

    sendBtn?.addEventListener('click', sendMessage);
    input?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
}

// ============== HEATMAP ==============
// ============== HEATMAP (3D) ==============
import { createHeatmapView } from './heatmapView.js?v=2.13';
import { SettingsView } from './settingsView.js';

// Global State (See top of file)
// let brainView = null; 
let hiveView = null;
// let latentView = null;
let playground = null;
let settings = null;
let apiStatus = 'offline';

let heatmapView = null;
let heatmapInterval = null;
let currentMatrix = null;
let currentNeurons = null;

async function generateHeatmap(bins = 20) {
    if (heatmapInterval) clearInterval(heatmapInterval);
    log('Generating 3D heatmap...');

    try {
        const res = await fetch(`${API_BASE}/api/graph`);
        if (!res.ok) return;

        const graph = await res.json();
        if (!graph.nodes?.length) return;

        currentNeurons = graph.nodes.slice(0, 25).map(n => n.id);
        const timeBins = Array.from({ length: bins }, (_, i) => i);

        // Initial Random Data
        currentMatrix = currentNeurons.map(() => timeBins.map(() => Math.random() * 0.5));

        // Init view if needed
        if (!heatmapView) {
            const canvas = document.getElementById('heatmap-canvas');
            if (canvas) {
                heatmapView = createHeatmapView({ canvas, log });
                heatmapView.start();
            }
        }

        if (heatmapView) {
            heatmapView.renderData({ neurons: currentNeurons, timeBins, matrix: currentMatrix });
            log(`Heatmap 3D rendered: ${currentNeurons.length} x ${bins} (Live)`);

            // Start Real-Time Loop
            heatmapInterval = setInterval(() => {
                if (!currentMatrix || !heatmapView) return;

                // Shift data left and add new random value at end (Simulate time scrolling)
                for (let i = 0; i < currentNeurons.length; i++) {
                    currentMatrix[i].shift();
                    // Biased random for "activity spikes"
                    const spike = Math.random() > 0.9 ? 1.0 : Math.random() * 0.3;
                    currentMatrix[i].push(spike);
                }
                heatmapView.updateData(currentMatrix);
            }, 100);
        }

    } catch (e) {
        log(`Heatmap error: ${e.message}`);
    }
}

// Legacy D3 render removed


// ============== LATENT SPACE ==============
async function renderLatent() {
    log('renderLatent: Starting sequence...');
    try {
        const start = document.getElementById('scanStart')?.value || '';
        const end = document.getElementById('scanEnd')?.value || '';
        const all = document.getElementById('scanAll')?.checked;

        let query = `?all=${all}`;
        if (start) query += `&start=${start}`;
        if (end) query += `&end=${end}`;

        log(`Fetching ${API_BASE}/api/attractors${query}...`);
        const res = await fetch(`${API_BASE}/api/attractors${query}`);
        log(`Fetch status: ${res.status}`);

        if (!res.ok) {
            log(`Fetch failed: ${res.statusText}`);
            return;
        }

        const data = await res.json();
        log(`Data received: ${JSON.stringify(data).slice(0, 100)}...`);

        if (!data.attractors || !data.attractors.length) {
            log('No attractors found in data payload');
            return;
        }

        // Init view if needed
        if (!latentView) {
            log('Initializing LatentView...');
            const canvas = document.getElementById('latent-canvas');
            if (canvas) {
                latentView = createLatentView({ canvas, log });
                latentView.start();
                log('LatentView started successfully');
            } else {
                log('CRITICAL: latent-canvas element NOT found in DOM');
                return;
            }
        }

        if (latentView) {
            log(`Passing ${data.attractors.length} attractors to view`);
            latentView.setAttractors(data.attractors);

            // [NEW] Update Attractor List Panel
            updateAttractorList(data.attractors);

            // Interaction Hook: When user clicks an attractor
            window.probeNeuron = async (seed) => {
                log(`Probing Attractor ${seed}...`);

                // [VISUAL] Seeking Animation
                const btn = document.querySelector(`button[onclick*="${seed}"]`);
                const originalText = btn ? btn.innerHTML : '';
                if (btn) btn.innerHTML = '<span class="spin">💠</span>';

                try {
                    const res = await fetch(`${API_BASE}/api/probe?seed=${seed}`);
                    const sonarData = await res.json();

                    if (sonarData.error) {
                        log(`Probe error from server: ${sonarData.error}`);
                        alert(`Probe Failed: ${sonarData.error}`);
                        if (btn) btn.innerHTML = originalText;
                        return;
                    }

                    // Restore button
                    if (btn) btn.innerHTML = originalText;

                    log(`Sonar scan complete for ${seed}`);
                    log(`Data: ${JSON.stringify(sonarData).slice(0, 100)}...`);

                    if (!sonarData.intensities) {
                        log('Error: No intensities in response');
                        return;
                    }

                    const modal = document.getElementById('probeInspector');
                    const title = document.getElementById('probeTitle');
                    const stats = document.getElementById('probeStats');
                    const svg = document.getElementById('probeSvg');

                    if (modal && title && stats && svg) {
                        modal.style.display = 'flex';
                        title.innerHTML = `PROBE: <span style="color:var(--accent-magenta)">${seed}</span>`;

                        // Metadata
                        const maxInt = Math.max(...sonarData.intensities);
                        const meanInt = sonarData.intensities.reduce((a, b) => a + b, 0) / sonarData.intensities.length;

                        stats.innerHTML = `
                            <div style="margin-bottom:8px"><strong>TAU (Threshold):</strong> <span style="color:var(--accent-cyan)">${sonarData.tau.toFixed(3)}</span></div>
                            <div style="margin-bottom:8px"><strong>PEAK INTENSITY:</strong> <span style="color:${maxInt > 0.8 ? '#ef4444' : 'var(--accent-magenta)'}">${maxInt.toFixed(3)}</span></div>
                            <div style="margin-bottom:8px"><strong>MEAN INTENSITY:</strong> ${meanInt.toFixed(3)}</div>
                            <div style="margin-bottom:8px"><strong>BASIN STABILITY:</strong> ${(meanInt / sonarData.tau).toFixed(2)}x</div>
                            <br>
                            <div style="color:var(--text-secondary)">Radial plot shows critical activation intensity per angle.</div>
                        `;

                        // D3 Polar Plot
                        const width = svg.clientWidth;
                        const height = svg.clientHeight;
                        const radius = Math.min(width, height) / 2 - 20;

                        // Clear previous
                        d3.select(svg).selectAll("*").remove();
                        const g = d3.select(svg).append("g")
                            .attr("transform", `translate(${width / 2},${height / 2})`);

                        // Scales
                        const r = d3.scaleLinear().domain([0, 1]).range([0, radius]);
                        const a = d3.scaleLinear().domain([0, sonarData.angles.length]).range([0, 2 * Math.PI]);

                        // Grid
                        const grids = [0.2, 0.4, 0.6, 0.8, 1.0];
                        grids.forEach(tick => {
                            g.append("circle")
                                .attr("r", r(tick))
                                .attr("fill", "none")
                                .attr("stroke", "#334155")
                                .attr("stroke-dasharray", "3,3");
                            g.append("text")
                                .attr("y", -r(tick))
                                .attr("dy", "-0.4em")
                                .attr("text-anchor", "middle")
                                .attr("fill", "#64748b")
                                .style("font-size", "9px")
                                .text(tick);
                        });

                        // Data Line
                        const line = d3.lineRadial()
                            .angle((d, i) => a(i))
                            .radius(d => r(d))
                            .curve(d3.curveLinearClosed);

                        // Path
                        g.append("path")
                            .datum(sonarData.intensities)
                            .attr("fill", "rgba(236, 72, 153, 0.2)")
                            .attr("stroke", "var(--accent-magenta)")
                            .attr("stroke-width", 2)
                            .attr("d", line);

                        // Points
                        g.selectAll(".point")
                            .data(sonarData.intensities)
                            .enter().append("circle")
                            .attr("cx", (d, i) => r(d) * Math.sin(a(i)))
                            .attr("cy", (d, i) => -r(d) * Math.cos(a(i)))
                            .attr("r", 2)
                            .attr("fill", "#fff");

                    }
                } catch (e) {
                    log(`Probe failed: ${e.message}`);
                    console.error(e);
                }
            }; // End of probeNeuron
        } else {
            log('CRITICAL: latentView instance is null');
        }

        log('Latent space render sequence complete');
    } catch (e) {
        log(`Latent error: ${e.message}`);
        console.error(e);
    }
} // End of renderLatent

// ============== LIST VIEW ==============
let currentAttractors = [];

function updateAttractorList(attractors) {
    currentAttractors = attractors;
    const tbody = document.getElementById('attractorTableBody');
    const totalEl = document.getElementById('attractorTotalCount');
    if (!tbody) return;

    tbody.innerHTML = '';
    totalEl.textContent = attractors.length;

    attractors.slice(0, 500).forEach(a => { // Limit render for perf
        const tr = document.createElement('tr');
        tr.style.borderBottom = '1px solid #1e293b';
        tr.style.cursor = 'pointer';
        tr.className = 'attractor-row';

        tr.innerHTML = `
            <td style="padding: 6px; color: var(--accent-cyan);">${a.seed}</td>
            <td style="padding: 6px; color: #94a3b8;">${a.family || 'Unknown'}</td>
            <td style="padding: 6px; color: var(--accent-magenta); font-weight: bold;">${a.mass ? a.mass.toFixed(2) : '-'}</td>
            <td style="padding: 6px;"><button class="small-btn" onclick="event.stopPropagation(); window.probeNeuron(${a.seed})">🔍</button></td>
        `;

        // Click row to probe
        tr.addEventListener('click', () => window.probeNeuron(a.seed));

        tbody.appendChild(tr);
    });
}




// ============== CONSOLE ==============
function logToConsole(type, message, timestamp) {
    const el = document.getElementById('consoleLogs');
    if (!el) return;

    const div = document.createElement('div');
    div.className = `log-entry ${type}`;

    const t = timestamp ? new Date(timestamp * 1000).toLocaleTimeString() : new Date().toLocaleTimeString();

    div.innerHTML = `
        <span class="timestamp">${t}</span>
        <span class="type">[${type.toUpperCase()}]</span>
        <span class="message">${message}</span>
    `;

    el.appendChild(div);
    el.scrollTop = el.scrollHeight;

    // Limit history
    while (el.children.length > 100) {
        el.firstChild.remove();
    }
}

function bindConsoleControls() {
    const panel = document.getElementById('console-panel');
    const toggle = document.getElementById('toggleConsole');
    const clear = document.getElementById('clearConsole');

    toggle?.addEventListener('click', () => {
        panel.classList.toggle('minimized');
    });

    clear?.addEventListener('click', () => {
        const logs = document.getElementById('consoleLogs');
        if (logs) logs.innerHTML = '';
    });
}

// ============== POLLING ==============
async function pollEvents() {
    try {
        const res = await fetch(`${API_BASE}/api/events`);
        if (res.ok) {
            const data = await res.json();
            if (data.events && data.events.length > 0) {
                const messages = document.getElementById('chatMessages');
                data.events.forEach(event => {
                    const msg = event.text || event.message;

                    // 1. Log to Console Panel (System events only, thoughts go to chat)
                    if (event.type !== 'thought') {
                        logToConsole(event.type, msg, event.timestamp);
                    }

                    // 2. Thoughts also go to Chat
                    if (event.type === 'thought') {
                        const div = document.createElement('div');
                        div.className = 'msg response thought';
                        div.innerHTML = `<i>💭 ${msg}</i>`;
                        const messages = document.getElementById('chatMessages');
                        messages?.appendChild(div);
                        if (messages) messages.scrollTop = messages.scrollHeight;

                        // Keep legacy log for now
                        log(`Gemma thought: ${msg}`);
                    }
                });
            }
        }
    } catch (e) {
        // Silent fail for event polling
    }
}

async function poll() {
    // Guard clause removed to allow reconnection logic to run
    // if (!state.connected && document.getElementById('apiStatus')) return;

    try {
        const start = performance.now();
        const res = await fetch(`${API_BASE}/status`);

        if (res.ok) {
            const data = await res.json();
            const latency = Math.round(performance.now() - start);

            state.connected = true;
            updateStatus(data);
            updateLatency(latency);

            // HUD Effects
            // Try to use Server VRAM/Drift, fallback to simulation
            const drift = data.drift !== undefined ? data.drift : (0.2 + Math.sin(t * 0.1) * 0.15);
            const vram = data.vram !== undefined ? data.vram : (30 + (data.neurons / 10));

            const fps = Math.round(1000 / (latency || 16));

            updateHUD({ drift, vram, fps, latency });
        }
    } catch {
        state.connected = false;
        document.getElementById('apiStatus').className = 'indicator-dot error'; // Red dot

        // [DATA INTEGRITY] Clear Offline Indication
        document.getElementById('headerNeurons').textContent = 'OFFLINE';
        document.getElementById('infoNeurons').textContent = '0'; // Or 'ERR'
        document.getElementById('metricSeeds').textContent = '—';
        document.getElementById('headerLatency').textContent = '—';
        document.getElementById('hudPing').textContent = 'NO_SIGNAL';

        // Reset HUD visualization to safe zero-state
        updateHUD({ drift: 0, vram: 0, fps: 0, latency: 0 });
    }

    updateUptime();
    await pollEvents(); // Poll for thoughts
}


function startPoll() {
    poll();
    setInterval(poll, 3000);
}




// ============== INIT ==============
// ============== INIT ==============
async function init() {
    log('MTI-EVO Control Center starting...');

    initTabs();
    initBrainView();
    bindControls();
    bindChatControls();

    // Initial data load
    await poll();
    await loadGraph();

    // Start polling
    startPoll();

    // [PHASE 50] Initial Archetype Fetch & Slow Poll
    setTimeout(fetchArchetypes, 1000);
    setInterval(fetchArchetypes, 15000);

    log('Initialization complete');
}

// Robust Initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
