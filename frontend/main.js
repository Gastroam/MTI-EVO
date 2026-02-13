/**
 * MTI Brain 3D Monitor - Main Bootstrap
 * Wires all modules together and starts the visualization.
 */
import { createTransport } from './transport.js';
import { createScene } from './scene.js';
import { createGraphView } from './graphView.js';
import { createMonitor } from './monitor.js';
import { createController } from './controller.js';

const canvas = document.getElementById('three-canvas');
const logsEl = document.getElementById('logs');

// Global state
let startTime = Date.now();
let requestCount = 0;
let lastDreamTime = null;

// Logging utility
const log = (msg) => {
    const t = new Date().toLocaleTimeString();
    if (logsEl) {
        logsEl.textContent = `[${t}] ${msg}\n` + logsEl.textContent;
        const lines = logsEl.textContent.split('\n').slice(0, 50);
        logsEl.textContent = lines.join('\n');
    }
    console.log(`[${t}] ${msg}`);
};

// Initialize modules
const transport = createTransport({ log });
const scene = createScene({ canvas });
const graphView = createGraphView({ scene });
const monitor = createMonitor({ graphView, log });
const controller = createController({ transport, monitor, log });

// ============== TAB NAVIGATION ==============
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            // Deactivate all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Activate selected
            btn.classList.add('active');
            document.getElementById(`tab-${tabId}`)?.classList.add('active');
        });
    });
}

// ============== STATUS UPDATE ==============
function updateStatus(payload) {
    requestCount++;

    const statusEl = document.getElementById('systemStatus');
    if (statusEl) {
        statusEl.textContent = payload.status === 'online' ? 'ONLINE' : 'OFFLINE';
        statusEl.className = `status-badge ${payload.status}`;
    }

    const neuronEl = document.getElementById('neuronCount');
    if (neuronEl) neuronEl.textContent = payload.neurons ?? '—';

    const versionEl = document.getElementById('version');
    if (versionEl) versionEl.textContent = payload.version ?? '—';

    const modeEl = document.getElementById('brainMode');
    if (modeEl) modeEl.textContent = payload.mode ?? '—';

    // Telemetry tab
    const telNeuronsEl = document.getElementById('telNeurons');
    if (telNeuronsEl) telNeuronsEl.textContent = payload.neurons ?? '—';
}

// ============== TELEMETRY POLLING ==============
let telemetryInterval = null;

async function pollTelemetry() {
    const autoRefresh = document.getElementById('autoRefresh');
    if (autoRefresh && !autoRefresh.checked) return;

    try {
        const start = performance.now();
        const res = await fetch('http://localhost:8800/status');
        const latency = Math.round(performance.now() - start);

        if (res.ok) {
            const data = await res.json();
            updateStatus(data);

            // Update telemetry displays
            const telLatency = document.getElementById('telLatency');
            if (telLatency) telLatency.textContent = `${latency}ms`;

            const telUptime = document.getElementById('telUptime');
            if (telUptime) {
                const uptime = Math.round((Date.now() - startTime) / 1000);
                const mins = Math.floor(uptime / 60);
                const secs = uptime % 60;
                telUptime.textContent = `${mins}m ${secs}s`;
            }

            const telLastDream = document.getElementById('telLastDream');
            if (telLastDream && lastDreamTime) {
                const ago = Math.round((Date.now() - lastDreamTime) / 1000);
                telLastDream.textContent = `${ago}s ago`;
            }

            const reqPerMin = document.getElementById('reqPerMin');
            if (reqPerMin) {
                const mins = (Date.now() - startTime) / 60000;
                reqPerMin.textContent = Math.round(requestCount / Math.max(mins, 1));
            }
        }
    } catch (e) {
        log(`Telemetry error: ${e.message}`);
    }
}

function startTelemetry() {
    pollTelemetry();
    telemetryInterval = setInterval(pollTelemetry, 3000);
}

// ============== GRAPH LOADING ==============
async function loadGraph() {
    log('Loading graph topology...');
    try {
        const res = await fetch('http://localhost:8800/api/graph');
        if (res.ok) {
            const data = await res.json();

            if (data.nodes && data.nodes.length > 0) {
                graphView.loadGraph(data.nodes, data.edges || []);

                // Update overlay
                const nodeCountEl = document.getElementById('nodeCount');
                if (nodeCountEl) nodeCountEl.textContent = `Nodes: ${data.nodes.length}`;

                const edgeCountEl = document.getElementById('edgeCount');
                if (edgeCountEl) edgeCountEl.textContent = `Edges: ${data.edges?.length || 0}`;

                log(`Graph loaded: ${data.nodes.length} nodes, ${data.edges?.length || 0} edges`);
            } else {
                log('Graph empty or error');
            }
        } else {
            log(`Graph fetch failed: ${res.status}`);
        }
    } catch (e) {
        log(`Graph error: ${e.message}`);
    }
}

// ============== TEST BINDINGS ==============
function bindTests() {
    const testLog = document.getElementById('testLog');
    const testResults = document.getElementById('testResults');

    const tlog = (msg) => {
        const t = new Date().toLocaleTimeString();
        if (testLog) testLog.textContent = `[${t}] ${msg}\n` + testLog.textContent;
    };

    // Test Connection
    document.getElementById('testConnectionBtn')?.addEventListener('click', async () => {
        tlog('Testing connection...');
        try {
            const res = await fetch('http://localhost:8800/status');
            if (res.ok) {
                const data = await res.json();
                tlog(`✅ Connection OK: ${data.neurons} neurons, ${data.mode}`);
                if (testResults) testResults.textContent = JSON.stringify(data, null, 2);
            } else {
                tlog(`❌ Connection failed: ${res.status}`);
            }
        } catch (e) {
            tlog(`❌ Connection error: ${e.message}`);
        }
    });

    // Load Graph
    document.getElementById('testGraphBtn')?.addEventListener('click', async () => {
        tlog('Loading graph...');
        await loadGraph();
        tlog('Graph load complete');
    });

    // Quick Dream
    document.getElementById('testDreamBtn')?.addEventListener('click', async () => {
        tlog('Running quick dream (sun)...');
        const result = await transport.dream('sun', 5);
        if (result) {
            tlog(`✅ Dream: ${result.path?.join(' → ')}`);
            if (testResults) testResults.textContent = JSON.stringify(result, null, 2);
            lastDreamTime = Date.now();
        } else {
            tlog('❌ Dream failed');
        }
    });

    // Hebbian Recall
    document.getElementById('testHebbianBtn')?.addEventListener('click', async () => {
        tlog('Testing Hebbian recall (prime)...');
        const result = await transport.dream('prime', 3);
        if (result) {
            tlog(`✅ Recall: ${result.path?.join(' → ')}`);
        } else {
            tlog('❌ Recall failed');
        }
    });

    // Stress Test
    document.getElementById('runStressBtn')?.addEventListener('click', async () => {
        const iterations = parseInt(document.getElementById('stressIterations')?.value) || 5;
        const stressResults = document.getElementById('stressResults');

        tlog(`🔥 Running ${iterations} stress iterations...`);
        const times = [];

        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            await transport.dream(`stress_${i}`, 3);
            const elapsed = performance.now() - start;
            times.push(elapsed);
            tlog(`  Iteration ${i + 1}: ${elapsed.toFixed(0)}ms`);
        }

        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const max = Math.max(...times);
        const min = Math.min(...times);

        tlog(`✅ Stress complete: avg=${avg.toFixed(0)}ms, min=${min.toFixed(0)}ms, max=${max.toFixed(0)}ms`);
        if (stressResults) {
            stressResults.textContent = `Avg: ${avg.toFixed(0)}ms | Min: ${min.toFixed(0)}ms | Max: ${max.toFixed(0)}ms`;
        }
    });
}

// ============== EVENT HANDLERS ==============
transport.on('status', updateStatus);

transport.on('graph', (payload) => {
    graphView.loadGraph(payload.nodes, payload.edges);
    log(`Graph: ${payload.nodes.length} nodes, ${payload.edges.length} edges`);
});

transport.on('dream', (payload) => {
    const dreamPath = document.getElementById('dreamPath');
    if (dreamPath) dreamPath.textContent = payload.path?.join(' → ') ?? '';

    if (payload.path) {
        payload.path.forEach((id, i) => {
            const intensity = 1.0 - (i / payload.path.length) * 0.5;
            graphView.setActivation(id, intensity);
        });
    }

    lastDreamTime = Date.now();
    log(`Dream: ${payload.drift_length} steps`);
});

transport.on('interview', (payload) => {
    const el = document.getElementById('interviewResult');
    if (el) {
        el.innerHTML = `<b>Asociaciones:</b> ${payload.associations?.join(', ')}<br><b>Explicación:</b> ${payload.explanation}`;
    }
    log(`Interview: ${payload.target}`);
});

// ============== STARTUP ==============
async function init() {
    log('MTI Brain Monitor v2.0 starting...');

    // Initialize tabs
    initTabs();

    // Start Three.js render loop
    scene.start();

    // Connect transport (fetches initial status)
    await transport.connect();

    // Load initial graph
    await loadGraph();

    // Bind controls
    controller.bindUI();

    // Bind test buttons
    bindTests();

    // Start telemetry polling
    startTelemetry();

    log('Initialization complete');
}

init();
