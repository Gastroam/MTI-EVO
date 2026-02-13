/**
 * MTI-EVO Advanced Monitor - Transport Layer
 * WebSocket and REST communication with backend.
 */
const API_BASE = 'http://localhost:8800';
const WS_URL = 'ws://localhost:8800/ws';

export function createTransport({ log }) {
    let ws = null;
    let reconnectTimer = null;
    const listeners = new Map();

    /**
     * Register event listener
     */
    function on(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
    }

    /**
     * Emit event to listeners
     */
    function emit(type, payload) {
        const set = listeners.get(type);
        if (set) set.forEach(fn => fn(payload));
    }

    /**
     * Connect to backend
     */
    async function connect() {
        // Fetch initial status
        try {
            const res = await fetch(`${API_BASE}/status`);
            if (res.ok) {
                const data = await res.json();
                emit('status', data);
                log?.('Status fetched');
            }
        } catch (e) {
            log?.(`Status fetch failed: ${e.message}`);
        }

        // Fetch initial graph
        try {
            const res = await fetch(`${API_BASE}/api/graph`);
            if (res.ok) {
                const data = await res.json();
                emit('graph', data);
                log?.('Graph fetched');
            }
        } catch (e) {
            log?.(`Graph fetch failed: ${e.message}`);
        }

        // WebSocket connection
        try {
            ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                log?.('WebSocket connected');
                clearTimeout(reconnectTimer);
                emit('connected', true);
            };

            ws.onmessage = (ev) => {
                try {
                    const msg = JSON.parse(ev.data);
                    emit(msg.type, msg.payload);
                } catch (e) {
                    log?.(`WS parse error: ${e.message}`);
                }
            };

            ws.onclose = () => {
                log?.('WebSocket closed, reconnecting...');
                emit('connected', false);
                reconnectTimer = setTimeout(connect, 5000);
            };

            ws.onerror = () => {
                log?.('WebSocket error');
            };
        } catch (e) {
            log?.(`WebSocket failed: ${e.message}`);
        }
    }

    /**
     * Trigger dream via REST
     */
    async function dream(seed, steps = 10) {
        try {
            const res = await fetch(`${API_BASE}/control/dream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ seed, steps })
            });
            if (res.ok) {
                const data = await res.json();
                emit('dream', data);
                return data;
            }
        } catch (e) {
            log?.(`Dream failed: ${e.message}`);
        }
        return null;
    }

    /**
     * Trigger interview via REST
     */
    async function interview(target) {
        try {
            const res = await fetch(`${API_BASE}/control/interview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target })
            });
            if (res.ok) {
                const data = await res.json();
                emit('interview', data);
                return data;
            }
        } catch (e) {
            log?.(`Interview failed: ${e.message}`);
        }
        return null;
    }

    /**
     * Fetch graph topology
     */
    async function fetchGraph() {
        try {
            const res = await fetch(`${API_BASE}/api/graph`);
            if (res.ok) {
                const data = await res.json();
                emit('graph', data);
                return data;
            }
        } catch (e) {
            log?.(`Graph fetch failed: ${e.message}`);
        }
        return null;
    }

    /**
     * Fetch HIVE topology (stub - needs backend endpoint)
     */
    async function fetchHive() {
        // TODO: Implement when HIVE endpoint exists
        // For now, emit mock data
        emit('hive_graph', {
            nodes: [
                { id: 'master', role: 'master', region: 'caracas' },
                { id: 'worker-1', role: 'worker', region: 'miami' },
                { id: 'worker-2', role: 'worker', region: 'madrid' }
            ],
            edges: [
                { source: 'master', target: 'worker-1', type: 'control', weight: 0.9 },
                { source: 'master', target: 'worker-2', type: 'control', weight: 0.8 }
            ]
        });
    }

    return { on, emit, connect, dream, interview, fetchGraph, fetchHive };
}
