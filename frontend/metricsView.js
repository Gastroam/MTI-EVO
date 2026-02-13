/**
 * MTI-EVO Advanced Monitor - Metrics View
 * Updates metric cards and timeline charts.
 */
export function createMetricsView({ log }) {
    let history = [];

    function update(metrics) {
        if (!metrics) return;

        const driftEl = document.getElementById('metricDrift');
        const actP95El = document.getElementById('metricActP95');
        const latencyEl = document.getElementById('metricLatency');
        const seedsEl = document.getElementById('metricSeeds');

        if (driftEl) driftEl.textContent = metrics.driftAvg?.toFixed(2) ?? '—';
        if (actP95El) actP95El.textContent = metrics.activationP95?.toFixed(2) ?? '—';
        if (latencyEl) latencyEl.textContent = metrics.latencyMs ? `${metrics.latencyMs}ms` : '—';
        if (seedsEl) seedsEl.textContent = metrics.activeSeeds ?? '—';

        // Store history
        history.push({ ...metrics, timestamp: Date.now() });
        if (history.length > 100) history.shift();
    }

    function updateLatency(latencyMs) {
        const el = document.getElementById('metricLatency');
        if (el) el.textContent = `${latencyMs}ms`;
    }

    function updateSeeds(count) {
        const el = document.getElementById('metricSeeds');
        if (el) el.textContent = count ?? '—';
    }

    function updateUptime(seconds) {
        // Calculate formatted uptime
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        // Could update an uptime display if present
    }

    function getHistory() {
        return [...history];
    }

    return { update, updateLatency, updateSeeds, updateUptime, getHistory };
}
