/**
 * MTI Brain 3D Monitor - Monitor Module
 * Handles real-time activation updates and metrics.
 */
export function createMonitor({ graphView, log }) {
    let activationHistory = [];
    const MAX_HISTORY = 100;

    /**
     * Update node activations from a list
     */
    function updateActivations(list) {
        if (!Array.isArray(list)) return;

        list.forEach(({ id, value }) => {
            graphView.setActivation(id, value);
        });

        // Track history for p95 calculation
        const values = list.map(a => a.value);
        activationHistory.push(...values);
        if (activationHistory.length > MAX_HISTORY) {
            activationHistory = activationHistory.slice(-MAX_HISTORY);
        }
    }

    /**
     * Get current metrics
     */
    function getMetrics() {
        if (activationHistory.length === 0) return { p95: 0, mean: 0 };

        const sorted = [...activationHistory].sort((a, b) => a - b);
        const p95Index = Math.floor(sorted.length * 0.95);
        const p95 = sorted[p95Index] ?? 0;
        const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;

        return { p95, mean };
    }

    /**
     * Highlight a path in the graph (for dream visualization)
     */
    function highlightPath(path, decay = true) {
        if (!Array.isArray(path)) return;

        path.forEach((id, i) => {
            const intensity = decay
                ? 1.0 - (i / path.length) * 0.6
                : 1.0;
            graphView.setActivation(id, intensity);
        });
    }

    return { updateActivations, getMetrics, highlightPath };
}
