/**
 * MTI Brain 3D Monitor - Controller
 * Binds UI controls to backend actions.
 */
export function createController({ transport, monitor, log }) {

    function bindUI() {
        // Dream controls
        const seedInput = document.getElementById('seedInput');
        const stepsInput = document.getElementById('stepsInput');
        const dreamBtn = document.getElementById('dreamBtn');
        const dreamPath = document.getElementById('dreamPath');

        dreamBtn.onclick = async () => {
            const seed = seedInput.value.trim() || 'consciousness';
            const steps = parseInt(stepsInput.value) || 10;

            dreamBtn.disabled = true;
            dreamBtn.textContent = '🌙 Soñando...';

            log(`Starting dream: seed="${seed}", steps=${steps}`);

            const result = await transport.dream(seed, steps);

            if (result && result.path) {
                dreamPath.textContent = result.path.join(' → ');
                monitor.highlightPath(result.path);
                log(`Dream complete: ${result.drift_length} steps`);
            } else {
                dreamPath.textContent = 'Error en el sueño';
                log('Dream failed');
            }

            dreamBtn.disabled = false;
            dreamBtn.textContent = '🌙 Soñar';
        };

        // Interview controls
        const interviewTarget = document.getElementById('interviewTarget');
        const interviewBtn = document.getElementById('interviewBtn');
        const interviewResult = document.getElementById('interviewResult');

        interviewBtn.onclick = async () => {
            const target = interviewTarget.value.trim() || 'self';

            interviewBtn.disabled = true;
            interviewBtn.textContent = '🕵️ Interrogando...';

            log(`Starting interview: target="${target}"`);

            const result = await transport.interview(target);

            if (result) {
                interviewResult.innerHTML = `
          <b>Asociaciones:</b> ${result.associations?.join(', ') || '—'}<br>
          <b>Explicación:</b> ${result.explanation || '—'}<br>
          <small>Latencia: ${result.latency_ms || '—'}ms</small>
        `;
                log(`Interview complete: ${result.target}`);
            } else {
                interviewResult.textContent = 'Error en la interrogación';
                log('Interview failed');
            }

            interviewBtn.disabled = false;
            interviewBtn.textContent = '🕵️ Interrogar';
        };
    }

    return { bindUI };
}
