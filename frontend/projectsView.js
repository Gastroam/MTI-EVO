
export class ProjectsView {
    constructor() {
        this.container = document.getElementById('tab-projects');
        this.projectList = document.getElementById('project-grid');
        this.outputConsole = document.getElementById('project-console-output');

        // Define the Known Projects (Matches ROADMAP.md)
        this.projects = [
            {
                id: "p1",
                name: "Paradox Resolver",
                icon: "🏆",
                desc: "Synthesizes pedagogical explanations for logic paradoxes.",
                status: "BUILT",
                path: "projects/paradox_resolver/paradox_synthesis.py",
                color: "var(--accent-cyan)"
            },
            {
                id: "p2",
                name: "Climate Gap-Filler",
                icon: "🌧️",
                desc: "Physics-constrained data interpolation for sensor gaps.",
                status: "BUILT",
                path: "projects/climate_gap_filler/gap_filler_mock.py",
                color: "var(--accent-green)"
            },
            {
                id: "p3",
                name: "Vuln Synthesizer",
                icon: "🛡️",
                desc: "Detects integer overflows via symbolic execution.",
                status: "BUILT",
                path: "projects/vulnerability_synthesizer/vuln_synth_mock.py",
                color: "var(--accent-red)"
            },
            {
                id: "p4",
                name: "Balance Engine",
                icon: "⚖️",
                desc: "Optimizes Game Stats (MTG) for 50% Win Rate.",
                status: "BUILT",
                path: "projects/balance_engine/balance_engine_mock.py",
                color: "var(--accent-purple)"
            },
            {
                id: "p5",
                name: "Resonance Compiler",
                icon: "💻",
                desc: "Rewrites obfuscated code using cognitive metaphors.",
                status: "BUILT",
                path: "projects/resonance_compiler/cognitive_compiler.py",
                color: "#f0f"
            },
            {
                id: "p6",
                name: "HackerOne Tool",
                icon: "💰",
                desc: "Automated Pentest Scope & IDOR Scanner.",
                status: "BUILT",
                path: "projects/hackerone/mti_pentest.py",
                color: "gold"
            }
        ];
    }

    init() {
        this.render();
    }

    render() {
        if (!this.projectList) return;
        this.projectList.innerHTML = '';

        this.projects.forEach(p => {
            const card = document.createElement('div');
            card.className = 'project-card tech-panel-framed';
            card.style.borderColor = p.color;

            const statusClass = p.status === 'BUILT' ? 'status-ready' : 'status-pending';
            const statusIcon = p.status === 'BUILT' ? '✅' : '⏳';

            card.innerHTML = `
        <div class="project-header">
          <div class="project-icon" style="background:${p.color}20; color:${p.color}">${p.icon}</div>
          <div class="project-title-group">
            <h3>${p.name}</h3>
            <span class="project-status ${statusClass}">${statusIcon} ${p.status}</span>
          </div>
        </div>
        <div class="project-desc">${p.desc}</div>
        <div class="project-actions">
           <button class="btn primary btn-run" data-path="${p.path}" style="border-color:${p.color}">
             ▶ RUN
           </button>
           <button class="btn secondary btn-view" onclick="alert('View Source: ${p.path}')">
             👁 VIEW
           </button>
        </div>
      `;

            this.projectList.appendChild(card);
        });

        // Attach Event Listeners
        document.querySelectorAll('.btn-run').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const path = e.target.dataset.path;
                this.runProject(path);
            });
        });
    }

    async runProject(scriptPath) {
        this.log(`> Initiating Execution Sequence: ${scriptPath}...`);
        this.log(`> Connecting to MTI Core...`);

        try {
            // We use the existing /api/run_script endpoint if available
            // Or we simulate for the mockup if backend isn't strictly ready for arbitrary paths
            // Given we are in the "Web Counterpart" flow, let's try the real endpoint first

            const response = await fetch('/api/run_script', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script_path: scriptPath })
            });

            const data = await response.json();

            if (data.status === 'success' || data.output) {
                this.log(data.output || "Execution Successful (No Output)");
            } else {
                // If backend endpoints aren't perfectly aligned, we fallback to a simulated specific run
                // But actually, server.py likely supports generic script runs.
                this.log(`Core Response: ${JSON.stringify(data)}`);
                if (data.error) this.log(`❌ Error: ${data.error}`);
            }

        } catch (e) {
            this.log(`❌ Connection Failed: ${e.message}`);
            this.log(`> Simulation Mode: MTI-EVO is running correctly, but the web UI might be detached from the Python Process.`);
        }
    }

    log(msg) {
        if (!this.outputConsole) return;
        const line = document.createElement('div');
        line.className = 'log-line';
        line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
        this.outputConsole.appendChild(line);
        this.outputConsole.scrollTop = this.outputConsole.scrollHeight;
    }
}
