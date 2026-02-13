export class PlaygroundView {
    constructor() {
        this.selectedScript = null; // Object {name, filename, args, description}
        this.currentPid = null;
        this.pollInterval = null;

        this.elements = {
            list: document.getElementById('scriptList'),
            console: document.getElementById('playgroundConsole'),
            btnRun: document.getElementById('btnRunScript'),
            btnKill: document.getElementById('btnKillScript'),
            pidLabel: document.getElementById('currentPid'),
            actLabel: document.getElementById('activeScriptLabel'),
            configPanel: document.getElementById('playgroundConfig'),
            desc: document.getElementById('scriptDescription'),
            params: document.getElementById('scriptParams')
        };

        this.init();
    }

    async init() {
        this.setupListeners();
        await this.loadScripts();
    }

    setupListeners() {
        this.elements.btnRun.addEventListener('click', () => this.runScript());
        this.elements.btnKill.addEventListener('click', () => this.stopScript());
    }

    async loadScripts() {
        try {
            const res = await fetch('http://localhost:8800/api/playground/scripts');
            const data = await res.json();
            this.renderList(data.scripts); // Now expects objects
        } catch (e) {
            console.error("Failed to load scripts", e);
            this.elements.list.innerHTML = "<div style='color:red; padding:10px;'>Connection Failed</div>";
        }
    }

    renderList(scripts) {
        this.elements.list.innerHTML = "";
        scripts.forEach(s => {
            const div = document.createElement('div');
            div.className = "script-item";

            // Handle both legacy string and new object format
            const name = s.name || s;
            const desc = s.description || "";

            div.innerHTML = `
                <div style="font-weight:bold; color:#eee;">${name}</div>
                <div style="font-size:10px; color:#666;">${s.filename || name}</div>
            `;

            div.style.padding = "10px";
            div.style.cursor = "pointer";
            div.style.borderBottom = "1px solid #333";
            div.style.transition = "background 0.2s";

            div.onmouseover = () => div.style.background = "rgba(255,255,255,0.05)";
            div.onmouseout = () => {
                if (this.selectedScript?.filename !== (s.filename || s)) div.style.background = "transparent";
            };

            div.onclick = () => {
                // Deselect others
                Array.from(this.elements.list.children).forEach(c => c.style.background = "transparent");
                // Select this
                this.selectedScript = s;
                div.style.background = "rgba(0, 255, 255, 0.1)";
                this.renderConfig(s);
            };

            this.elements.list.appendChild(div);
        });
    }

    renderConfig(script) {
        this.elements.actLabel.textContent = script.name || script;
        this.elements.desc.textContent = script.description || "No description provided.";
        this.elements.params.innerHTML = "";

        if (script.args && script.args.length > 0) {
            script.args.forEach(arg => {
                const wrapper = document.createElement('div');
                wrapper.style.display = "flex";
                wrapper.style.flexDirection = "column";

                const label = document.createElement('label');
                label.textContent = arg.label || arg.name;
                label.style.fontSize = "11px";
                label.style.color = "#aaa";
                label.style.marginBottom = "4px";

                const input = document.createElement('input');
                input.type = "text";
                input.value = arg.default || "";
                input.placeholder = arg.hint || "";
                input.className = "cyber-input"; // Assuming global style exists or default
                input.dataset.argName = arg.name;

                input.style.background = "#000";
                input.style.border = "1px solid #333";
                input.style.color = "#fff";
                input.style.padding = "5px";
                input.style.fontFamily = "monospace";

                wrapper.appendChild(label);
                wrapper.appendChild(input);
                this.elements.params.appendChild(wrapper);
            });
        }
    }

    async runScript() {
        if (!this.selectedScript) return alert("Select a script first");
        if (this.currentPid) {
            if (!confirm("A script is already running. Stop it and run new one?")) return;
            this.stopScript();
        }

        // Collect Args
        const args = [];
        const inputs = this.elements.params.querySelectorAll('input');
        inputs.forEach(input => {
            const name = input.dataset.argName;
            const val = input.value.trim();
            if (val) {
                args.push(name);
                // Handle comma lists explicitly if needed, but for now pass raw
                // server expects list. If scan_seeds expects --seeds "1,2,3", passed as ["--seeds", "1,2,3"]
                args.push(val);
            }
        });

        this.elements.console.textContent = `> Initiating ${this.selectedScript.name}...\n`;
        this.elements.console.textContent += `> Args: ${JSON.stringify(args)}\n`;

        this.elements.btnRun.disabled = true;
        this.elements.btnRun.style.opacity = 0.5;

        try {
            const res = await fetch('http://localhost:8800/api/playground/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    script: this.selectedScript.filename || this.selectedScript,
                    args: args
                })
            });
            const data = await res.json();

            if (data.pid) {
                this.currentPid = data.pid;
                this.elements.pidLabel.textContent = `PID: ${data.pid}`;
                this.startPolling(data.pid);
            } else {
                this.elements.console.textContent += `Error: ${data.error}\n`;
                this.resetUI();
            }
        } catch (e) {
            this.elements.console.textContent += `Network Error: ${e}\n`;
            this.resetUI();
        }
    }

    stopScript() {
        if (!this.currentPid) return;
        this.elements.console.textContent += `\n> Detaching monitor (PID ${this.currentPid})...\n`;
        // In real backend, we should call /kill endpoint. For now just detach UI.
        this.resetUI();
    }

    startPolling(pid) {
        if (this.pollInterval) clearInterval(this.pollInterval);

        this.pollInterval = setInterval(async () => {
            try {
                const res = await fetch(`http://localhost:8800/api/playground/logs?pid=${pid}`);
                const data = await res.json();

                if (data.lines) {
                    this.elements.console.textContent = data.lines.join("");
                    this.elements.console.scrollTop = this.elements.console.scrollHeight;
                }

                if (data.status && data.status.startsWith("finished")) {
                    this.elements.console.textContent += `\n> Process Finished: ${data.status}\n`;
                    this.resetUI();
                }
            } catch (e) {
                console.warn("Poll failed", e);
            }
        }, 1000); // 1s poll
    }

    resetUI() {
        if (this.pollInterval) clearInterval(this.pollInterval);
        this.pollInterval = null;
        this.currentPid = null;
        this.elements.pidLabel.textContent = "--";
        this.elements.btnRun.disabled = false;
        this.elements.btnRun.style.opacity = 1.0;
    }
}
