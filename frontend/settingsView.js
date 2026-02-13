export class SettingsView {
    constructor() {
        this.elements = {
            modelPath: document.getElementById('confModelPath'),
            modelSelector: document.getElementById('modelSelector'),
            loadBtn: document.getElementById('btnLoadModel'),
            customModelPath: document.getElementById('confCustomModelPath'),
            modelType: document.getElementById('confModelType'),
            ctx: document.getElementById('confCtx'),
            gpu: document.getElementById('confGpuLayers'),
            cacheType: document.getElementById('confCacheType'),
            temp: document.getElementById('confTemp'),
            maxTokens: document.getElementById('confMaxTokens'),
            saveBtn: document.getElementById('saveSettingsBtn'),
            msg: document.getElementById('settingsMsg')
        };

        this.init();
    }

    async init() {
        this.elements.saveBtn.addEventListener('click', () => this.save());
        this.elements.loadBtn.addEventListener('click', () => this.loadModel());

        // Populate Dropdown
        await this.fetchModels();
        await this.load();
    }

    async fetchModels() {
        try {
            const res = await fetch('http://localhost:8800/api/models');
            if (res.ok) {
                const data = await res.json();
                this.elements.modelSelector.innerHTML = '<option value="">-- Select Model --</option>';
                data.models.forEach(m => {
                    const opt = document.createElement('option');
                    opt.value = m;
                    opt.textContent = m;
                    this.elements.modelSelector.appendChild(opt);
                });

                // Smart Auto-Detect on Change
                this.elements.modelSelector.addEventListener('change', (e) => {
                    const val = e.target.value.toLowerCase();
                    let type = 'auto';

                    if (val.endsWith('.gguf')) {
                        type = 'gguf';
                    } else if (val.includes('27b') || val.includes('quantum')) {
                        type = 'quantum';
                    } else if (val === "") {
                        return; // Do nothing
                    }

                    this.elements.modelType.value = type;
                    this.applyEngineDefaults(type);
                });

                // Manual Type Change
                this.elements.modelType.addEventListener('change', (e) => {
                    this.applyEngineDefaults(e.target.value);
                });
            }
        } catch (e) { console.error("Failed to fetch models", e); }
    }

    applyEngineDefaults(type) {
        const DEFAULTS = {
            'quantum': { ctx: 8192, gpu: -1, temp: 0.7, cache: 'f16' },
            'gguf': { ctx: 4096, gpu: 25, temp: 0.8, cache: 'q8_0' },
            'native': { ctx: 2048, gpu: 0, temp: 0.7, cache: 'f16' },
            'api': { ctx: 32768, gpu: 0, temp: 1.0, cache: 'f16' },
            'auto': { ctx: 4096, gpu: -1, temp: 0.7, cache: 'f16' }
        };

        const config = DEFAULTS[type] || DEFAULTS['auto'];

        // Apply defaults to input fields
        if (this.elements.ctx) this.elements.ctx.value = config.ctx;
        if (this.elements.gpu) this.elements.gpu.value = config.gpu;
        if (this.elements.temp) this.elements.temp.value = config.temp;
        if (this.elements.cacheType) this.elements.cacheType.value = config.cache;

        console.log(`[Settings] Applied defaults for engine: ${type}`, config);
    }

    async loadModel() {
        // Prefer custom path if user entered one
        const customPath = this.elements.customModelPath.value.trim();
        const selected = customPath || this.elements.modelSelector.value;
        if (!selected) return this.showMessage("Please select a model or enter a custom path", "orange");

        this.elements.loadBtn.disabled = true;
        this.elements.loadBtn.textContent = "LOADING...";
        this.showMessage("Initializing Neural Engine...", "#00ffff");

        const payload = {
            path: selected,
            model_type: this.elements.modelType.value || "auto",
            n_ctx: parseInt(this.elements.ctx.value) || 2048,
            gpu_layers: parseInt(this.elements.gpu.value),
            temperature: parseFloat(this.elements.temp.value) || 0.7,
            cache_type_k: this.elements.cacheType.value || "f16"
        };

        console.log("Sending Load Payload:", payload);

        try {
            const res = await fetch('http://localhost:8800/api/model/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (res.ok) {
                this.elements.modelPath.value = data.path;
                this.showMessage(`Engine Online: ${selected} (${data.backend})`, "#00ff41");
            } else {
                this.showMessage(`Load Failed: ${data.error}`, "red");
            }
        } catch (e) {
            this.showMessage(`Connection Error: ${e}`, "red");
        }

        this.elements.loadBtn.disabled = false;
        this.elements.loadBtn.textContent = "INITIALIZE";
    }

    async load() {
        try {
            const res = await fetch('http://localhost:8800/api/settings');
            if (!res.ok) throw new Error("Failed to load");
            const config = await res.json();

            // Store defaults for dynamic switching
            this.backendDefaults = config.engine_defaults || {};

            this.elements.modelPath.value = config.model_path || "No Model Loaded";
            // Pre-select in dropdown if possible
            const basename = config.model_path ? config.model_path.split(/[\\/]/).pop() : "";
            if (basename) this.elements.modelSelector.value = basename;

            if (this.elements.modelType) this.elements.modelType.value = config.model_type || "auto";
            this.elements.ctx.value = config.n_ctx || 8192;
            this.elements.gpu.value = config.gpu_layers ?? -1;
            if (this.elements.cacheType) this.elements.cacheType.value = config.cache_type_k || "f16";
            this.elements.temp.value = config.temperature || 0.7;
            this.elements.maxTokens.value = config.max_tokens || 1024;
        } catch (e) {
            console.error(e);
            this.showMessage("Failed to load settings (Backend Offline?)", "red");
        }
    }

    async save() {
        const config = {
            // model_path is now set via Load button, but we preserve it
            model_path: this.elements.modelPath.value === "No Model Loaded" ? "" : this.elements.modelPath.value,
            model_type: this.elements.modelType.value || "auto",
            n_ctx: parseInt(this.elements.ctx.value),
            gpu_layers: parseInt(this.elements.gpu.value),
            cache_type_k: this.elements.cacheType.value || "f16",
            temperature: parseFloat(this.elements.temp.value),
            max_tokens: parseInt(this.elements.maxTokens.value)
        };

        this.elements.saveBtn.disabled = true;
        this.elements.saveBtn.textContent = "Saving...";

        try {
            const res = await fetch('http://localhost:8800/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            if (res.ok) {
                this.showMessage("Settings Saved!", "#00ff41");
                // Reload to confirm sync
                await this.load();
            } else {
                this.showMessage("Error Saving Settings", "red");
            }
        } catch (e) {
            this.showMessage(`Connection Error: ${e}`, "red");
        }

        this.elements.saveBtn.disabled = false;
        this.elements.saveBtn.textContent = "💾 Save Configuration";
    }

    showMessage(text, color) {
        this.elements.msg.textContent = text;
        this.elements.msg.style.color = color;
        // setTimeout(() => this.elements.msg.textContent = "", 3000);
    }
}
