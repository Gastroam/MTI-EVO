"""
MTI-EVO Unified Server
======================
A single HTTP server exposing the Brain control plane for frontend integration.
Combines the Telepathy Bridge and API endpoints.

API Tiers:
- PUBLIC:      /status, /api/graph, /api/probe, /api/models, /api/model/load, /api/settings
- RESEARCHER:  /api/attractors, /api/events, /api/metrics, /api/playground/*, /api/inject
- EXPERIMENTAL: /v1/local/reflex, /control/dream, /control/interview, /control/mutate
"""
import http.server
import socketserver
import threading
import queue
import time
import random
import json
import os
import subprocess
import sys
import sys
import subprocess

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mti_evo.api import EvoAPI
from mti_evo.llm_adapter import LLMAdapter
from mti_evo.security.sanitizer import get_sanitizer
import mti_evo.telemetry as telemetry

PORT = 8800  # New unified port
TELEPATHY_PORT = 8766  # Internal bridge for LLM

class ThoughtLoop(threading.Thread):
    """Background thread for spontaneous LLM thoughts."""
    def __init__(self, llm_adapter):
        super().__init__(daemon=True)
        self.llm = llm_adapter
        self.running = True

    def run(self):
        telemetry.log_event("system", "Thought Loop Started (Spontaneous Mode)")
        while self.running:
            # [Metrics] Log System Vitals
            try:
                if self.llm:
                    # Log simulated VRAM
                     # ... or just neuron count
                     pass
                     
                status = EvoControlHandler.api.get_status()
                telemetry.log_metric("neurons", status.get("neurons", 0))
                
                # Simulate latency drift for visuals
                import random
                lat = 20 + random.random() * 10
                telemetry.log_metric("latency", lat)
                
            except Exception as e:
                pass

            time.sleep(5) # Tick faster for metrics (5s)

            # Wait for a random interval for THOUGHTS (separate counter or check)
            # Simplified: Every ~6th tick (30s) generate thought
            # Wait for a random interval for THOUGHTS (separate counter or check)
            # [REMOVED] Spontaneous thoughts disabled by user request.
            pass
 
class PlaygroundManager:
    """
    Manages execution of playground scripts.
    Handles auditing (file logging) and process management.
    """
    def __init__(self, playground_dir):
        self.playground_dir = playground_dir
        self.running_processes = {} # {pid: {'proc': proc, 'log_file': path, 'script': name}}
        self.logs_dir = os.path.join(playground_dir, "..", "logs", "playground")
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def list_scripts(self):
        """Returns list of scripts with metadata."""
        try:
            # Read Manifest
            manifest = {}
            manifest_path = os.path.join(self.playground_dir, "manifest.json")
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                except:
                    pass

            scripts = []
            for f in os.listdir(self.playground_dir):
                if f.endswith(".py") and not f.startswith("__"):
                    meta = manifest.get(f, {
                        "name": f, 
                        "description": "No description provided.",
                        "args": []
                    })
                    # Ensure filename is attached
                    meta['filename'] = f
                    scripts.append(meta)
            
            # Sort by Name
            return sorted(scripts, key=lambda x: x['name'])
        except Exception as e:
            return []

    def run_script(self, script_name, args=None):
        """Spawns a script execution with optional arguments."""
        # Validate existence (simple check)
        if not os.path.exists(os.path.join(self.playground_dir, script_name)):
             return {"error": "Script not found"}
            
        script_path = os.path.join(self.playground_dir, script_name)
        timestamp = int(time.time())
        log_filename = f"{timestamp}_{script_name}.log"
        log_path = os.path.join(self.logs_dir, log_filename)
        
        try:
            # Open log file for redirection
            log_file = open(log_path, "w")
            
            # Build Command
            cmd = [sys.executable, "-u", script_path]
            if args and isinstance(args, list):
                cmd.extend(args)
            
            # Spawn subprocess (Non-blocking)
            proc = subprocess.Popen(
                cmd,
                cwd=self.playground_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT, 
                text=True
            )
            
            self.running_processes[proc.pid] = {
                'proc': proc,
                'log_file_handle': log_file,
                'log_path': log_path,
                'script': script_name,
                'start_time': timestamp
            }
            
            return {"pid": proc.pid, "status": "started", "log_file": log_filename}
        except Exception as e:
            return {"error": str(e)}

    def get_log_tail(self, pid, lines=50):
        """Reads the last N lines of the log file for a process."""
        try:
            pid = int(pid)
        except:
            return {"error": "Invalid PID"}

        if pid not in self.running_processes:
             return {"error": "Process not active (or finished and cleared)"}
             
        info = self.running_processes[pid]
        proc = info['proc']
        
        # Check if done
        status = "running"
        if proc.poll() is not None:
            status = f"finished (code {proc.returncode})"
            # Close handle if not closed
            if not info['log_file_handle'].closed:
                 info['log_file_handle'].close()

        # Read file from disk
        try:
            if os.path.exists(info['log_path']):
                with open(info['log_path'], 'r') as f:
                    content = f.readlines()
                    return {
                        "pid": pid,
                        "status": status,
                        "lines": content[-lines:],
                        "full_path": info['log_path']
                    }
            return {"pid": pid, "status": status, "lines": [], "error": "Log file not found"}
        except Exception as e:
            return {"error": f"Failed to read log: {e}"}

    def stop_process(self, pid):
         try:
             pid = int(pid)
             if pid in self.running_processes:
                 proc = self.running_processes[pid]['proc']
                 proc.kill()
                 return {"status": "killed"}
             return {"error": "Process not found"}
         except:
             return {"error": "Invalid PID"}

class EvoControlHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Handler for the MTI-EVO Control Plane."""
    
    # Shared State (Singleton Pattern)
    api = None
    llm = None
    sanitizer = None

    @classmethod
    def initialize(cls):
        """Initialize shared resources once."""
        # Load Config FIRST
        from mti_evo.config import load_config
        cls.config = load_config()

        if cls.api is None:
            cls.api = EvoAPI()
        
        if cls.llm is None:
            # Pass config on init to avoid double-loading (GGUF -> Quantum)
            # [FIX] Disable Auto-Load for Frontend Selector
            cls.llm = LLMAdapter(config=cls.config, auto_load=False)
            
        if cls.sanitizer is None:
            cls.sanitizer = get_sanitizer()

        if getattr(cls, 'playground_mgr', None) is None:
            playground_path = os.path.abspath(os.path.join(os.getcwd(), "playground"))
            cls.playground_mgr = PlaygroundManager(playground_path)
            
        # Update just in case it was already loaded (singleton safety)
        if cls.llm and cls.llm.config != cls.config:
             cls.llm.update_config(cls.config)

        # [PHASE 27] Semantic Grounding (Code DNA Injection)
        # We scan the codebase and feed the symbols to Broca so neurons match real code.
        try:
             # Need to import inside to avoid circular deps if any
             from mti_evo.ghost.indexer import GhostIndexer
             
             print("👻 Ghost Protocol: Scanning Codebase...")
             indexer = GhostIndexer(root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
             symbols = indexer.scan()
             
             count = 0
             if cls.api.broca:
                 print(f"👻 Ghost Protocol: Injecting {len(symbols)} symbols into Cortex...")
                 for sym in symbols:
                     # Create a sentence like "class AuthService auth.py"
                     # Pass map {token: label} to assign label to the symbol name
                     name = sym['name']
                     label = f"{sym['type'].upper()}:{sym['name']}" # e.g. CLASS:AuthService
                     
                     # Map the token (name) to the label we want stored
                     labels = {name.lower(): label}
                     cls.api.broca.process_thought(name, learn=True, labels=labels)
                     count += 1
                     
             print(f"👻 Ghost Protocol: Grounded {count} semantic concepts.")
        except Exception as e:
             # If fail (e.g. ghost module not found yet), just warn
             print(f"👻 Ghost Protocol Warning: {e}")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        # [PHASE 55] Settings API
        if self.path == '/api/settings':
             from mti_evo.config import load_config
             self._send_json(load_config())
             return

        # [PHASE 54] Playground API
        if self.path == '/api/playground/scripts':
             scripts = self.playground_mgr.list_scripts()
             self._send_json({"scripts": scripts})
             return
             
        # [PHASE 54] Model Selection API
        if self.path == '/api/models':
            # List .gguf files AND directories (for Native/Quantum) in models/
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            model_files = []
            if os.path.exists(models_dir):
                for f in os.listdir(models_dir):
                    full_p = os.path.join(models_dir, f)
                    if os.path.isdir(full_p):
                        model_files.append(f) # Add directory (e.g. gemma-27b)
                    elif f.endswith(".gguf") or f.endswith(".safetensors"):
                        model_files.append(f)
            self._send_json({"models": model_files})
            return
             
        if self.path.startswith('/api/playground/logs'):
             from urllib.parse import urlparse, parse_qs
             query = parse_qs(urlparse(self.path).query)
             pid = query.get('pid', [None])[0]
             if pid:
                 self._send_json(self.playground_mgr.get_log_tail(pid))
             else:
                 self.send_error(400, "Missing pid")
             return

        if self.path == '/status':
            result = self.api.get_status()
            self._send_json(result)
        elif self.path == '/' or self.path == '/help':
            # Comprehensive API documentation
            self._send_json({
                "service": "MTI-EVO Control Plane",
                "version": "2.2.6",
                "tiers": {
                    "public": {
                        "GET /status": "Brain health, neuron count, mode",
                        "GET /api/graph": "Graph topology (nodes, edges)",
                        "GET /api/probe?seed=X": "Deep inspection of a single neuron",
                        "GET /api/models": "List available models",
                        "POST /api/model/load": "Load a model by path",
                        "GET /api/settings": "Retrieve current config",
                        "POST /api/settings": "Update live config"
                    },
                    "researcher": {
                        "GET /api/attractors": "Attractor field scan",
                        "GET /api/events": "Telemetry events",
                        "GET /api/metrics": "Metrics history",
                        "GET /api/playground/scripts": "List playground scripts",
                        "POST /api/playground/run": "Execute a playground script",
                        "GET /api/playground/logs?pid=X": "Tail log for running script",
                        "POST /api/inject": "Resonance Injection (requires resonant engine)"
                    },
                    "experimental": {
                        "POST /v1/local/reflex": "Legacy Telepathy Bridge",
                        "POST /control/dream": "Trigger Hebbian Drift",
                        "POST /control/interview": "Cognitive Interview",
                        "POST /control/mutate": "Mutate a dream archetype",
                        "GET /api/dreams/archetypes": "Dream archetype analysis"
                    }
                },
                "engines": ["gguf", "native", "quantum", "resonant", "bicameral", "qoop", "hybrid", "api"]
            })
        elif self.path == '/api/graph':
            result = self.api.get_graph_topology()
            self._send_json(result)
        elif self.path == '/api/metrics':
            # [PHASE 48] Metrics History
            history = telemetry.get_metric_history()
            self._send_json({"history": history})

        # [PHASE 50] Dream Archetypes Endpoint
        elif self.path == "/api/dreams/archetypes":
            # Lazy load analyzer to avoid startup freeze
            if not hasattr(self.server, 'dream_analyzer'):
                from mti_evo.dream_analyzer import DreamAnalyzer
                # Use the existing LLMAdapter from the server if available
                # Note: self.server in BaseHTTPRequestHandler refers to the TCPServer instance
                # We attached llm to EvoControlHandler class, but we need an instance
                # The LLM is available at EvoControlHandler.llm
                # The Broca instance is available at EvoControlHandler.api.broca
                self.server.dream_analyzer = DreamAnalyzer(
                    adapter=EvoControlHandler.llm,
                    broca=getattr(EvoControlHandler.api, 'broca', None)
                )
            
            # Demo Data (In future, fetch from database)
            dreams = [
                {"text": "I was flying over a vast city, the wind in my hair, feeling absolute freedom.", "mood": "Ecstatic", "vividness": 9},
                {"text": "Soaring above the clouds, looked down at the tiny ocean below. No gravity.", "mood": "Peaceful", "vividness": 8},
                {"text": "Levitating just a few feet off the ground, moving effortlessly through the streets.", "mood": "Curious", "vividness": 7},
                {"text": "Debugging a recursive function that kept calling itself into infinity.", "mood": "Frustrated", "vividness": 5},
                {"text": "I was inside the computer, watching data streams flow like neon rivers. I fixed a syntax error in the sky.", "mood": "Focused", "vividness": 8},
                {"text": "Writing python code on a blackboard but the chalk kept changing colors.", "mood": "Confused", "vividness": 4},
                {"text": "My teeth started crumbling and falling out one by one.", "mood": "Terrified", "vividness": 10},
                {"text": "Running down a hallway that never ends, being chased by a shadow.", "mood": "Anxious", "vividness": 8},
                {"text": "I lost my wallet and couldn't find my way home. Everything was dark.", "mood": "Panic", "vividness": 6},
                {"text": "Eating a sandwich made of glass.", "mood": "Weird", "vividness": 3},
                {"text": "A giant cat was explaining quantum physics to me.", "mood": "Amused", "vividness": 7}
            ]
            
            archetypes = self.server.dream_analyzer.analyze_archetypes(dreams)
            self.server.last_archetypes = archetypes # Cache for mutation
            self._send_json({"archetypes": archetypes})
        elif self.path == '/api/events':
            # Pop all messages from telemetry queue
            events = telemetry.get_events()
            self._send_json({"events": events})
        elif self.path.startswith('/api/attractors'):
            # [PHASE 39] Dynamic Attractor Field Scan
            # Parse query params manually since we don't have urllib here
            query = {}
            if '?' in self.path:
                qs = self.path.split('?')[1]
                for pair in qs.split('&'):
                    if '=' in pair:
                        k, v = pair.split('=')
                        query[k] = v
            
            scan_all = query.get('all', 'true') == 'true'
            start = int(query.get('start')) if query.get('start') else None
            end = int(query.get('end')) if query.get('end') else None
            
            attractors = self.api.get_attractor_field(start_seed=start, end_seed=end, scan_all=scan_all)
            self._send_json({"attractors": attractors})
        elif self.path.startswith('/api/probe'):
            # [PHASE 39] Probe Specific Neuron
            query = {}
            if '?' in self.path:
                qs = self.path.split('?')[1]
                for pair in qs.split('&'):
                    if '=' in pair:
                        k, v = pair.split('=')
                        query[k] = v
             
            seed = query.get('seed')
            if not seed:
                self.send_error(400, "Missing seed parameter")
                return

            result = self.api.probe_neuron(seed)
            self._send_json(result)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        # [PHASE 56] Project Runner
        if self.path == '/api/run_script':
            script_path = data.get("script_path")
            if not script_path:
                self.send_error(400, "Missing script_path")
                return
            
            # Resolve Path (relative to project root)
            # server.py is in src/mti_evo, we want root d:\VMTIDE\MTI-EVO
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            full_path = os.path.join(root_dir, script_path)
            
            if not os.path.exists(full_path):
                # Fallback check
                if os.path.exists(script_path): full_path = script_path
                else:
                    self._send_json({"status": "error", "error": f"Script not found: {full_path}"})
                    return

            print(f"🚀 Executing Project: {full_path}")
            try:
                # Run with timeout to prevent hangs
                res = subprocess.run([sys.executable, full_path], cwd=root_dir, capture_output=True, text=True, timeout=60)
                output = (res.stdout + "\n" + res.stderr).strip()
                if not output: output = "[Process finished with no output]"
                self._send_json({"status": "success", "output": output})
            except Exception as e:
                self._send_json({"status": "error", "error": str(e), "output": str(e)})
            return

        # [PHASE 55] Settings Update
        if self.path == '/api/settings':
            from mti_evo.config import save_config
            new_conf = save_config(data)
            if self.llm:
                self.llm.update_config(new_conf)
            self._send_json({"status": "updated", "config": new_conf})
            return

        # [RESEARCHER API] Resonance Injection
        if self.path == '/api/inject':
            # Cross-Model Resonance Injection endpoint
            # Requires: ResonantEngine or engine with resonance_loader
            layer_idx = data.get("layer", 0)
            target_key = data.get("key", "self_attn.q_proj.weight")
            alpha = float(data.get("alpha", 0.1))
            vector_data = data.get("vector")  # List of floats or None for placeholder
            
            # Check if current engine supports injection
            if hasattr(EvoControlHandler.llm, 'engine') and hasattr(EvoControlHandler.llm.engine, 'loader'):
                loader = EvoControlHandler.llm.engine.loader
                if hasattr(loader, 'inject_vector'):
                    try:
                        import torch
                        if vector_data:
                            vector = torch.tensor(vector_data, dtype=torch.float32)
                        else:
                            # Placeholder: Create noise vector for testing
                            vector = torch.randn(1024, 1024) * 0.01
                        
                        success = loader.inject_vector(layer_idx, target_key, vector, alpha)
                        self._send_json({
                            "status": "injected" if success else "failed",
                            "layer": layer_idx,
                            "key": target_key,
                            "alpha": alpha
                        })
                    except Exception as e:
                        self._send_json({"status": "error", "error": str(e)}, status=500)
                    return
            
            self._send_json({
                "status": "error",
                "error": "Current engine does not support injection. Use model_type='resonant'."
            }, status=400)
            return

        # [PHASE 52] Model Loading API
        if self.path == '/api/model/load':
            model_path_rel = data.get("path") # Relative to 'models/' preferably
            if not model_path_rel:
                 self.send_error(400, "Missing model path")
                 return
            
            # Construct absolute path to be safe
            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
            full_path = os.path.join(models_dir, model_path_rel)
            
            if not os.path.exists(full_path):
                 # Try as absolute path
                 if os.path.exists(model_path_rel):
                     full_path = model_path_rel
                 else:
                     self._send_json({"error": "Model file not found", "path": full_path}, status=404)
                     return

            print(f"🔄 Requesting Model Load: {full_path}")
            print(f"🔍 Load Payload Params: {data}")
            
            # Update Config with provided params
            EvoControlHandler.config["model_path"] = full_path
            
            # Apply other init parameters if provided
            if "n_ctx" in data: EvoControlHandler.config["n_ctx"] = int(data["n_ctx"])
            if "gpu_layers" in data: EvoControlHandler.config["gpu_layers"] = int(data["gpu_layers"])
            if "n_threads" in data: EvoControlHandler.config["n_threads"] = int(data["n_threads"])
            if "temperature" in data: EvoControlHandler.config["temperature"] = float(data["temperature"])
            if "model_type" in data: EvoControlHandler.config["model_type"] = data["model_type"]
            
            # [Advanced] KV Cache Quantization
            if "cache_type_k" in data: EvoControlHandler.config["cache_type_k"] = data["cache_type_k"] # f16, q8_0, q4_0
            
            if EvoControlHandler.llm:
                # Force backend reset if necessary
                EvoControlHandler.llm.update_config(EvoControlHandler.config)
                
                # If backend was 'none' (Lazy), update_config might not trigger load if path matches default
                if EvoControlHandler.llm.backend == "none":
                     EvoControlHandler.llm.load_model()
                     
            self._send_json({"status": "loaded", "path": full_path, "backend": EvoControlHandler.llm.backend, "config": EvoControlHandler.config})
            return

        # [PHASE 54] Playground Run
        if self.path == '/api/playground/run':
            script = data.get("script")
            args = data.get("args", []) # Get optional args
            if not script:
                 self.send_error(400, "Missing script name")
                 return
            
            result = EvoControlHandler.playground_mgr.run_script(script, args)
            self._send_json(result)
            return

        # [PHASE 52] Mutation Control
        if self.path == '/control/mutate':
            arch_id = data.get("id")
            if arch_id is None:
                self.send_error(400, "Missing archetype id")
                return
            
            # Find in cache
            if not hasattr(self.server, 'last_archetypes'):
                self.send_error(400, "No archetypes analyzed yet. Call GET /api/dreams/archetypes first.")
                return
                
            target = next((a for a in self.server.last_archetypes if a['id'] == arch_id), None)
            if not target:
                self.send_error(404, "Archetype ID not found in cache")
                return
                
            # Trigger Mutation
            if hasattr(self.server, 'dream_analyzer'):
                result_msg = self.server.dream_analyzer.mutate_archetype(target)
                self._send_json({"status": "success", "message": result_msg})
            else:
                 self.send_error(500, "DreamAnalyzer not initialized")
            return

        if self.path == '/control/dream':
            seed = data.get("seed", "consciousness")
            steps = data.get("steps", 10)
            result = self.api.trigger_dream(seed, steps)
            self._send_json(result)

        elif self.path == '/control/interview':
            target = data.get("target", "self")
            dream_result = self.api.trigger_dream(target, steps=5)
            associations = dream_result.get("path", [])[1:6]
            
            prompt = (
                f"I asked about '{target}'. Associations: {associations}. "
                f"Explain the logic briefly."
            )
            llm_response = self.llm.infer(prompt, max_tokens=150)
            
            self._send_json({
                "target": target,
                "associations": associations,
                "explanation": llm_response.text,
            })

        elif self.path == '/v1/local/reflex':
            action = data.get("action")
            if action == "telepathy":
                raw_input = data.get("prompt", "")
                max_tokens = data.get("max_tokens", 1024)
                
                # 1. Sanitize Input (The Blood-Brain Barrier)
                clean_input = self.sanitizer.sanitize(raw_input, context="cloud")
                
                # 2. Get Real-time System State (Self-Awareness)
                status = self.api.get_status()
                neuron_count = status.get("neurons", 0)
                app_mode = status.get("mode", "Unknown")
                
                # [PHASE 28] Collect Active Cortical Concepts (Semantic Grounding)
                cortical_context = self.get_cortical_context()

                # 3. Context Wrapper (Neutral Grounding)
                # User requested removal of rigid Persona to allow direct manipulation.
                # We still provide the Graph State as "Context" for the model to use IF it wants.
                
                system_prompt = (
                    f"System Context: {neuron_count} Active Neurons | Mode: {app_mode}\n"
                    f"Active Cortical Regions: [{cortical_context}]\n\n"
                    f"{clean_input}"
                )

                response = self.llm.infer(system_prompt, max_tokens=max_tokens)
                self._send_json({"response": response.text})
            else:
                self.send_error(400, "Unknown action")
        else:
            self.send_error(404, "Endpoint Not Found")

    @classmethod
    def get_cortical_context(cls):
        """Helper to get semantic labels for prompts."""
        active_concepts = []
        if cls.api and cls.api.broca:
            # Get labels from active tissue
            for n in cls.api.broca.cortex.active_tissue.values():
                if hasattr(n, 'label') and n.label:
                    active_concepts.append(n.label)
        
        # Limit to top 20 random concepts to avoid token overflow but give context
        import random
        if len(active_concepts) > 20:
            cortical_context = ", ".join(random.sample(active_concepts, 20))
        else:
            cortical_context = ", ".join(active_concepts)
        
        if not cortical_context:
            cortical_context = "No accessible semantic labels (Training Mode)"
        return cortical_context

    def log_message(self, format, *args):
        pass


def run_server(port=PORT):
    """Start the MTI-EVO Control Plane Server."""
    EvoControlHandler.initialize()
    
    # Start Spontaneous Thought Loop
    thought_thread = ThoughtLoop(EvoControlHandler.llm)
    thought_thread.start()
    
    print(f"🧠 MTI-EVO Control Plane v2.1 (Autonomy Enabled)")
    print(f"   Port: {port}")
    print("   Endpoints: /status, /api/graph, /api/events")
    print("Ready for connections...")
    
    # [FIX] Threading Mixin for Concurrency
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True

    # [FIX] Enable address reuse to prevent port sticking
    ThreadedHTTPServer.allow_reuse_address = True
    
    with ThreadedHTTPServer(("", port), EvoControlHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server Stopping (SIGINT)...")
        except Exception as e:
            print(f"\n❌ Server Error: {e}")
        finally:
            thought_thread.running = False
            httpd.server_close()
            
            # [FIX] Explicit Unload of LLM to free VRAM
            print("🧹 Unloading Model...")
            if EvoControlHandler.llm:
                 if hasattr(EvoControlHandler.llm, 'unload_model'):
                     EvoControlHandler.llm.unload_model()
                 elif hasattr(EvoControlHandler.llm, 'unload'):
                     EvoControlHandler.llm.unload()
                     
            print("✅ Server Socket Closed.")


def main():
    """CLI entry point for mti-server command."""
    import argparse
    parser = argparse.ArgumentParser(description="MTI-EVO Control Plane Server")
    parser.add_argument("-p", "--port", type=int, default=PORT, help="Server port (default: 8800)")
    args = parser.parse_args()
    run_server(args.port)


if __name__ == "__main__":
    main()
