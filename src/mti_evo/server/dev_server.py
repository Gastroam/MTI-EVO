"""
MTI-EVO Unified Server (Refactored)
===================================
Legacy/Dev server running everything in a single process.
Connects VRAM, API, and HTTP in one loop.
"""
import socketserver
import os
from http.server import BaseHTTPRequestHandler
import json

from mti_evo.config import load_config
from mti_evo.engines.registry import EngineRegistry, discover_engines
from mti_evo.server.playground import PlaygroundManager
from mti_evo.server.router import ControlPlaneRouter

# Plugin: IDRE Security
try:
    from mti_evo_plugins.idre.sanitizer import get_sanitizer
except ImportError:
    # Build a dummy sanitizer if plugin missing?
    # Or just warn.
    print("[WARN] IDRE plugin not found. Security features disabled.")
    def get_sanitizer(path=None): return None

class UnifiedHandler(BaseHTTPRequestHandler):
    """
    Delegates to ControlPlaneRouter.
    """
    router = None
    sanitizer = None
    
    def do_GET(self):
        if self.router:
            result, status = self.router.handle_get(self.path)
            if result is not None:
                self._send_json(result, status)
                return
        self._send_json({'error': 'not_found'}, 404)
        
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        data = json.loads(body) if body else {}

        # 1. Reflex needs Sanitizer + Local Context (Unified exclusive feature)
        # The router doesn't know about sanitizer yet (maybe it should?)
        # For now, we keep specific Unified logic here for backward compat
        if self.path == '/v1/local/reflex':
            self._handle_reflex(data)
            return

        if self.router:
             result, status = self.router.handle_post(self.path, data)
             if result is not None:
                 self._send_json(result, status)
                 return
                 
        self._send_json({'error': 'not_found'}, 404)

    def _handle_reflex(self, data):
        prompt = data.get('prompt', '')
        # Sanitize
        if self.sanitizer:
            prompt = self.sanitizer.sanitize(prompt)

        llm = self.router.llm
        if llm:
            # EngineProtocol: infer returns EngineResult
            resp = llm.infer(prompt, max_tokens=data.get('max_tokens', 1024))
            self._send_json({"response": resp.text})
        else:
            self._send_json({"error": "No LLM loaded"}, 500)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
        
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
    def log_message(self, format, *args):
        pass

class UnifiedServer:
    def __init__(self, port=8800):
        self.port = port
        self.config = load_config()
        
        # Load LLM Locally (Unified Mode)
        discover_engines()
        engine_type = self.config.model_type
        if engine_type == "auto":
            engine_type = "gguf"
        
        try:
             engine_cfg = self.config.to_dict(include_private=True)
             self.llm = EngineRegistry.create(engine_type, engine_cfg)
             # Auto-load in dev mode? 
             # Protocol says load(config).
             self.llm.load(engine_cfg)
             print(f"   [OK] Engine loaded: {engine_type}")
        except Exception as e:
             print(f"   [WARN] Engine load failed: {e}")
             self.llm = None
        
        # Shared Components (Runtime)
        from mti_evo.runtime.substrate_runtime import SubstrateRuntime
        self.runtime = SubstrateRuntime(config=self.config, persistence_id="unified_dev")
        
        self.playground_mgr = PlaygroundManager(os.path.abspath("playground"))
        
        self.router = ControlPlaneRouter(
            runtime=self.runtime,
            playground_mgr=self.playground_mgr,
            config=self.config,
            llm_provider=self.llm
        )
        
        UnifiedHandler.router = self.router
        UnifiedHandler.sanitizer = get_sanitizer()

    def start(self):
        print(f"[SERVER] MTI-EVO Unified Server (dev mode) on {self.port}")
        
        # Threaded Server
        class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            daemon_threads = True
            allow_reuse_address = True
            
        with ThreadedHTTPServer(("", self.port), UnifiedHandler) as httpd:
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("Stopping...")

def run_dev_server(port=8800):
    s = UnifiedServer(port)
    s.start()

if __name__ == "__main__":
    run_dev_server()
