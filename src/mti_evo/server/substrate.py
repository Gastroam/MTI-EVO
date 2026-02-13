"""
MTI-EVO Substrate Server
========================
Process orchestrator for substrate-aware multiprocessing.
Production Entrypoint.
"""
import os
import sys
import json
import time
import uuid
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from multiprocessing import Queue
from typing import Dict, Optional, Any

# Core Imports
from mti_evo.core.config import MTIConfig

from mti_evo.server.playground import PlaygroundManager
from mti_evo.server.router import ControlPlaneRouter

class ResponseRouter:
    """
    Dedicated thread to route responses from InferenceProcess to HTTP threads.
    Eliminates race conditions and polling.
    """
    def __init__(self, response_queue: Queue):
        self.response_queue = response_queue
        self.futures: Dict[str, dict] = {} # {req_id: {'event': Event, 'data': None}}
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._route_loop, daemon=True)
        
    def start(self):
        self.thread.start()
        
    def _route_loop(self):
        while self.running:
            try:
                # Blocking get with timeout to allow shutdown check
                resp = self.response_queue.get(timeout=1.0)
                if resp is None: continue 
                
                req_id = resp.get('request_id')
                if req_id:
                    with self.lock:
                        if req_id in self.futures:
                            self.futures[req_id]['data'] = resp
                            self.futures[req_id]['event'].set()
                        else:
                            # Orphaned logic: Store in expring dict? 
                            # For S4, we just drop it if no waiter.
                            pass
            except Exception:
                continue

    def await_response(self, req_id: str, timeout: float) -> Optional[dict]:
        event = threading.Event()
        with self.lock:
            self.futures[req_id] = {'event': event, 'data': None}
            
        signaled = event.wait(timeout)
        
        result = None
        with self.lock:
            if signaled:
                result = self.futures[req_id]['data']
            # Always cleanup
            if req_id in self.futures:
                del self.futures[req_id]
            
        return result
        
    def stop(self):
        self.running = False

class SubstrateHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP worker inhabiting shared substrate.
    """
    # Class Variables
    request_queue: Optional[Queue] = None
    response_router: Optional[ResponseRouter] = None
    router: Optional[ControlPlaneRouter] = None
    
    def do_GET(self):
        # 1. Try Control Plane Router
        if self.router:
            result, status = self.router.handle_get(self.path)
            if result is not None:
                self._send_json(result, status)
                return

        # 2. Fallback
        self._send_json({'error': 'not_found'}, status=404)
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({'error': 'invalid_json'}, status=400)
            return

        # 1. Intercept /v1/local/reflex (Inference Queue)
        if self.path == '/v1/local/reflex':
            self._handle_reflex(data)
            return
            
        # 1b. Intercept /api/model/load (Inference IPC)
        if self.path == '/api/model/load':
             # We cannot load model here (no VRAM).
             # We must stick it in request_queue if InferenceProcess supports it.
             # Currently InferenceProcess loop handles 'infer' and 'telepathy'.
             # TODO: Update InferenceProcess to handle 'load_model' action.
             self._send_json({"error": "Dynamic model loading via IPC not implemented yet in Substrate Mode. Restart server with new config."}, 501)
             return

        # 2. Control Plane Router (Playground, Settings, etc)
        if self.router:
             result, status = self.router.handle_post(self.path, data)
             if result is not None:
                 self._send_json(result, status)
                 return

        self._send_json({'error': 'not_found'}, status=404)

    def _handle_reflex(self, data):
        if not self.request_queue:
            self._send_json({'error': 'inference_worker_unavailable'}, 503)
            return
            
        prompt = data.get('prompt', data.get('query', ''))
        req_id = str(uuid.uuid4())
        
        self.request_queue.put({
            'id': req_id,
            'action': data.get('action', 'telepathy'),
            'prompt': prompt,
            'max_tokens': data.get('max_tokens', 1024),
            'temperature': data.get('temperature', 0.7),
            'trace': data.get('trace', False),
            'timestamp': time.time()
        })
        
        # Wait using Router
        resp = self.response_router.await_response(req_id, timeout=data.get('timeout', 60))
        
        if resp:
            if resp.get('success', True):
                self._send_json({
                    'response': resp.get('response', ''),
                    'resonance': resp.get('resonance', 0.0),
                    'latency_ms': resp.get('latency_ms', 0.0)
                })
            else:
                 self._send_json({'error': resp.get('error', 'failed')}, 500)
        else:
            self._send_json({'error': 'timeout'}, 504)

    def _send_json(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        
    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def log_message(self, format, *args):
        pass # Suppress console noise

class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

class SubstrateServer:
    def __init__(self, port=8800, multiprocessing=True):
        self.port = port
        self.multiprocessing = multiprocessing
        self.config = MTIConfig() # Direct instantiation

        
        # IPC
        self.request_queue = Queue(maxsize=100) if multiprocessing else None
        self.response_queue = Queue(maxsize=100) if multiprocessing else None
        
        # Router
        self.resp_router = None
        if self.response_queue:
            self.resp_router = ResponseRouter(self.response_queue)
            
        # Shared Components (Runtime)
        from mti_evo.runtime.substrate_runtime import SubstrateRuntime
        # [Phase O] Enforce Read-Only for HTTP workers to prevent MMAP corruption
        self.runtime = SubstrateRuntime(config=self.config, read_only=True)
        
        playground_path = os.path.abspath(os.path.join(os.getcwd(), "playground"))
        self.playground_mgr = PlaygroundManager(playground_path)
        
        self.control_router = ControlPlaneRouter(
            runtime=self.runtime, 
            playground_mgr=self.playground_mgr, 
            config=self.config,
            llm_provider=None # No VRAM access here
        )
        
        self.inference_proc = None
        self.httpd = None

    def start(self):
        print("🌐 MTI-EVO Substrate Server v2.3 (Production)")
        
        if self.multiprocessing:
            self._start_inference()
            self.resp_router.start()
            
        self._start_http()
        
    def _start_inference(self):
        from mti_evo.adapters.inference import InferenceProcess
        
        model_cfg = {
            'model_type': self.config.get('model_type', 'gguf'),
            'model_path': self.config.get('model_path', ''),
            'n_ctx': self.config.get('n_ctx', 4096),
            'gpu_layers': self.config.get('gpu_layers', -1)
        }
        
        self.inference_proc = InferenceProcess(
            self.request_queue, 
            self.response_queue,
            ".mti-brain/cortex.mmap",
            model_cfg
        )
        self.inference_proc.start()
        print("🧠 Inference Process Launched")

    def _start_http(self):
        # Configure Handler
        SubstrateHTTPHandler.request_queue = self.request_queue
        SubstrateHTTPHandler.response_router = self.resp_router
        SubstrateHTTPHandler.router = self.control_router
        
        self.httpd = ThreadedHTTPServer(("", self.port), SubstrateHTTPHandler)
        print(f"🚀 HTTP Serving on {self.port}")
        
    def serve_forever(self):
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
            
    def shutdown(self):
        print("Stopping...")
        if self.request_queue:
            self.request_queue.put(None) # Poison
        if self.inference_proc:
            self.inference_proc.join()
        if self.resp_router:
            self.resp_router.stop()
        if self.httpd:
            self.httpd.server_close()

def run_substrate_server(port=8800):
    s = SubstrateServer(port)
    s.start()
    s.serve_forever()

if __name__ == "__main__":
    run_substrate_server()
