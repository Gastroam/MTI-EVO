"""
MTI-EVO Substrate Server
========================
Process orchestrator for substrate-aware multiprocessing.

Architecture:
- Single InferenceProcess: Holds VRAM model, animates substrate
- Multiple HTTP Workers: Inhabit shared mmap substrate (ThreadingMixIn)
- Queue IPC: Workers enqueue requests, inference process responds
"""
import os
import sys
import json
import time
import uuid
import socketserver
from http.server import BaseHTTPRequestHandler
from multiprocessing import Queue
from typing import Dict, Optional, Any

# Import existing components
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mti_evo.config import load_config


class SubstrateHTTPHandler(BaseHTTPRequestHandler):
    """
    HTTP worker inhabiting shared substrate (no VRAM model).
    
    These workers:
    1. Read/write to shared mmap substrate (zero-copy)
    2. Enqueue inference requests to the metabolic heart
    3. Return responses without holding VRAM
    """
    
    # Class variables set by SubstrateServer
    request_queue: Optional[Queue] = None
    response_queue: Optional[Queue] = None
    pending_responses: Dict[str, Any] = {}
    config: dict = {}
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health' or self.path == '/status':
            self._send_json({
                'status': 'healthy',
                'architecture': 'substrate_multiprocessing',
                'inference_mode': 'queue_ipc'
            })
        elif self.path == '/help':
            self._send_json({
                'endpoints': {
                    '/health': 'Health check',
                    '/v1/local/reflex': 'POST - Inference with resonance',
                    '/v1/local/resonate': 'POST - Resonance calculation only'
                },
                'architecture': 'Substrate Multiprocessing',
                'inference_process': 'Single VRAM holder',
                'http_workers': 'ThreadingMixIn (mmap inhabitants)'
            })
        else:
            self._send_json({'error': 'not_found'}, status=404)
    
    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        
        try:
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({'error': 'invalid_json'}, status=400)
            return
        
        if self.path == '/v1/local/reflex':
            self._handle_reflex(data)
        elif self.path == '/v1/local/resonate':
            self._handle_resonate(data)
        else:
            self._send_json({'error': 'not_found'}, status=404)
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()
    
    def _handle_reflex(self, data: dict):
        """
        Handle inference request via queue to inference process.
        """
        if self.request_queue is None:
            self._send_json({'error': 'inference_not_available'}, status=503)
            return
        
        prompt = data.get('prompt', data.get('query', ''))
        if not prompt:
            self._send_json({'error': 'prompt_required'}, status=400)
            return
        
        # Generate request ID
        req_id = str(uuid.uuid4())
        
        # Enqueue to inference process
        self.request_queue.put({
            'id': req_id,
            'action': data.get('action', 'telepathy'),
            'prompt': prompt,
            'max_tokens': data.get('max_tokens', 1024),
            'temperature': data.get('temperature', 0.7),
            'timestamp': time.time()
        })
        
        # Wait for response (with timeout)
        timeout = data.get('timeout', 60)  # 60s default
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                # Non-blocking check
                if not self.response_queue.empty():
                    resp = self.response_queue.get_nowait()
                    
                    if resp.get('request_id') == req_id:
                        if resp.get('success', True):
                            self._send_json({
                                'response': resp.get('response', ''),
                                'resonance': resp.get('resonance', 0.0),
                                'latency_ms': resp.get('latency_ms', 0.0)
                            })
                        else:
                            self._send_json({
                                'error': resp.get('error', 'inference_failed')
                            }, status=500)
                        return
                    else:
                        # Not our response, put it back for other workers
                        self.pending_responses[resp.get('request_id')] = resp
                        
            except Exception:
                pass
            
            # Check pending responses
            if req_id in self.pending_responses:
                resp = self.pending_responses.pop(req_id)
                self._send_json({
                    'response': resp.get('response', ''),
                    'resonance': resp.get('resonance', 0.0),
                    'latency_ms': resp.get('latency_ms', 0.0)
                })
                return
            
            time.sleep(0.01)  # 10ms poll interval
        
        self._send_json({'error': 'inference_timeout'}, status=504)
    
    def _handle_resonate(self, data: dict):
        """
        Handle resonance-only request (no LLM inference).
        """
        prompt = data.get('prompt', data.get('query', ''))
        if not prompt:
            self._send_json({'error': 'prompt_required'}, status=400)
            return
        
        # Direct resonance calculation (if we have local broca)
        # For now, just acknowledge - full implementation would use shared mmap
        self._send_json({
            'status': 'resonate_queued',
            'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt
        })
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response with CORS headers."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _send_cors_headers(self):
        """Add CORS headers."""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded HTTP server for I/O-bound request handling."""
    daemon_threads = True
    allow_reuse_address = True


class SubstrateServer:
    """
    Master process orchestrator.
    
    Spawns:
    1. Single InferenceProcess (VRAM holder)
    2. Threaded HTTP server (substrate inhabitants)
    """
    
    def __init__(self, port: int = 8800, multiprocessing: bool = True):
        self.port = port
        self.multiprocessing = multiprocessing
        self.config = load_config()
        
        # IPC queues
        self.request_queue = Queue(maxsize=100) if multiprocessing else None
        self.response_queue = Queue(maxsize=100) if multiprocessing else None
        
        self.inference_proc = None
        self.httpd = None
    
    def start(self):
        """Start the substrate server."""
        print("🌐 MTI-EVO Substrate Server v2.2")
        print(f"   Port: {self.port}")
        print(f"   Mode: {'Substrate Multiprocessing' if self.multiprocessing else 'Threading'}")
        
        if self.multiprocessing:
            self._start_inference_process()
        
        self._start_http_server()
    
    def _start_inference_process(self):
        """Launch the single VRAM-holding inference process."""
        from mti_evo.inference_process import InferenceProcess
        
        model_config = {
            'model_type': self.config.get('model_type', 'gguf'),
            'model_path': self.config.get('model_path', ''),
            'n_ctx': self.config.get('n_ctx', 4096),
            'gpu_layers': self.config.get('gpu_layers', -1),
            'temperature': self.config.get('temperature', 0.7),
        }
        
        self.inference_proc = InferenceProcess(
            request_queue=self.request_queue,
            response_queue=self.response_queue,
            mmap_path=".mti-brain/cortex.mmap",
            model_config=model_config
        )
        self.inference_proc.start()
        
        # Give it time to load
        print("⏳ Waiting for inference process to load model...")
        time.sleep(2)
    
    def _start_http_server(self):
        """Start the threaded HTTP server."""
        # Configure handler with queues
        SubstrateHTTPHandler.request_queue = self.request_queue
        SubstrateHTTPHandler.response_queue = self.response_queue
        SubstrateHTTPHandler.config = self.config
        
        self.httpd = ThreadedHTTPServer(("", self.port), SubstrateHTTPHandler)
        
        print(f"🌐 HTTP Server READY on port {self.port}")
        if self.inference_proc:
            print(f"🧠 Inference Process PID: {self.inference_proc.pid}")
        print("Ready for connections...")
    
    def serve_forever(self):
        """Run until interrupted."""
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown."""
        # Send poison pill to inference process
        if self.request_queue:
            self.request_queue.put(None)
        
        # Wait for inference process
        if self.inference_proc:
            self.inference_proc.join(timeout=5)
            if self.inference_proc.is_alive():
                self.inference_proc.terminate()
        
        # Close HTTP server
        if self.httpd:
            self.httpd.server_close()
        
        print("✅ Server shutdown complete.")


def run_substrate_server(port: int = 8800, multiprocessing: bool = True):
    """Entry point for substrate server."""
    server = SubstrateServer(port=port, multiprocessing=multiprocessing)
    server.start()
    server.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MTI-EVO Substrate Server")
    parser.add_argument("-p", "--port", type=int, default=8800, help="Server port")
    parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing")
    args = parser.parse_args()
    
    run_substrate_server(port=args.port, multiprocessing=not args.no_multiprocessing)
