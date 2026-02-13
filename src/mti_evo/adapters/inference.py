"""
MTI-EVO Inference Process
==========================
Single process holding VRAM model + updating shared substrate.

Ontology: The inference process is the "metabolic heart" that animates
the substrate. HTTP workers inhabit the substrate; this process gives it life.
"""
import os
import sys
import time
from multiprocessing import Process, Queue
from typing import Optional


class InferenceProcess(Process):
    """
    Single process that:
    1. Holds the VRAM model (ONLY process with CUDA context)
    2. Receives inference requests via queue
    3. Updates the shared mmap substrate
    """
    
    def __init__(self, 
                 request_queue: Queue, 
                 response_queue: Queue,
                 mmap_path: str,
                 model_config: dict,
                 dim: int = 64):
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.mmap_path = mmap_path
        self.model_config = model_config
        self.dim = dim
        self.broca = None
        self.llm = None
        self.running = True
    
    def run(self):
        """Main loop - runs in separate process."""
        print(f"🧠 Inference Process STARTING (PID {os.getpid()})...")
        
        try:
            # 1. Load shared substrate (mmap)
            # Use SubstrateRuntime as Composition Root (Option A)
            from mti_evo.runtime.substrate_runtime import SubstrateRuntime
            
            # This runtime instance is the WRITER (read_only=False)
            self.runtime = SubstrateRuntime(config=self.model_config, read_only=False, persistence_id="substrate")
            # Note: Startup might be slightly slower due to full init, but it guarantees consistency.
            
            # Alias for convenience
            self.broca = self.runtime.broca
            self.hippocampus = self.runtime.hippocampus
            
            print(f"   ✅ Substrate Runtime initialized (Writer Mode)")
            
            # 2. Load VRAM model (ONLY in this process)
            # Use Engine Registry to load appropriate engine
            from mti_evo.engines.registry import EngineRegistry, discover_engines
            
            # Ensure registry is populated (including plugins)
            discover_engines()
            
            # Determine engine type from config
            model_type = self.model_config.get("model_type", "auto")
            
            # Heuristic for Auto if needed
            if model_type == "auto":
                 p = self.model_config.get("model_path", "")
                 if p.endswith(".gguf"): model_type = "gguf"
                 elif p.endswith(".safetensors"): model_type = "native"
                 else: model_type = "gguf"

            try:
                self.llm = EngineRegistry.create(model_type, self.model_config)
                # EngineProtocol.load(config)
                self.llm.load(self.model_config)
                print(f"   ✅ Engine Loaded: {model_type}")
            except Exception as e:
                print(f"   ❌ Engine Load Failed: {e}")
                self.llm = None
            
            print(f"🧠 Inference Process READY (PID {os.getpid()})")
            
            # 3. Process requests
            while self.running:
                try:
                    # Non-blocking get with timeout
                    request = self.request_queue.get(timeout=0.1)
                    
                    if request is None:  # Poison pill
                        print("🧠 Inference Process: Poison pill received, shutting down...")
                        break
                    
                    self._handle_request(request)
                    
                except Exception as e:
                    if "Empty" not in str(type(e).__name__):
                        print(f"⚠️ Inference Process error: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Inference Process failed to start: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _handle_request(self, request: dict):
        """Process a single inference request."""
        start_time = time.time()
        
        try:
            prompt = request.get('prompt', '')
            action = request.get('action', 'telepathy')
            
            # 1. Resonance calculation (writes to mmap substrate via Runtime/Broca)
            if self.broca:
                resonance_result = self.broca.process_thought(prompt, learn=True)
                resonance = resonance_result.get('resonance', 0.0) if isinstance(resonance_result, dict) else float(resonance_result)
            else:
                resonance = 0.0
            
            # 2. Flush mmap for cross-process visibility (Windows fix)
            self._ensure_coherency()
            
            # 3. Inference with resonance context
            if self.llm:
                # EngineProtocol: infer(prompt, **kwargs) -> EngineResult
                res = self.llm.infer(
                    prompt=prompt,
                    max_tokens=request.get('max_tokens', 1024),
                    temperature=request.get('temperature', 0.7),
                    stop=request.get('stop', ["<end_of_turn>"])
                )
                
                response_text = res.text
                # metrics = res.metrics
            else:
                response_text = "[Error: No Brain Loaded]"
            
            # 4. Return result
            latency_ms = (time.time() - start_time) * 1000
            self.response_queue.put({
                'request_id': request.get('id', 'unknown'),
                'response': response_text,
                'resonance': resonance,
                'latency_ms': latency_ms,
                'success': True
            })
            
        except Exception as e:
            self.response_queue.put({
                'request_id': request.get('id', 'unknown'),
                'error': str(e),
                'success': False
            })
    
    def _ensure_coherency(self):
        """
        Windows-specific: Force mmap flush visible to other processes.
        """
        if sys.platform == 'win32':
            try:
                # Flush hippocampus mmap
                if hasattr(self.hippocampus, 'flush'):
                    self.hippocampus.flush()
            except Exception as e:
                pass  # Non-critical, coherency will happen eventually
    
    def _cleanup(self):
        """Graceful shutdown."""
        print("🧠 Inference Process: Cleaning up...")
        
        # Save substrate state
        if self.broca and hasattr(self.broca, 'sleep'):
            try:
                self.broca.sleep() 
            except:
                pass
        
        # Unload model
        if self.llm:
            try:
                if hasattr(self.llm, 'unload'):
                    self.llm.unload()
            except:
                pass
        
        print("🧠 Inference Process: Shutdown complete.")
