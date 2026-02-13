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
            from mti_evo.adapters.llm_adapter import LLMAdapter
            self.llm = LLMAdapter(config=self.model_config, auto_load=True)
            print(f"   ✅ VRAM model loaded")
            
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
            if action == 'telepathy' or action == 'infer':
                response_text = self.llm.infer(
                    prompt,
                    max_tokens=request.get('max_tokens', 1024),
                    temperature=request.get('temperature', 0.7)
                )
            else:
                response_text = f"[Resonance: {resonance:.4f}] Processed."
            
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
                if hasattr(self.llm, 'unload_model'):
                    self.llm.unload_model()
            except:
                pass
        
        print("🧠 Inference Process: Shutdown complete.")
