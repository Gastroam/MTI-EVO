"""
Hybrid Engine (Local + API Fusion)
===================================
Combines fast local inference with cloud API reasoning.

Architecture:
- Local (GGUF/Native): Fast, private, offline-capable
- API (Cloud): Deep reasoning escalation (Claude, GPT, Gemini)

Use Cases:
- Privacy-first: Local handles sensitive data, API handles reasoning
- Cost-optimized: Local for simple queries, API for complex ones
- Low-latency: Local for real-time, API for deep analysis
"""
import time
import os
from typing import List, Optional, Dict, Any
from .base import BaseEngine, LLMResponse


class HybridEngine(BaseEngine):
    """
    Dual-mode engine combining local model with cloud API.
    
    Config Keys:
    - model_path: Path to local GGUF model
    - api_provider: 'openai', 'anthropic', 'google', 'ollama' (default: 'openai')
    - api_model: Model name for API (e.g., 'gpt-4o', 'claude-3-opus')
    - api_key: API key (or set via environment variable)
    - mode: 'local_first', 'api_first', 'parallel', 'escalate' (default: 'local_first')
    - escalation_threshold: Confidence threshold to escalate to API (default: 0.5)
    """
    
    PROVIDERS = {
        'openai': {'env_key': 'OPENAI_API_KEY', 'base_url': 'https://api.openai.com/v1'},
        'anthropic': {'env_key': 'ANTHROPIC_API_KEY', 'base_url': 'https://api.anthropic.com'},
        'google': {'env_key': 'GOOGLE_API_KEY', 'base_url': 'https://generativelanguage.googleapis.com'},
        'ollama': {'env_key': None, 'base_url': 'http://localhost:11434'},
    }
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.backend_name = "hybrid"
        
        # Local engine config
        self.local_engine = None
        self.local_model_path = config.get("model_path", "")
        
        # API config
        self.api_provider = config.get("api_provider", "openai")
        self.api_model = config.get("api_model", "gpt-4o-mini")
        self.api_key = config.get("api_key", "")
        self.api_base_url = config.get("api_base_url", "")
        
        # Hybrid mode
        self.mode = config.get("mode", "local_first")
        self.escalation_threshold = config.get("escalation_threshold", 0.5)
        
        # Stats
        self.stats = {"local_calls": 0, "api_calls": 0, "escalations": 0}

    def load_model(self):
        print(f"[HybridEngine] 🔀 Loading Hybrid Engine...")
        print(f"   Mode: {self.mode}")
        
        # 1. Load Local Engine
        if self.local_model_path:
            try:
                from .gguf_engine import GGUFEngine
                self.local_engine = GGUFEngine({
                    "model_path": self.local_model_path,
                    "n_ctx": self.config.get("local_n_ctx", self.config.get("n_ctx", 4096)),
                    "gpu_layers": self.config.get("local_gpu_layers", self.config.get("gpu_layers", -1)),
                })
                self.local_engine.load_model()
                print(f"   ✅ Local: {self.local_model_path}")
            except Exception as e:
                print(f"   ⚠️ Local failed: {e}")
        
        # 2. Setup API
        if not self.api_key:
            provider_info = self.PROVIDERS.get(self.api_provider, {})
            env_key = provider_info.get('env_key')
            if env_key:
                self.api_key = os.environ.get(env_key, "")
        
        if self.api_key or self.api_provider == "ollama":
            print(f"   ✅ API: {self.api_provider}/{self.api_model}")
        else:
            print(f"   ⚠️ API: No key found for {self.api_provider}")

    def _call_api(self, prompt: str, max_tokens: int = 512) -> Optional[str]:
        """Call cloud API provider."""
        try:
            import requests
            
            if self.api_provider == "openai":
                url = f"{self.api_base_url or 'https://api.openai.com/v1'}/chat/completions"
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                data = {
                    "model": self.api_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": self.temperature
                }
                r = requests.post(url, headers=headers, json=data, timeout=60)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            
            elif self.api_provider == "anthropic":
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.api_model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
                r = requests.post(url, headers=headers, json=data, timeout=60)
                r.raise_for_status()
                return r.json()["content"][0]["text"]
            
            elif self.api_provider == "ollama":
                url = f"{self.api_base_url or 'http://localhost:11434'}/api/generate"
                data = {"model": self.api_model, "prompt": prompt, "stream": False}
                r = requests.post(url, json=data, timeout=60)
                r.raise_for_status()
                return r.json()["response"]
            
            elif self.api_provider == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.api_model}:generateContent?key={self.api_key}"
                data = {"contents": [{"parts": [{"text": prompt}]}]}
                r = requests.post(url, json=data, timeout=60)
                r.raise_for_status()
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            else:
                return None
                
        except Exception as e:
            print(f"[HybridEngine] ⚠️ API error: {e}")
            return None

    def _should_escalate(self, local_response: LLMResponse, prompt: str) -> bool:
        """Determine if query should escalate to API."""
        # Simple heuristics - can be enhanced with classifier
        complexity_indicators = [
            "explain", "analyze", "compare", "why", "how", "what is",
            "step by step", "in detail", "comprehensive", "quantum", "science",
            "art", "function", "write", "generate", "list"
        ]
        is_complex = any(ind in prompt.lower() for ind in complexity_indicators)
        low_confidence = local_response.coherence < self.escalation_threshold
        
        print(f"[Hybrid] Escalation check: Complex={is_complex} (Triggered by: {[ind for ind in complexity_indicators if ind in prompt.lower()]}), LowConf={low_confidence}")
        
        return is_complex or low_confidence

    def infer(self, prompt: str, max_tokens: int = 512, stop: list = None, **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        
        local_result = None
        api_result = None
        source = "unknown"
        
        # === MODE: LOCAL FIRST ===
        if self.mode == "local_first":
            if self.local_engine:
                local_result = self.local_engine.infer(prompt, max_tokens=max_tokens, stop=stop)
                self.stats["local_calls"] += 1
                
                if self._should_escalate(local_result, prompt):
                    api_text = self._call_api(prompt, max_tokens)
                    if api_text:
                        self.stats["api_calls"] += 1
                        self.stats["escalations"] += 1
                        source = "escalated"
                        output = f"[Local] {local_result.text}\n\n[API:{self.api_provider}] {api_text}"
                    else:
                        source = "local"
                        output = local_result.text
                else:
                    source = "local"
                    output = local_result.text
            else:
                api_text = self._call_api(prompt, max_tokens)
                self.stats["api_calls"] += 1
                source = "api_only"
                output = api_text or "No response"
        
        # === MODE: API FIRST ===
        elif self.mode == "api_first":
            api_text = self._call_api(prompt, max_tokens)
            if api_text:
                self.stats["api_calls"] += 1
                source = "api"
                output = api_text
            elif self.local_engine:
                local_result = self.local_engine.infer(prompt, max_tokens=max_tokens, stop=stop)
                self.stats["local_calls"] += 1
                source = "local_fallback"
                output = local_result.text
            else:
                output = "No engines available"
                source = "none"
        
        # === MODE: PARALLEL ===
        elif self.mode == "parallel":
            local_text = ""
            api_text = ""
            
            if self.local_engine:
                local_result = self.local_engine.infer(prompt, max_tokens=max_tokens // 2, stop=stop)
                local_text = local_result.text
                self.stats["local_calls"] += 1
            
            api_text = self._call_api(prompt, max_tokens) or ""
            if api_text:
                self.stats["api_calls"] += 1
            
            source = "parallel"
            output = f"[Local] {local_text}\n\n[API:{self.api_provider}] {api_text}"
        
        # === MODE: ESCALATE ===
        elif self.mode == "escalate":
            # Local handles simple, API handles complex
            complexity_indicators = ["explain", "analyze", "why", "how", "compare", "detail"]
            is_complex = any(ind in prompt.lower() for ind in complexity_indicators)
            
            if is_complex and self.api_key:
                api_text = self._call_api(prompt, max_tokens)
                self.stats["api_calls"] += 1
                source = "api_escalated"
                output = api_text or "API unavailable"
            elif self.local_engine:
                local_result = self.local_engine.infer(prompt, max_tokens=max_tokens, stop=stop)
                self.stats["local_calls"] += 1
                source = "local"
                output = local_result.text
            else:
                output = "No engine available"
                source = "none"
        
        else:
            output = f"Unknown mode: {self.mode}"
            source = "error"
        
        latency = (time.perf_counter() - t0) * 1000
        
        return LLMResponse(
            text=output,
            tokens=len(prompt.split()),
            latency_ms=latency,
            coherence=0.9 if source.startswith("api") else 0.7,
            gpu_stats={"source": source, "mode": self.mode, "stats": self.stats}
        )

    def embed(self, text: str) -> List[float]:
        """Use local engine for embeddings (faster, private)."""
        if self.local_engine:
            return self.local_engine.embed(text)
        return []

    def unload(self):
        if self.local_engine:
            self.local_engine.unload()
        print(f"[HybridEngine] ✅ Unloaded. Stats: {self.stats}")

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return self.stats
