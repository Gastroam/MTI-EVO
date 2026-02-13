import time
from .base import BaseEngine, LLMResponse

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

class NativeEngine(BaseEngine):
    """Engine for Native (Safetensors) models via Transformers."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.hf_model = None
        self.tokenizer = None
        self.backend_name = "native"
        if not HAS_TRANSFORMERS:
            # Silent initialization, only warn if load attempted
            pass

    def load_model(self):
        import os
        if not HAS_TRANSFORMERS: 
            print("[NativeEngine] ❌ Transformers/Torch not installed. Cannot load model.")
            return
        
        # [FIX] Robust Path Resolution
        local_path = os.path.abspath(self.model_path)
        if not os.path.isdir(local_path):
             print(f"[NativeEngine] ❌ Model directory not found: {local_path}")
             return
             
        print(f"[NativeEngine] 🤗 Loading: {local_path}")
        
        try:

            # Dynamic Quantization Config
            quant_mode = self.config.get("quantization", "none")
            quant_config = None
            
            if quant_mode == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                print("   [dim]Optimization: 4-bit NF4 Enabled via Config[/]")
            elif quant_mode == "8bit":
                 quant_config = BitsAndBytesConfig(load_in_8bit=True)
                 print("   [dim]Optimization: 8-bit Enabled via Config[/]")

            # Attention Implementation
            attn_impl = "eager"
            if self.config.get("flash_attention", False):
                try:
                    import flash_attn
                    attn_impl = "flash_attention_2"
                except ImportError:
                    print("[NativeEngine] ⚠️ Flash Attention requested but not installed. Falling back to eager.")

            if quant_config:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    local_path, 
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                    local_files_only=True,
                    attn_implementation=attn_impl
                )
            else:
                 self.hf_model = AutoModelForCausalLM.from_pretrained(
                     local_path,
                     device_map="auto",
                     torch_dtype=torch.float16,
                     trust_remote_code=True,
                     local_files_only=True,
                     attn_implementation=attn_impl
                 )
            
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            
            # [DEBUG] Device Map Reporting to detect CPU offloading
            if hasattr(self.hf_model, "hf_device_map"):
                device_counts = {}
                for layer, device in self.hf_model.hf_device_map.items():
                    device_str = str(device)
                    device_counts[device_str] = device_counts.get(device_str, 0) + 1
                
                print(f"[NativeEngine] 🗺️ Layer distribution: {device_counts}")
                if any("cpu" in d for d in device_counts):
                    print(f"[NativeEngine] ⚠️ WARNING: CPU OFFLOADING DETECTED! This causes extreme slowness.")
            
            print(f"[NativeEngine] ✅ Loaded on {self.hf_model.device}")
            
        except Exception as e:
            print(f"[NativeEngine] ❌ Load Failed: {e}")

    def infer(self, prompt: str, max_tokens: int = 1024, stop: list = None, **kwargs) -> LLMResponse:
        t0 = time.perf_counter()
        if not self.hf_model:
             return LLMResponse("Native Engine Not Loaded", 0, 0, 0.0)

        # [PHASE 64] Temperature & Stop handling
        eff_temp = kwargs.get("temperature", self.temperature)
        do_sample = True if eff_temp > 0 else False

        try:
            # Templating with Fallback
            input_ids = None
            try:
                messages = [{"role": "user", "content": prompt}]
                tokens = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors="pt"
                )
            except:
                # Fallback to raw prompt
                tokens = self.tokenizer(prompt, return_tensors="pt")

            # Robust Tensor Movement (handles BatchEncoding or Tensors)
            device = self.hf_model.device
            if hasattr(tokens, "to"):
                tokens = tokens.to(device)
            elif isinstance(tokens, dict):
                tokens = {k: v.to(device) for k, v in tokens.items()}
            
            # Handle input_ids retrieval
            if isinstance(tokens, torch.Tensor):
                input_ids = tokens
            else:
                input_ids = tokens["input_ids"]

            # [FIX] Explicit Attention Mask to satisfy Transformers warning
            attention_mask = torch.ones_like(input_ids)

            print(f"[NativeEngine] 🚀 Generating with input shape: {input_ids.shape} (temp={eff_temp})")
            
            with torch.no_grad():
                # Prevent VRAM fragmentation
                torch.cuda.empty_cache()
                
                print("   [dim]Stage: Starting hf_model.generate...[/]")
                out = self.hf_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=eff_temp,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_k=kwargs.get("top_k", 40)
                )
                print("   [dim]Stage: Generation complete.[/]")
            
            new_tokens = out[0][input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            latency = (time.perf_counter() - t0) * 1000
            
            return LLMResponse(text, len(out[0]), latency, 0.98)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return LLMResponse(f"Error: {e}", 0, 0, 0.0)

    def embed(self, text: str):
        if self.hf_model and HAS_TRANSFORMERS:
            # [MEMORY SAFETY]
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to(self.hf_model.device)
                outputs = self.hf_model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                vec = hidden.mean(dim=1).squeeze().tolist()
                return vec
        return []

    def unload(self):
        if self.hf_model:
            print("[NativeEngine] 🗑️ Offloading model from VRAM...")
            # Do NOT move to CPU (can cause System RAM OOM). Just delete.
            del self.hf_model
            del self.tokenizer
            self.hf_model = None
            self.tokenizer = None
            
            # Aggressive GC
            import gc
            gc.collect()
            if HAS_TRANSFORMERS:
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except:
                    pass
            print("[NativeEngine] ✅ RAM Released.")
