
import torch
import os
import json
import time
from rich.console import Console
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

console = Console()

MODEL_PATH = r"H:\models\gemma-3-4b-unq"
MATRICES_DIR = os.path.join(os.getcwd(), "matrices")

# CONFIGURATION
USE_4BIT = False 
# DUAL-LAYER CONFIGURATION
INJECTION_LAYERS = [14, 20]
# Gain Schedule: Gentle Nudge
GAIN_SCHEDULE = {
    14: 0.5,
    20: 0.3 
}
INJECTION_GAIN = 1.0 

class ChimeraDreamer:
    def __init__(self, model_path=MODEL_PATH):
        console.print(f"[cyan]🏥 Initializing Chimera Dreamer (Dual-Layer)...[/]")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        try:
             dtype_to_use = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
             self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=dtype_to_use)
             self.tokenizer = AutoTokenizer.from_pretrained(model_path)
             console.print(f"[green]Prodigy Host Ready on {self.device}.[/]")
        except Exception as e:
            console.print(f"[red]Failed to load host: {e}[/]")

    def get_layer_module(self, layer_idx):
        candidates = []
        candidates.append(self.model)
        if hasattr(self.model, "model"): candidates.append(self.model.model)
        if hasattr(self.model, "model") and hasattr(self.model.model, "language_model"):
            candidates.append(self.model.model.language_model)
            if hasattr(self.model.model.language_model, "model"):
                candidates.append(self.model.model.language_model.model)
        
        for cand in candidates:
            if hasattr(cand, "layers"): return cand.layers[layer_idx]
            if hasattr(cand, "blocks"): return cand.blocks[layer_idx]
            if hasattr(cand, "h"): return cand.h[layer_idx]
        return None

    def compute_native_vector(self, domain_name, seed):
        console.print(f"   [dim]Harvesting Native Vector for {domain_name} (Seed {seed})...[/]")
        captured_vector = None
        def capture_hook(module, inputs, outputs):
            nonlocal captured_vector
            if isinstance(outputs, tuple): h = outputs[0]
            else: h = outputs
            captured_vector = h[0].mean(dim=0).detach() 
            return outputs

        # Harvest from Layer 14 (The Source)
        layer_module = self.get_layer_module(14) 
        if not layer_module: return None

        handle = layer_module.register_forward_hook(capture_hook)
        
        try:
            torch.manual_seed(seed)
            inputs = self.tokenizer("The", return_tensors="pt").to(self.model.device)
            self.model.generate(
                **inputs, 
                max_new_tokens=40, 
                do_sample=True, 
                temperature=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        finally:
            handle.remove()
            
        if captured_vector is not None:
             return captured_vector * INJECTION_GAIN
        return None

    def inject_and_dream(self, domain_name, vector, prompt="The secret of the universe is"):
        if vector is None: return
        
        console.print(f"\n[bold magenta]💉 Injecting {domain_name} Serum (Dual-Layer 14/20)...[/]")
        
        hooks = []
        
        # Factory for closure
        def make_hook(gain):
            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple): h = outputs[0]
                else: h = outputs
                h_norm = h.norm(dim=-1, keepdim=True)
                v_unit = vector / (vector.norm(dim=-1, keepdim=True) + 1e-6)
                injection = v_unit * h_norm * gain
                h = h + injection.to(h.device)
                if isinstance(outputs, tuple): return (h,) + outputs[1:]
                return h
            return hook

        for layer_idx in INJECTION_LAYERS:
            mod = self.get_layer_module(layer_idx)
            if mod:
                gain = GAIN_SCHEDULE.get(layer_idx, 1.0)
                h = mod.register_forward_hook(make_hook(gain))
                hooks.append(h)
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            torch.manual_seed(42) 
            
            out = self.model.generate(
                **inputs, 
                max_new_tokens=60, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            
            style = "cyan"
            if "Quantum" in domain_name: style = "blue"
            if "Physics" in domain_name: style = "yellow"
            if "Python" in domain_name: style = "green"
            
            console.print(Panel(text, title=f"Chimera Dream: {domain_name}", style=style))
            
        except Exception as e:
            console.print(f"[red]Generation failed: {e}[/]")
        finally:
            for h in hooks: h.remove()

    def run_baseline(self, prompt):
        console.print(f"\n[dim]Baseline (No Injection)...[/]")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        torch.manual_seed(42)
        out = self.model.generate(
            **inputs, 
            max_new_tokens=60, 
            do_sample=True, 
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        console.print(Panel(text, title="Baseline", style="dim"))

if __name__ == "__main__":
    dreamer = ChimeraDreamer()
    
    prompt = "The secret of the universe is"
    
    if dreamer.model:
        # 0. Baseline
        dreamer.run_baseline(prompt)
        
        # 1. Native Physics (Seed 2000)
        v_phys = dreamer.compute_native_vector("PHYSICS (Native)", 2000)
        dreamer.inject_and_dream("PHYSICS (Native)", v_phys, prompt)
        
        # 2. Native Python (Seed 4000)
        v_py = dreamer.compute_native_vector("PYTHON (Native)", 4000)
        # dreamer.inject_and_dream("PYTHON (Native)", v_py, "To write perfect code, one must")
        dreamer.inject_and_dream("PYTHON (Native)", v_py, prompt) # Force strict topic steering
