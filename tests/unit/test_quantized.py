
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import os
from rich.console import Console

console = Console()
MODEL_PATH = r"H:\models\gemma-3-12B"

def test_quantization():
    console.rule("[bold cyan]Testing 4-bit Quantization (CPU Offload)")
    
    # 1. Config for 4-bit NF4 (Standard for high precision 4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    console.print(f"[dim]Loading {MODEL_PATH}...[/]")
    t0 = time.time()
    
    try:
        # device_map="auto" will fill GPU (5.5GB) first, then spill to CPU RAM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        t1 = time.time()
        console.print(f"[green]Loaded in {t1-t0:.2f}s[/]")
        console.print(f"[dim]VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB[/]")
        console.print(f"[dim]Model Memory Footprint: {model.get_memory_footprint()/1e9:.2f}GB[/]")
        
        # Test Inference
        prompts = ["What is 2 + 2?", "Why is the sky blue?"]
        
        for p in prompts:
            console.print(f"\n[bold]User:[/] {p}")
            inputs = tokenizer(p, return_tensors="pt").to(model.device)
            
            t_start = time.time()
            out = model.generate(**inputs, max_new_tokens=50)
            t_end = time.time()
            
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            console.print(f"[bold green]AI:[/] {response}")
            console.print(f"[dim]Time: {t_end - t_start:.2f}s[/]")

    except Exception as e:
        console.print(f"[red]Quantization Failed:[/] {e}")
        # Check if bitsandbytes is missing
        if "No module named 'bitsandbytes'" in str(e):
            console.print("[yellow]Tip: Run 'pip install bitsandbytes'[/]")

if __name__ == "__main__":
    test_quantization()
