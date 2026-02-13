
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rich.console import Console
from rich.panel import Panel

console = Console()

def health_check_tensors():
    # Target the UNQUANTIZED (Safetensors) directory
    model_path = r"D:\VMTIDE\MTI-EVO\models\gemma-3-4b-unq"
    
    console.print(Panel(f"[bold magenta]🏥 GEMMA 4B SAFETENSORS CHECK[/]\nPath: {model_path}", style="magenta"))
    
    # 1. File Integrity
    required_files = ["model.safetensors.index.json", "tokenizer.json", "config.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing:
        console.print(f"[red bold]❌ CRITICAL: Missing files: {missing}[/]")
        return
    else:
        console.print("[green]✅ Structural files present.[/]")

    # 2. Initialization
    console.print("\n[yellow]1. Loading Model (Transformers 4-bit)...[/]")
    try:
        # 4-bit loading to fit on 3070 alongside OS overhead
        quant_opts = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_opts,
            device_map="auto"
        )
        console.print("[green]✅ Model Loaded on GPU.[/]")
        
    except Exception as e:
        console.print(f"[red]❌ Load Failed: {e}[/]")
        return

    # 3. Inference Test (TEMPLATED)
    console.print("\n[yellow]2. Running Cognitive Diagnostics (With Chat Template)...[/]")
    
    test_prompts = [
        [{"role": "user", "content": "Identify yourself briefly."}],
        [{"role": "user", "content": "Calculate 2 + 2. Return only the number."}]
    ]
    
    for messages in test_prompts:
        try:
            # Apply Chat Template (Fixes repetition/eos issues)
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64,
                do_sample=False, # Deterministic
                temperature=None,
                top_p=None
            )
            
            # Decode only the new tokens
            response_tokens = outputs[0][inputs.input_ids.shape[1]:]
            text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            console.print(f"   [bold]Input:[/] {messages[0]['content']}")
            console.print(f"   [dim]Output:[/] '{text.strip()}'")
            
            if text.strip():
                console.print("   [green]✅ PASS[/]")
            else:
                 console.print("   [red]⚠️  Empty Response[/]")

        except Exception as e:
            console.print(f"   [red]❌ Inference Error: {e}[/]")

    console.print("\n[bold magenta]Diagnostics Complete.[/]")

if __name__ == "__main__":
    health_check_tensors()
