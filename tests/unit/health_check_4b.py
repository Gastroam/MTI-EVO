
import sys
import os
import json
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.append(os.path.abspath("d:/VMTIDE/MTI-EVO/src"))
from mti_evo.llm_adapter import LLMAdapter

console = Console()

def health_check():
    model_path = r"D:\VMTIDE\MTI-EVO\models\gemma-3-4b-it-q4_0.gguf"
    
    console.print(Panel(f"[bold cyan]🏥 GEMMA 4B HEALTH CHECK[/]\nPath: {model_path}", style="cyan"))
    
    # 1. File Integrity (Basic)
    if not os.path.exists(model_path):
        console.print("[red bold]❌ CRITICAL: Model file missing![/]")
        return
        
    size = os.path.getsize(model_path)
    size_gb = size / (1024**3)
    console.print(f"[dim]File Size: {size_gb:.2f} GB[/]")
    
    if size < 1000: # < 1KB
        console.print("[red bold]❌ CRITICAL: Model file is empty or corrupted![/]")
        return

    # 2. Test Configuration
    config = {
        "model_path": model_path,
        "n_ctx": 2048,
        "gpu_layers": -1,
        "temperature": 0.0, # Deterministic for health check
        "model_type": "auto" # Explicitly NOT quantum
    }
    
    # 3. Load Model
    console.print("\n[yellow]1. Loading Model (Llama-cpp)...[/]")
    try:
        adapter = LLMAdapter(config=config)
        if adapter.backend != "llama":
            console.print(f"[red]❌ Backend Mismatch. Expected 'llama', got '{adapter.backend}'[/]")
            # Proceed anyway if it fell back to something else, but report it
    except Exception as e:
        console.print(f"[red]❌ Load Exception: {e}[/]")
        return

    # 4. Inference Tests
    test_cases = [
        {
            "name": "Basic Greeting",
            "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello, are you functional?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "expect_any": ["Yes", "functional", "I am", "Hello"]
        },
        {
            "name": "Math Logic (Encoding Check)",
            "prompt": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nCalculate 2 + 2.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
            "expect_any": ["4", "four"]
        }
    ]
    
    console.print("\n[yellow]2. Running Cognitive Diagnostics...[/]")
    
    passed = 0
    for test in test_cases:
        console.print(f"   Testing: [bold]{test['name']}[/]...")
        try:
            res = adapter.infer(test['prompt'], max_tokens=64)
            text = res.text.strip()
            
            # Check for gibberish (high ascii ratio or unreadable)
            # Simple check: does it contain expected words?
            valid = any(kw.lower() in text.lower() for kw in test['expect_any'])
            
            if valid:
                console.print(f"     [green]✅ PASS[/] Output: '{text}'")
                passed += 1
            else:
                console.print(f"     [red]❌ FAIL[/] Output: '{text}'")
                console.print(f"     [dim]Expected one of: {test['expect_any']}[/]")
                
        except Exception as e:
            console.print(f"     [red]❌ ERROR: {e}[/]")

    # 5. Summary
    if passed == len(test_cases):
        console.print("\n[bold green]✅ HEALTH CHECK PASSED: Gemma 4B is healthy.[/]")
    else:
        console.print("\n[bold red]⚠️ HEALTH CHECK FAILED: Re-download recommended.[/]")

if __name__ == "__main__":
    health_check()
