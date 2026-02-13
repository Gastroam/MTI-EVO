
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.quanto import quantize, freeze, qint4, qint8
from rich.console import Console
import time
import gc

console = Console()
MODEL_PATH = r"H:\models\gemma-3-12B"

def test_quanto():
    console.rule("[bold magenta]Testing Optimum Quanto (Native Python Quantization)")
    
    # Load just ONE layer to test quantization flow without OOM
    console.print("[dim]Loading model shell...[/]")
    try:
        # Load empty model structure
        with torch.device("meta"):
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(MODEL_PATH)
            model = AutoModelForCausalLM.from_config(config)
            
        console.print("[green]Shell loaded.[/]")
        
        # Test Quantization on a dummy layer
        console.print("[dim]Creating dummy layer for quantization test...[/]")
        layer = torch.nn.Linear(2048, 2048).to("cuda")
        
        console.print(f"Original size: {layer.weight.numel() * layer.weight.element_size() / 1024**2:.2f} MB")
        
        console.print("[dim]Quantizing to 4-bit (qint4)...[/]")
        quantize(layer, weights=qint4, activations=None)
        freeze(layer)
        
        console.print(f"Quantized size (approx): {layer.weight.numel() * 0.5 / 1024**2:.2f} MB")
        console.print("[green]Quantization successful on dummy![/]")
        
        # Now try to load the REAL model with device_map="auto" and quantize?
        # Note: Quanto usually requires loading full model then quantizing.
        # But we can try loading via accelerate with hooks if supported.
        # For now, let's just confirm the library works.
        
    except Exception as e:
        console.print(f"[red]Quanto Failed:[/] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quanto()
