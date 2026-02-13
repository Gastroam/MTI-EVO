import torch
import torch.nn.functional as F
import sys
import os
import json
from rich.console import Console
from rich.panel import Panel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from mti_evo.resonance_model import ResonanceGemmaForCausalLM
    from transformers import AutoTokenizer
except: pass

console = Console()
HOST_MODEL_PATH = r"H:\models\gemma-3-12B"
JOURNAL_FILE = "dream_journal_russell.json"

def generate(model, tokenizer, prompt, temp=1.5):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    in_len = inputs.input_ids.shape[1]
    
    generated = inputs.input_ids
    with torch.no_grad():
        for _ in range(120): # Short burst
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break
            
    return tokenizer.decode(generated[0][in_len:], skip_special_tokens=True).strip()

def run_single_step():
    # Load previous state
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
        
    cycle = len(history) + 1
    
    # Context from previous synthesis
    if history:
        last = history[-1]
        context = f"Previous Synthesis: {last['synthesis']}"
    else:
        context = "Problem: Russell's Paradox. Contains(R, R) <-> !Contains(R, R)."

    console.print(f"[bold cyan]🌙 DREAM STEP {cycle}[/]")
    console.print(f"[dim]Context: {context}[/]")
    
    # Init Model
    model = ResonanceGemmaForCausalLM(HOST_MODEL_PATH)
    model.to("cuda")
    
    # Paradox Resolver Active (Grounding)
    model.resolve_paradox("This statement is false")
    
    prompt = f"Context: {context}\nGoal: Radical/Novel Solution.\nDream:"
    dream_text = generate(model, model.loader.tokenizer if hasattr(model.loader, 'tokenizer') else AutoTokenizer.from_pretrained(HOST_MODEL_PATH), prompt)
    
    console.print(Panel(dream_text, title="Generated Dream", border_style="cyan"))
    
    # Save TEMPORARY object
    step_data = {
        "cycle": cycle,
        "dream": dream_text,
        "critique": "PENDING_AGENT_REVIEW",
        "synthesis": "PENDING_AGENT_REVIEW"
    }
    
    history.append(step_data)
    with open(JOURNAL_FILE, 'w') as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    run_single_step()
