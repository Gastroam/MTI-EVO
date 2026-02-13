
import torch
import torch.nn.functional as F
import sys
import os
import json
import time
from rich.console import Console
from rich.panel import Panel

# Adjust path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from mti_evo.resonance_model import ResonanceGemmaForCausalLM
    from transformers import AutoTokenizer
except:
    print("Error importing resonance model")
    sys.exit(1)

console = Console()
HOST_MODEL_PATH = r"H:\models\gemma-3-12B"
JOURNAL_FILE = "dream_journal_russell_10_resonant.json"

def generate(model, tokenizer, prompt, temp=1.5, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    in_len = inputs.input_ids.shape[1]
    
    generated = inputs.input_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id: break
            
    return tokenizer.decode(generated[0][in_len:], skip_special_tokens=True).strip()

def run_cycle():
    console.print("[bold cyan]🧠 Initializing Resonance Model (Dream Cycle)...[/]")
    model = ResonanceGemmaForCausalLM(HOST_MODEL_PATH)
    model.to("cuda")
    tokenizer = getattr(model.loader, 'tokenizer', None) or AutoTokenizer.from_pretrained(HOST_MODEL_PATH)

    history = []
    
    # Load previous if exists (to resume) or start fresh
    start_cycle = 1
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, 'r') as f:
                history = json.load(f)
                start_cycle = len(history) + 1
        except: pass

    context = "Problem: Russell's Paradox. Contains(R, R) <-> !Contains(R, R)."
    if history:
        context = f"Previous Insight: {history[-1]['synthesis']}. Current Problem: Refine this logic."

    for cycle in range(start_cycle, 11):
        console.print(f"\n[bold magenta]=== CYCLE {cycle}/10 ===[/]")
        
        # 1. DREAM (High Temp, Creative)
        console.print("[dim]Phase 1: Dreaming...[/]")
        # Ensure Paradox Resolver is active
        model.resolve_paradox("This statement is false") 
        
        prompt_dream = f"Context: {context}\nGoal: Radical solution to Russell's Paradox.\nDream:"
        dream = generate(model, tokenizer, prompt_dream, temp=1.5)
        console.print(Panel(dream, title=f"Dream {cycle}", border_style="cyan"))
        
        # 2. CRITIQUE (Low Temp, Analytical)
        console.print("[dim]Phase 2: Critiquing...[/]")
        prompt_critique = f"Proposition: {dream}\nAnalysis: Is this logically consistent? Does it solve the paradox? Critique:"
        critique = generate(model, tokenizer, prompt_critique, temp=0.4) 
        console.print(Panel(critique, title=f"Critique {cycle}", border_style="yellow"))
        
        # 3. SYNTHESIS (Med Temp, Constructive)
        console.print("[dim]Phase 3: Synthesizing...[/]")
        prompt_synthesis = f"Dream: {dream}\nCritique: {critique}\nTask: Formulate a new research direction or axiom.\nSynthesis:"
        synthesis = generate(model, tokenizer, prompt_synthesis, temp=0.8)
        console.print(Panel(synthesis, title=f"Synthesis {cycle}", border_style="green"))
        
        # Update History & Context
        step_data = {
            "cycle": cycle,
            "dream": dream,
            "critique": critique,
            "synthesis": synthesis
        }
        history.append(step_data)
        context = f"Previous Insight: {synthesis}. Current Problem: Refine this logic."
        
        # Save
        with open(JOURNAL_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
        time.sleep(1) 

if __name__ == "__main__":
    run_cycle()
