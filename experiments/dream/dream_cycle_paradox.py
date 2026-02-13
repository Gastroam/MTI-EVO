import torch
import torch.nn.functional as F
import sys
import os
import json
import time
from rich.console import Console
from rich.panel import Panel

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from mti_evo.resonance_model import ResonanceGemmaForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    pass

console = Console()
HOST_MODEL_PATH = r"H:\models\gemma-3-12B"
JOURNAL_FILE = "dream_journal_russell.json"

def generate(model, tokenizer, prompt, temp=1.0, max_new=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    in_len = inputs.input_ids.shape[1]
    
    # Simple generation loop with manual temperature control
    generated = inputs.input_ids
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(generated)
            next_token_logits = logits[:, -1, :]
            
            # Temp Scaling
            if temp < 1e-4: # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temp, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    return tokenizer.decode(generated[0][in_len:], skip_special_tokens=True).strip()

def run_dream_cycle():
    console.print(Panel("[bold cyan]🌙 DREAM CYCLE: RUSSELL'S PARADOX[/]", subtitle="10 Iterations"))
    
    # 1. Init
    tokenizer = AutoTokenizer.from_pretrained(HOST_MODEL_PATH)
    model = ResonanceGemmaForCausalLM(HOST_MODEL_PATH)
    model.to("cuda")
    
    # Activate Resolver Permanently for this session (Safety Net)
    # model.resolver.manifolds["SELF_REFERENCE"].activate() # Logic TBD
    # We will rely on calling 'resolve_paradox' before Dream step.
    
    history = []
    
    # Initial Context
    context = "We must solve Russell's Paradox: Contains(R, R) iff not Contains(R, R)."
    
    for cycle in range(1, 11): # 10 Steps
        console.print(f"\n[bold white on blue] CYCLE {cycle} [/]")
        
        # --- STEP 1: DREAM (Temp 1.5) ---
        console.print("[dim]Phase 1: Dream (High Entropy)...[/]")
        model.resolve_paradox("This statement is false") # Prime the jammer
        
        dream_prompt = f"Context: {context}\nMission: Propose a radical, new mathematical framework to solve this. Ignore tradition.\nSolution:"
        dream = generate(model, tokenizer, dream_prompt, temp=1.5, max_new=80)
        console.print(Panel(dream, title="Dream", border_style="cyan"))
        
        # --- STEP 2: CRITIQUE (Temp 0.2) ---
        console.print("[dim]Phase 2: Critique (Grounding)...[/]")
        crit_prompt = f"Proposal: {dream}\nTask: Criticize this proposal logically. Identify contradictions or nonsense.\nCritique:"
        critique = generate(model, tokenizer, crit_prompt, temp=0.2, max_new=60)
        console.print(Panel(critique, title="Critique", border_style="red"))
        
        # --- STEP 3: SYNTHESIS (Temp 0.8) ---
        console.print("[dim]Phase 3: Synthesis (Evolution)...[/]")
        syn_prompt = f"Dream: {dream}\nCritique: {critique}\nTask: Synthesize a refined axiom. Discard nonsense.\nAxiom:"
        synthesis = generate(model, tokenizer, syn_prompt, temp=0.8, max_new=60)
        console.print(Panel(synthesis, title="Synthesis", border_style="green"))
        
        # Save Entry
        entry = {
            "cycle": cycle,
            "dream": dream,
            "critique": critique,
            "synthesis": synthesis
        }
        history.append(entry)
        
        # Evolve Context (The "Pulse")
        context = f"Previous Finding: {synthesis}"
        
        # Write Journal Partial
        with open(JOURNAL_FILE, 'w') as f:
            json.dump(history, f, indent=2)
            
    console.print(f"\n[bold green]✅ Dream Cycle Complete. Saved to {JOURNAL_FILE}[/]")

if __name__ == "__main__":
    run_dream_cycle()
