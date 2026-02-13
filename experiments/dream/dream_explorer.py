
import sys
import os
import time
from rich.console import Console
from rich.panel import Panel

# Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from mti_evo.llm_adapter import LLMAdapter

console = Console()
MODEL_PATH = r"H:\models\gemma-3-4b-unq"

PROBLEMS = [
    {
        "id": "1_harmonic_primes",
        "title": "The Harmonic Prime Sieve",
        "prompt": "Dream of a musical algorithm where Prime Number gaps dictate melody and rhythm. Describe the mathematical mapping between Number Theory and Audio Frequencies."
    },
    {
        "id": "2_bio_automaton",
        "title": "The Bioluminescent Automaton",
        "prompt": "Dream of a Cellular Automaton (like Game of Life) where cells don't just die, they 'fade' like embers. Define the rules for Birth, Survival, and Bioluminescent Decay."
    },
    {
        "id": "3_semantic_zip",
        "title": "The Semantic Compressor",
        "prompt": "Dream of a compression algorithm that replaces frequent conceptual clusters with dense Unicode symbols (Emojis/Runes) instead of bits. Describe the 'Concept Hashing' logic."
    },
    {
        "id": "4_subjective_economy",
        "title": "The Subjective Barter Loop",
        "prompt": "Dream of an economic simulation with 3 Agents (Red, Green, Blue) who have different internal 'Needs'. Describe a trade algorithm where Value is relative to Need, not potential currency."
    },
    {
        "id": "5_lorenz_cipher",
        "title": "The Lorenz Chaos Cipher",
        "prompt": "Dream of a cryptographic system using the XYZ coordinates of the Lorenz Attractor as a dynamic keystream. Describe how Chaos Theory protects the message."
    }
]

def run_dream_explorer():
    # Force UTF-8 for redirection safety
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except: pass

    console.print(Panel("[bold cyan]Initiating Dream Explorer (5 Cycles)[/]", title="Native Dreamer"))

    try:
        # Initialize Once
        adapter = LLMAdapter(config={
            "model_path": MODEL_PATH,
            "gpu_layers": -1,
            "n_ctx": 4096,
            "temperature": 1.2 # High Creativity
        })

        # for i, problem in enumerate(PROBLEMS, 1):
        problem = PROBLEMS[0] # Test Single Cycle First
        i = 1
        
        print(f"\nCycle {i}: {problem['title']}")
        
        full_prompt = (
            f"You are a Mathematical Visionary.\n"
            f"Task: {problem['prompt']}\n"
            "Output: A vivid, conceptual description of the algorithm and its logic. Do not write full code, just the 'Dream'."
            "Format: Plain text."
        )
        
        print("Dreaming...")
        response = adapter.infer(full_prompt, max_tokens=400)
        
        print(f"--- DREAM START ---\n{response.text}\n--- DREAM END ---")
        
        # console.print(Panel(response.text, title=f"Dream Output: {problem['id']}"))
        
        # Artificial sleep to let the USB/GPU cool or settle if needed
        # time.sleep(2)

    except Exception as e:
        console.print(f"[red]Dreamer Failed: {e}[/]")

if __name__ == "__main__":
    run_dream_explorer()
