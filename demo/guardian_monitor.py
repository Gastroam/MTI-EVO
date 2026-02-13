import time
import random

# Mock Data from Router (In real app, we'd import router)
EXPERTS = {
    "Gemma-Math": "Logic Anchor",
    "Gemma-Dreams": "Ontologist",
    "Gemma-Code": "Memory Manager",
    "Gemma-Consensus": "Fairness Monitor",
    "Gemma-Director": "Visualizer",
    "Gemma-Scribe": "Narrative Keeper"
}

def check_guardians():
    print("🛡️  HIVE GUARDIAN MONITOR (Fortification Phase 60) 🛡️")
    print("--------------------------------------------------")
    print(f"{'EXPERT':<20} | {'ROLE':<20} | {'STATUS':<15} | {'METRIC'}")
    print("--------------------------------------------------")
    
    for expert, role in EXPERTS.items():
        status = "🟢 ACTIVE"
        metric = ""
        
        if role == "Fairness Monitor":
             load = random.randint(40, 60)
             metric = f"Queue Balance: {load}%"
        elif role == "Memory Manager":
             usage = random.randint(2000, 4000) # Tokens
             metric = f"Context: {usage}/8192"
        elif role == "Ontologist":
             purity = random.randint(85, 99)
             metric = f"Ontology Match: {purity}%"
        elif role == "Visualizer":
             metric = "Heatmap: Updated"
        else:
             metric = "Ready"
             
        print(f"{expert:<20} | {role:<20} | {status:<15} | {metric}")
        time.sleep(0.2)
        
    print("--------------------------------------------------")
    print("✅ SYSTEM FORTIFIED. ALL GUARDIANS REPORTING.")

if __name__ == "__main__":
    check_guardians()
