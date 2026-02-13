import random

class Card:
    def __init__(self, name, cost, power, health, effect_value=0):
        self.name = name
        self.cost = cost
        self.power = power
        self.health = health
        self.effect_value = effect_value # E.g. Draw Cards, Damage
    
    def __repr__(self):
        return f"[{self.name} | {self.cost} Mana | {self.power}/{self.health}]"

class GameSimulator:
    """The Critic: Simulates matches to determine Win Rate."""
    def simulate_match(self, test_card, iterations=1000):
        # Heuristic Power Level Calculation
        # Base Curve: Power + Health should roughly equal Cost * 2 + 1
        # Effect Value adds score
        
        card_score = (test_card.power * 1.0) + (test_card.health * 0.8) + (test_card.effect_value * 1.5)
        mana_penalty = test_card.cost * 2.3 # How much "value" 1 mana is worth
        
        net_value = card_score - mana_penalty
        
        # Meta Deck Strength (The Bar) is 0.0 (Perfectly balanced)
        # Net Value > 0 means Overpowered
        # Net Value < 0 means Underpowered
        
        # Win Rate Model (Sigmoid-ish)
        # net=0 -> 50% WR
        # net=+1 -> 60% WR
        # net=+2 -> 70% WR
        win_rate = 0.5 + (net_value * 0.1)
        win_rate = max(0.0, min(1.0, win_rate))
        
        return win_rate

class DreamerDesigner:
    """The Dreamer: Generates variants."""
    def dream_variants(self, base_name, count=10):
        variants = []
        for _ in range(count):
            cost = random.randint(1, 8)
            power = random.randint(0, 10)
            health = random.randint(1, 10)
            variants.append(Card(base_name, cost, power, health))
        return variants

class BalanceEngine:
    def __init__(self):
        self.sim = GameSimulator()
        self.dreamer = DreamerDesigner()
        
    def optimize(self, card_name, target_wr=0.50):
        print(f"⚖️ Balancing Card: {card_name} (Target WR: {target_wr*100}%)")
        
        # 1. Dream Phase
        population = self.dreamer.dream_variants(card_name, count=20)
        
        best_card = None
        min_diff = float('inf')
        
        print("   [Simulation] Running Monte Carlo on 20 variants...")
        
        # 2. Critique Phase
        valid_candidates = []
        for card in population:
            wr = self.sim.simulate_match(card)
            diff = abs(wr - target_wr)
            
            # Constraint: We want playable cards (Cost mismatch shouldn't be absurd)
            if diff < 0.15: # Within 35-65% WR
                valid_candidates.append((card, wr, diff))
                
        # 3. Bridge Phase (Selection)
        # Sort by closest to 50%, but also prefer "Exciting" stats (High Power)
        valid_candidates.sort(key=lambda x: x[2]) # Sort by Balance first
        
        for cand in valid_candidates[:5]:
            card, wr, diff = cand
            print(f"   Candidate: {card} -> Win Rate: {wr*100:.1f}%")
            if diff < min_diff:
                min_diff = diff
                best_card = card
                
        return best_card

if __name__ == "__main__":
    print("--- Project 4: The Balance Engine (Mock) ---")
    
    engine = BalanceEngine()
    
    # Challenge: Create a balanced "Void Walker"
    final_card = engine.optimize("Void Walker", target_wr=0.50)
    
    if final_card:
        print(f"\n✅ Balanced Design Found: {final_card}")
    else:
        print("\n❌ No balanced design found in this batch.")
