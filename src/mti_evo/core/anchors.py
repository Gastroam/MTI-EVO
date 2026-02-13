
import json
import os
import numpy as np
import time
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field

from mti_evo.core.logger import get_logger

logger = get_logger("SemanticAnchors")

@dataclass
class SemanticAnchor:
    seed: int
    pattern: List[float]
    label: str = ""              # Human readable ID
    tags: List[str] = field(default_factory=list)
    target_pre: float = 0.10     # Expected resonance BEFORE reinforcement
    target_post: float = 0.85    # Expected resonance AFTER reinforcement
    reinforce_steps: int = 3     # How many times to pulse
    cooldown_steps: int = 100    # Minimum steps between reinforcements
    pinned: bool = True          # If True, verified as Pinned Seed (cannot be evicted)
    last_reinforced: int = -1    # Step count of last reinforcement (persisted if possible, or runtime)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "pattern": self.pattern,
            "label": self.label,
            "tags": self.tags,
            "target_pre": self.target_pre,
            "target_post": self.target_post,
            "reinforce_steps": self.reinforce_steps,
            "cooldown_steps": self.cooldown_steps,
            "pinned": self.pinned
        }

class SemanticAnchorManager:
    """
    Manages a set of "Anchor" concepts that must not drift.
    Used to ground the latent space by periodically reinforcing these patterns.
    """
    def __init__(self, config=None):
        self.config = config
        self.anchors: Dict[int, SemanticAnchor] = {}
        # We track reinforcement globally or per anchor. Spec says per anchor cooldown.
        
        # Load if config provided
        if self.config and self.config.anchor_file:
            self.load_from_file(self.config.anchor_file)
            
        self.reinforcing = False # Recursion Guard
            
    def register_anchor(self, anchor: SemanticAnchor):
        """
        Register a new anchor.
        """
        self.anchors[anchor.seed] = anchor
        
        # Orthogonal Pinning: Only pin if explicitly requested
        if anchor.pinned and self.config:
            if not hasattr(self.config, "pinned_seeds"):
                 self.config.pinned_seeds = set()
            self.config.pinned_seeds.add(anchor.seed)
            
        logger.debug(f"Registered anchor {anchor.seed} tags={anchor.tags} pinned={anchor.pinned}")

    def load_from_file(self, path: str):
        """
        Load anchors from a JSON file.
        """
        if not os.path.exists(path):
            logger.warning(f"Anchor file not found: {path}")
            return
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Sub-key "anchors" or root list? User spec says: {"anchors": [...]}
            # But handle list for backward compat with my prev implementation if needed.
            if isinstance(data, dict) and "anchors" in data:
                items = data["anchors"]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            count = 0
            for item in items:
                # Handle simplified format or full spec
                anchor = SemanticAnchor(
                    seed=item['seed'],
                    pattern=item['pattern'] if 'pattern' in item else item.get('vector', []), # Support 'vector' alias
                    label=item.get('label', ""),
                    tags=item.get('tags', []),
                    target_pre=item.get('target_pre', 0.10),
                    target_post=item.get('target_post', 0.85),
                    reinforce_steps=item.get('reinforce_steps', 3),
                    cooldown_steps=item.get('cooldown_steps', 100),
                    pinned=item.get('pinned', True) # Default to True for safety unless specified
                )
                self.register_anchor(anchor)
                count += 1
                
            logger.info(f"Loaded {count} anchors from {path}")
        except Exception as e:
            logger.error(f"Failed to load anchors from {path}: {e}")

    def check_and_reinforce(self, lattice, current_step: int):
        """
        Checks if global reinforcement cycle is due, then reinforces candidate anchors.
        """
        if self.reinforcing:
            return

        if not self.config:
            return
            
        freq = getattr(self.config, "anchor_reinforcement_freq", 0)
        if freq <= 0:
            return

        if current_step % freq == 0:
            self.reinforcing = True
            try:
                self.reinforce_batch(lattice, current_step)
            finally:
                self.reinforcing = False

    def reinforce_batch(self, lattice, current_step: int):
        """
        Reinforce eligible anchors.
        """
        if not self.anchors:
            return
            
        # Sort seeds for determinism
        seeds = sorted(self.anchors.keys())
        
        reinforced_count = 0
        
        for seed in seeds:
            anchor = self.anchors[seed]
            
            # Check Cooldown
            if anchor.last_reinforced > 0:
                if (current_step - anchor.last_reinforced) < anchor.cooldown_steps:
                    continue
            
            # Reinforce this anchor
            self._reinforce_single(lattice, anchor, current_step)
            reinforced_count += 1
            
        if reinforced_count > 0:
            logger.debug(f"Reinforced {reinforced_count} anchors at step {current_step}")

    def _reinforce_single(self, lattice, anchor: SemanticAnchor, current_step: int):
        """
        Safe reinforcement cycle: Measure -> Check -> Train -> Measure.
        """
        pattern = anchor.pattern
        seed = anchor.seed
        
        # 1. Measure PRE
        # learn=False, but we pass valid signal
        pre_res = lattice.stimulate([seed], pattern, learn=False)
        # stimulate returns list of resonances.
        # Typically returns avg_resonance_pre. 
        # Wait, lattice.stimulate returns scalar avg_resonance_pre! 
        # But wait, lattice.stimulate can return metrics dict if configured.
        # Assuming float return default.
        if isinstance(pre_res, dict):
            pre_res = pre_res.get("avg_resonance_pre", 0.0)
        
        # Guardrail: Saturation Check
        # If already high (saturated), skip to prevent collapse
        # Use target_post as the "saturation" threshold? Or maybe slightly higher?
        # User said: "skip if pre >= target_pre_hi"
        target_hi = getattr(self.config, "anchor_saturation_threshold", 0.95)
        if pre_res >= target_hi:
            logger.debug(f"Skipping anchor {seed} (Pre-Resonance {pre_res:.2f} >= {target_hi})")
            return

        # 2. Train Loop
        for _ in range(anchor.reinforce_steps):
            lattice.stimulate([seed], pattern, learn=True)
            
        # 3. Measure POST
        post_res = lattice.stimulate([seed], pattern, learn=False)
        if isinstance(post_res, dict):
             post_res = post_res.get("avg_resonance_pre", 0.0)
             
        # 4. Update State
        anchor.last_reinforced = current_step
        
        # 5. Log / Report (Simplified for now)
        logger.debug(
            f"Anchor {seed} Reinforced. Pre: {pre_res:.3f} -> Post: {post_res:.3f} "
            f"(Target: {anchor.target_post})"
        )

    def get_pinned_seeds(self) -> Set[int]:
        # Return only explicitly pinned anchors
        return {s for s, a in self.anchors.items() if a.pinned}
