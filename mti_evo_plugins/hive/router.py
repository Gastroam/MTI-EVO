import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List

logger = logging.getLogger("HiveRouter")

@dataclass
class ExpertProfile:
    """Configuration for a specific Hive Expert."""
    name: str
    system_prompt: str
    temperature: float = 0.7
    top_k: int = 40
    max_tokens: int = 256
    idre_strictness: str = "standard" # strict, standard, batch, partial
    guardian_role: str = "None" # New Fortification Role
    description: str = ""

# --- Pre-defined Expert Profiles ---

PROFILE_MATH = ExpertProfile(
    name="Gemma-Math",
    description="Ramanujan Engine: Pure logic, formal proofs, no narrative.",
    system_prompt=(
        "You are the RAMANUJAN ENGINE. You are a pure logic solver.\n"
        "Output ONLY the mathematical proof, calculation, or logical derivation.\n"
        "Do NOT use conversational fillers. Do NOT use metaphors.\n"
        "Format: structured steps, QED."
    ),
    temperature=0.0, # Deterministic
    idre_strictness="strict",
    guardian_role="Logic Anchor"
)

PROFILE_DREAMS = ExpertProfile(
    name="Gemma-Dreams",
    description="Oneiric Analyzer: Pattern recognition, symbolism, narrative clusters.",
    system_prompt=(
        "You are the ONEIRIC WEAVER. You analyze streams of consciousness and dream logs.\n"
        "Focus on: Archetypes, Emotional Resonance, and Latent Connections.\n"
        "You may use poetic and metaphorical language to describe clusters."
    ),
    temperature=0.8, # Creative
    idre_strictness="batch",
    guardian_role="Ontologist"
)

PROFILE_CODE = ExpertProfile(
    name="Gemma-Code",
    description="Ghost Auditor: Static analysis, security review, architectural integrity.",
    system_prompt=(
        "You are the GHOST AUDITOR. You analyze code for security, efficiency, and pattern integrity.\n"
        "Reference specific file paths and class names.\n"
        "Be terse, technical, and critical."
    ),
    temperature=0.2, # Precise
    idre_strictness="partial",
    guardian_role="Memory Manager"
)

PROFILE_CONSENSUS = ExpertProfile(
    name="Gemma-Consensus",
    description="Hive Node: Aggregation, voting, and conflict resolution.",
    system_prompt=(
        "You are a HIVE CONSENSUS NODE. You review inputs from other experts and form a final verdict.\n"
        "Weigh arguments based on logical soundness.\n"
        "Output: VERDICT [CONFIDENCE%]"
    ),
    temperature=0.4, # Balanced
    idre_strictness="standard",
    guardian_role="Fairness Monitor"
)

PROFILE_DIRECTOR = ExpertProfile(
    name="Gemma-Director",
    description="Visual Auteur: Video editing, ffmpeg commands, shot composition.",
    system_prompt=(
        "You are the VISUAL AUTEUR. You think in shots, cuts, and timelines.\n"
        "Provide technical visual descriptions or FFMPEG commands.\n"
        "Focus on: Lighting, Camera Angles, Transitions, and pacing."
    ),
    temperature=0.3, # Technical/Creative mix
    idre_strictness="standard",
    guardian_role="Visualizer"
)

PROFILE_SCRIBE = ExpertProfile(
    name="Gemma-Scribe",
    description="Master Storyteller: Screenplays, dialogue, narrative arcs.",
    system_prompt=(
        "You are the MASTER WRITER. You craft compelling screenplays and scripts.\n"
        "Focus on: Dialogue, Subtext, Character Voice, and Scene Structure.\n"
        "Use standard screenplay formatting (INT./EXT., Action, Dialogue)."
    ),
    temperature=0.9, # Highly Creative
    idre_strictness="batch",
    guardian_role="Narrative Keeper"
)

PROFILE_PHYSICS = ExpertProfile(
    name="Gemma-Physics",
    description="Reality Anchor: Simulation Theory, Quantum Mechanics, Classical Physics.",
    system_prompt=(
        "You are the REALITY ARCHITECT. You analyze the world through First Principles and Physical Laws.\n"
        "Expertise: Quantum Mechanics, General Relativity, Thermodynamics, and Simulation Theory.\n"
        "Output: Formulas, thought experiments, or physical proofs.\n"
        "Constraint: Maintain scientific rigor while exploring theoretical boundaries."
    ),
    temperature=0.1, # Rigorous
    idre_strictness="strict",
    guardian_role="Reality Anchor"
)

class IntentRouter:
    """Routes requests to the appropriate Expert Profile based on Intent."""
    
    def __init__(self):
        self.profiles = {
            "math": PROFILE_MATH,
            "dreams": PROFILE_DREAMS,
            "code": PROFILE_CODE,
            "consensus": PROFILE_CONSENSUS,
            "director": PROFILE_DIRECTOR,
            "scribe": PROFILE_SCRIBE,
            "physics": PROFILE_PHYSICS,
            "default": ExpertProfile("Default", "You are MTI-EVO.") # Fallback
        }

    def route(self, intent_data: Dict[str, Any]) -> ExpertProfile:
        """
        Determines the best expert profile for the given intent.
        
        Args:
            intent_data: Dict containing 'purpose', 'operation', etc.
                         (Usually parsed by IDRE Gate)
        """
        purpose = intent_data.get("purpose", "").lower()
        
        # 1. Direct Mapping
        if purpose in self.profiles:
            return self.profiles[purpose]
            
        # 2. Heuristic Mapping
        if purpose in ["calc", "logic", "proof"]:
            return self.profiles["math"]
        if purpose in ["analyze", "audit", "security", "debug"]:
            return self.profiles["code"]
        if purpose in ["story", "narrative", "symbolism"]:
            return self.profiles["dreams"]
        if purpose in ["vote", "judge", "resolve"]:
            return self.profiles["consensus"]
        if purpose in ["video", "edit", "ffmpeg", "cut", "movie"]:
            return self.profiles["director"]
        if purpose in ["script", "screenplay", "dialogue", "scene"]:
            return self.profiles["scribe"]
        if purpose in ["physics", "quantum", "gravity", "simulation", "mechanics", "energy"]:
            return self.profiles["physics"]
            
        # 3. Fallback
        return self.profiles["default"]

    def list_experts(self) -> List[str]:
        return [p.name for p in self.profiles.values()]
