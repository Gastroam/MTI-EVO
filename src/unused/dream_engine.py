"""
MTI Dream Engine - Semantic Sleep System
=========================================

Implements synthetic memory generation during idle time:
- Generates "phantom code" for future scenarios
- Pre-trains patterns in Broca Lobe
- Runs in Shadow Workspace for safety

COST CONTROLS (CRITICAL):
- Auto-dream mode is DISABLED by default
- Manual trigger required to avoid cloud API costs
- Local-only mode available for zero API calls
"""

import os
import json
import asyncio
import logging
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DreamMode(Enum):
    """Operating modes for the Dream Engine."""
    DISABLED = "disabled"        # No dreaming at all
    MANUAL = "manual"            # Only dream on explicit trigger
    AUTO_LOCAL = "auto_local"    # Auto-dream using local model only (no API costs)
    AUTO_CLOUD = "auto_cloud"    # Auto-dream using cloud (COSTS MONEY!)


class DreamTrigger(Enum):
    """What triggered a dream session."""
    MANUAL = "manual"           # User explicitly triggered
    IDLE = "idle"               # System was idle
    SHUTDOWN = "shutdown"       # Pre-shutdown consolidation
    SCHEDULED = "scheduled"     # Scheduled dream time


@dataclass
class DreamScenario:
    """A scenario to dream about."""
    id: str
    description: str
    source: str  # e.g., "TODO", "design_doc", "error_log"
    priority: float  # 0-1, higher = more important to dream about
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class DreamResult:
    """Result of a dream session."""
    scenario_id: str
    success: bool
    phantom_code: Optional[str]
    patterns_learned: List[str]
    tokens_used: int
    cost_estimate: float  # Estimated API cost
    duration_seconds: float
    error: Optional[str] = None


@dataclass
class DreamStats:
    """Statistics about dream sessions."""
    total_dreams: int = 0
    successful_dreams: int = 0
    failed_dreams: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    patterns_generated: int = 0
    last_dream_time: Optional[float] = None


class DreamEngine:
    """
    Semantic Sleep System - Pre-training through imagination.
    
    Generates synthetic experiences during idle time by:
    1. Reading TODOs, design notes, and error patterns
    2. Creating phantom code solutions in Shadow Workspace
    3. Storing successful patterns in Broca Lobe
    
    COST CONTROLS:
    - Auto-mode is DISABLED by default
    - Manual trigger always available
    - Tracks token usage and cost estimates
    """
    
    # Cost estimates per 1K tokens (conservative estimates)
    COST_PER_1K_TOKENS = {
        "gemini": 0.0005,    # Gemini Flash
        "gemini-pro": 0.002, # Gemini Pro
        "ollama": 0.0,       # Local = free
        "deepseek": 0.0001,  # DeepSeek cheap
    }
    
    # Idle threshold (seconds) before auto-dream kicks in
    IDLE_THRESHOLD = 300  # 5 minutes
    
    # Max tokens per dream session
    MAX_TOKENS_PER_DREAM = 2000
    
    def __init__(
        self,
        workspace_path: str,
        shadow_workspace: Optional[str] = None,
        llm_provider: str = "ollama",
        broca_store: Optional[Any] = None
    ):
        self.workspace = Path(workspace_path)
        self.shadow = Path(shadow_workspace) if shadow_workspace else self.workspace / ".dream_shadow"
        self.llm_provider = llm_provider
        self.broca = broca_store  # For storing learned patterns
        
        # State
        self.mode: DreamMode = DreamMode.DISABLED  # SAFE DEFAULT
        self.stats = DreamStats()
        self.dream_queue: List[DreamScenario] = []
        self.is_dreaming = False
        self.last_activity_time = time.time()
        self._dream_task: Optional[asyncio.Task] = None
        self._generate_fn: Optional[Callable] = None
        
        # Ensure shadow workspace exists
        self.shadow.mkdir(parents=True, exist_ok=True)
        
        # Load persisted stats
        self._load_stats()
    
    # ==================== MODE CONTROLS ====================
    
    def set_mode(self, mode: DreamMode) -> Dict[str, Any]:
        """
        Set the dream engine mode.
        
        WARNING: AUTO_CLOUD mode will incur API costs!
        """
        old_mode = self.mode
        self.mode = mode
        
        logger.info(f"[DreamEngine] Mode changed: {old_mode.value} -> {mode.value}")
        
        if mode == DreamMode.AUTO_CLOUD:
            logger.warning("[DreamEngine] ⚠️ AUTO_CLOUD enabled - this will incur API costs!")
        
        return {
            "old_mode": old_mode.value,
            "new_mode": mode.value,
            "warning": "AUTO_CLOUD will incur API costs!" if mode == DreamMode.AUTO_CLOUD else None
        }
    
    def disable(self):
        """Disable all dreaming (safest mode)."""
        self.set_mode(DreamMode.DISABLED)
        if self._dream_task and not self._dream_task.done():
            self._dream_task.cancel()
    
    def enable_manual_only(self):
        """Enable manual-only mode (no auto costs)."""
        self.set_mode(DreamMode.MANUAL)
    
    def enable_auto_local(self):
        """Enable auto-dreaming with local models only (free)."""
        self.set_mode(DreamMode.AUTO_LOCAL)
    
    # ==================== SCENARIO MANAGEMENT ====================
    
    def add_scenario(self, scenario: DreamScenario):
        """Add a scenario to the dream queue."""
        self.dream_queue.append(scenario)
        # Sort by priority
        self.dream_queue.sort(key=lambda s: s.priority, reverse=True)
        logger.info(f"[DreamEngine] Added scenario: {scenario.description[:50]}")
    
    def scan_todos(self) -> List[DreamScenario]:
        """Scan workspace for TODO comments and create scenarios."""
        scenarios = []
        
        # Simple TODO scanner
        for ext in ['*.py', '*.ts', '*.tsx', '*.js']:
            for file_path in self.workspace.rglob(ext):
                if self._should_skip(file_path):
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines):
                        if 'TODO' in line or 'FIXME' in line:
                            scenario = DreamScenario(
                                id=f"todo_{file_path.name}_{i}",
                                description=line.strip(),
                                source="TODO",
                                priority=0.7 if 'FIXME' in line else 0.5,
                                context={
                                    "file": str(file_path.relative_to(self.workspace)),
                                    "line": i + 1,
                                    "surrounding": lines[max(0, i-3):i+4]
                                }
                            )
                            scenarios.append(scenario)
                except Exception as e:
                    logger.debug(f"Error scanning {file_path}: {e}")
        
        return scenarios
    
    def _should_skip(self, path: Path) -> bool:
        """Check if path should be skipped."""
        skip_patterns = ['node_modules', '__pycache__', '.git', 'venv', 'dist', '.dream_shadow']
        return any(p in str(path) for p in skip_patterns)
    
    # ==================== DREAM EXECUTION ====================
    
    async def dream_manual(
        self,
        generate_fn: Callable,
        max_scenarios: int = 3
    ) -> List[DreamResult]:
        """
        Manually trigger a dream session.
        
        This is the SAFE way to dream - user explicitly triggers it.
        
        Args:
            generate_fn: Async function to generate LLM responses
            max_scenarios: Max scenarios to process
            
        Returns:
            List of DreamResult
        """
        if self.mode == DreamMode.DISABLED:
            return [DreamResult(
                scenario_id="blocked",
                success=False,
                phantom_code=None,
                patterns_learned=[],
                tokens_used=0,
                cost_estimate=0,
                duration_seconds=0,
                error="Dream Engine is DISABLED. Call enable_manual_only() first."
            )]
        
        self._generate_fn = generate_fn
        self.is_dreaming = True
        results = []
        
        try:
            # Refresh scenarios if queue is empty
            if not self.dream_queue:
                new_scenarios = self.scan_todos()
                self.dream_queue.extend(new_scenarios[:10])
            
            # Process top scenarios
            for scenario in self.dream_queue[:max_scenarios]:
                result = await self._dream_single(scenario)
                results.append(result)
                
                # Update stats
                self.stats.total_dreams += 1
                if result.success:
                    self.stats.successful_dreams += 1
                    self.stats.patterns_generated += len(result.patterns_learned)
                else:
                    self.stats.failed_dreams += 1
                self.stats.total_tokens += result.tokens_used
                self.stats.total_cost += result.cost_estimate
            
            self.stats.last_dream_time = time.time()
            self._save_stats()
            
            # Remove processed scenarios
            processed_ids = {r.scenario_id for r in results}
            self.dream_queue = [s for s in self.dream_queue if s.id not in processed_ids]
            
        finally:
            self.is_dreaming = False
        
        return results
    
    async def _dream_single(self, scenario: DreamScenario) -> DreamResult:
        """Process a single dream scenario."""
        start_time = time.time()
        
        try:
            # Build dream prompt
            prompt = self._build_dream_prompt(scenario)
            
            # Generate phantom solution
            if self._generate_fn:
                response = await self._generate_fn(prompt)
            else:
                response = "[No generate function provided]"
            
            # Extract code and patterns
            phantom_code = self._extract_code(response)
            patterns = self._extract_patterns(response, scenario)
            
            # Estimate tokens and cost
            tokens = len(response.split()) * 1.3  # Rough estimate
            cost = self._estimate_cost(int(tokens))
            
            # Save phantom code to shadow workspace
            if phantom_code:
                shadow_file = self.shadow / f"dream_{scenario.id}.py"
                with open(shadow_file, 'w', encoding='utf-8') as f:
                    f.write(f"# Dream: {scenario.description}\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                    f.write(phantom_code)
            
            # Store patterns in Broca (if available)
            if self.broca and patterns:
                for pattern in patterns:
                    try:
                        # Fix: Broca API uses process_thought, not store_pattern
                        # We combine pattern and description for richer context
                        thought = f"{pattern} {scenario.description}"
                        self.broca.process_thought(thought, learn=True)
                    except Exception as e:
                        logger.debug(f"Broca store failed: {e}")
            
            return DreamResult(
                scenario_id=scenario.id,
                success=True,
                phantom_code=phantom_code,
                patterns_learned=patterns,
                tokens_used=int(tokens),
                cost_estimate=cost,
                duration_seconds=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"[DreamEngine] Dream failed: {e}")
            return DreamResult(
                scenario_id=scenario.id,
                success=False,
                phantom_code=None,
                patterns_learned=[],
                tokens_used=0,
                cost_estimate=0,
                duration_seconds=time.time() - start_time,
                error=str(e)
            )
    
    def _build_dream_prompt(self, scenario: DreamScenario) -> str:
        """Build a prompt for dreaming about a scenario."""
        context_str = ""
        if scenario.context.get("surrounding"):
            context_str = "\n".join(scenario.context["surrounding"])
        
        return f"""You are in DREAM MODE - generating synthetic code for future use.

SCENARIO: {scenario.description}
SOURCE: {scenario.source}
FILE: {scenario.context.get('file', 'unknown')}

CONTEXT:
```
{context_str}
```

TASK:
1. Imagine a complete solution for this scenario
2. Write clean, production-ready code
3. Include error handling
4. Add brief comments explaining your approach

IMPORTANT: This is pre-training. The code may never be used, but if it is, 
it should be immediately usable.

```python
# Solution for: {scenario.description}
"""
    
    def _extract_code(self, response: str) -> Optional[str]:
        """Extract code block from response."""
        import re
        match = re.search(r'```(?:python|typescript|javascript)?\s*\n(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_patterns(self, response: str, scenario: DreamScenario) -> List[str]:
        """Extract reusable patterns from the dream."""
        patterns = []
        
        # Simple pattern extraction based on keywords
        keywords = ['pattern:', 'approach:', 'technique:', 'strategy:']
        for line in response.split('\n'):
            line_lower = line.lower()
            for kw in keywords:
                if kw in line_lower:
                    patterns.append(line.strip())
        
        # Add scenario-based pattern
        if scenario.source == "TODO":
            patterns.append(f"TODO_SOLUTION:{scenario.description[:50]}")
        
        return patterns[:5]  # Limit patterns
    
    def _estimate_cost(self, tokens: int) -> float:
        """Estimate API cost for token count."""
        # Update TokenTracker
        from mti_evo.token_tracker import get_token_tracker
        tracker = get_token_tracker()
        tracker.track_usage(self.llm_provider, tokens, "dream_model")
        
        rate = self.COST_PER_1K_TOKENS.get(self.llm_provider, 0.001)
        return (tokens / 1000) * rate
    
    # ==================== PERSISTENCE ====================
    
    def _load_stats(self):
        """Load stats from disk."""
        stats_file = self.shadow / "dream_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    data = json.load(f)
                self.stats = DreamStats(**data)
            except Exception:
                pass
    
    def _save_stats(self):
        """Save stats to disk."""
        stats_file = self.shadow / "dream_stats.json"
        try:
            with open(stats_file, 'w') as f:
                json.dump({
                    "total_dreams": self.stats.total_dreams,
                    "successful_dreams": self.stats.successful_dreams,
                    "failed_dreams": self.stats.failed_dreams,
                    "total_tokens": self.stats.total_tokens,
                    "total_cost": self.stats.total_cost,
                    "patterns_generated": self.stats.patterns_generated,
                    "last_dream_time": self.stats.last_dream_time,
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dream stats: {e}")
    
    # ==================== STATUS ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dream engine status."""
        return {
            "mode": self.mode.value,
            "is_dreaming": self.is_dreaming,
            "queue_size": len(self.dream_queue),
            "stats": {
                "total_dreams": self.stats.total_dreams,
                "successful": self.stats.successful_dreams,
                "failed": self.stats.failed_dreams,
                "total_tokens": self.stats.total_tokens,
                "total_cost": round(self.stats.total_cost, 4),
                "patterns_generated": self.stats.patterns_generated,
                "last_dream": self.stats.last_dream_time,
            },
            "cost_warning": self.mode == DreamMode.AUTO_CLOUD
        }
    
    # ==================== CONSOLIDATION (Cognitive Enhancement) ====================
    
    def wake(self):
        """
        Wake from dreaming - user activity detected.
        Cancels any active dream and marks activity time.
        """
        self.last_activity_time = time.time()
        if self._dream_task and not self._dream_task.done():
            self._dream_task.cancel()
            logger.info("☀️ [DreamEngine] Waking - dream cancelled")
        self.is_dreaming = False
    
    def record_activity(self):
        """Record user activity to reset idle timer."""
        self.last_activity_time = time.time()
    
    def is_idle(self) -> bool:
        """Check if system has been idle long enough to dream."""
        return (time.time() - self.last_activity_time) > self.IDLE_THRESHOLD
    
    async def consolidate_memories(self, hippocampus=None, wernicke=None) -> Dict[str, Any]:
        """
        Run memory consolidation (FIP/FID) during idle time.
        
        This is the "REM sleep" of the system - strengthening good memories,
        weakening unused ones.
        
        Args:
            hippocampus: HippocampusMemory instance
            wernicke: WernickeAnalyzer instance
            
        Returns:
            Consolidation results
        """
        results = {
            "hippocampus": None,
            "wernicke": None,
            "timestamp": time.time()
        }
        
        # Hippocampus: Apply sparse encoding to recent episodes
        if hippocampus:
            try:
                # Count episodes with sparse encoding
                encoded_count = 0
                for ep in hippocampus.episodes[-50:]:  # Last 50 episodes
                    if ep.id not in hippocampus.sparse_memories:
                        if ep.embedding:
                            sparse = hippocampus.sparse_encoder.encode(ep.embedding)
                            hippocampus.sparse_memories[ep.id] = sparse
                            encoded_count += 1
                
                results["hippocampus"] = {
                    "episodes_encoded": encoded_count,
                    "total_sparse": len(hippocampus.sparse_memories)
                }
                logger.info(f"🌙 [DreamEngine] Consolidated {encoded_count} Hippocampus episodes")
            except Exception as e:
                results["hippocampus"] = {"error": str(e)}
        
        # Wernicke: Apply FIP/FID to cell assemblies
        if wernicke and hasattr(wernicke, 'assembly_manager'):
            try:
                plasticity_stats = wernicke.assembly_manager.apply_plasticity()
                results["wernicke"] = plasticity_stats
                logger.info(f"🌙 [DreamEngine] Applied Wernicke plasticity: {plasticity_stats}")
            except Exception as e:
                results["wernicke"] = {"error": str(e)}
        
        return results
    
    async def start_idle_consolidation_loop(self, hippocampus=None, wernicke=None, check_interval: int = 60):
        """
        Start background loop that consolidates during idle time.
        
        Args:
            hippocampus: HippocampusMemory instance
            wernicke: WernickeAnalyzer instance
            check_interval: Seconds between idle checks
        """
        logger.info("🌙 [DreamEngine] Idle consolidation loop started")
        
        while True:
            await asyncio.sleep(check_interval)
            
            if self.mode == DreamMode.DISABLED:
                continue
                
            if self.is_idle() and not self.is_dreaming:
                logger.info("🌙 [DreamEngine] System idle - starting consolidation")
                await self.consolidate_memories(hippocampus, wernicke)


# Singleton
_dream_engine: Optional[DreamEngine] = None


def get_dream_engine(workspace_path: str = None) -> DreamEngine:
    """Get or create the dream engine singleton."""
    global _dream_engine
    if _dream_engine is None:
        if not workspace_path:
            workspace_path = os.getcwd()
        _dream_engine = DreamEngine(workspace_path)
    return _dream_engine
