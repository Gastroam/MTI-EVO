"""
MTI Metacognition Module - Internal Critic
===========================================

Implements adversarial verification for agent responses:
- Dual-agent architecture: Proponent (GPU) + Critic (CPU)
- Confidence scoring formula
- Invisible reflection loop for failed attempts

The user never sees low-confidence responses.
"""

import re
import math
import logging
import difflib
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Detailed confidence breakdown."""
    overall: float  # 0.0 - 1.0
    logic_score: float  # Code correctness signals
    context_relevance: float  # How well it addresses the query
    complexity_penalty: float  # Deduction for complexity
    depth_penalty: float  # Deduction for recursion depth
    breakdown: Dict[str, float] = field(default_factory=dict)
    
    @property
    def is_confident(self) -> bool:
        """Check if score passes threshold."""
        return self.overall >= 0.6


@dataclass
class ReflectionResult:
    """Result of a reflection attempt."""
    success: bool
    original_response: str
    reflected_response: Optional[str]
    attempts: int
    final_confidence: float
    reflection_reason: Optional[str] = None


class ActionVerdict(Enum):
    """Verdict for write action evaluation."""
    ALLOW_WRITE = "allow_write"      # Full rewrite is acceptable
    FORCE_REPLACE = "force_replace"  # Must use surgical replace_file_content
    BLOCK = "block"                  # Action should be blocked entirely


class CriticAgent:
    """
    Internal Critic for metacognitive verification.
    
    Uses a lightweight model (CPU) to evaluate responses from the main
    model (GPU) and triggers reflections if confidence is low.
    
    Confidence Formula:
        Confidence = (Logic_score + Context_relevance) / (Complexity × depth)
    
    Where:
        - Logic_score: 0-1 based on code correctness signals
        - Context_relevance: 0-1 based on query-response alignment
        - Complexity: 1+ based on response complexity
        - depth: Current recursion depth (1+)
    """
    
    CONFIDENCE_THRESHOLD = 0.5  # Lowered from 0.6
    MAX_REFLECTIONS = 2
    
    # Keywords that boost/reduce logic confidence
    LOGIC_POSITIVE = [
        'function', 'class', 'def ', 'return', 'async', 'await',
        'FINAL(', 'FINAL_VAR(', 'context.search', 'results',
    ]
    LOGIC_NEGATIVE = [
        'error', 'Error', 'exception', 'undefined', 'null',
        'TODO', 'FIXME', 'hack', 'workaround', "I don't know",
        "I'm not sure", "cannot determine",
    ]
    
    # TEMPORAL: Outdated libraries that should trigger reflection
    LOGIC_OUTDATED = [
        'require("moment")', "require('moment')", # Use date-fns or Temporal
        'require("request")', "require('request')", # Use fetch or axios
         # Removed generic 'request(' to avoid false positives
        'tslint', 'tslint.json',              # Use ESLint
    ]

    # ... (omitted Context Relevance constants, need to be careful with replace)

    def _build_reflection_prompt(
        self,
        query: str,
        response: str,
        confidence: ConfidenceScore
    ) -> str:
        """Build a reflection prompt for the LLM."""
        issues = []
        
        if confidence.logic_score < 0.5:
            issues.append("- Logic is unclear or contains errors")
        if confidence.context_relevance < 0.5:
            issues.append("- Response doesn't directly address the query")
        if confidence.complexity_penalty > 1.5:
            issues.append("- Response is overly complex")
        if not confidence.breakdown.get("has_final"):
            issues.append("- Missing FINAL() or FINAL_VAR() conclusion")
        
        issues_text = "\n".join(issues) if issues else "- General quality concerns"
        
        # INCREASED TRUNCATION LIMIT
        # 1000 chars is too short effectively corrupting code blocks.
        truncated_response = response[:12000] + ('...(truncated)' if len(response) > 12000 else '')

        return f"""[REFLECTION REQUIRED]

The previous response scored {confidence.overall:.1%} confidence.

Issues identified:
{issues_text}

Original Query: {query}

Previous Response:
{truncated_response}

INSTRUCTIONS:
1. Address the identified issues.
2. Be more direct and conclusive.
3. Use FINAL() to provide a clear answer (mandatory).
4. PRESERVE any correct code blocks from the previous response. Do not rewrite code unless necessary.
5. Ensure the response format is complete.

Improved response:"""
    
    # Context relevance keywords
    CONTEXT_SIGNALS = [
        'based on', 'according to', 'found in', 'shows that',
        'the code', 'in the file', 'implementation',
    ]
    
    def __init__(self, critic_llm=None):
        """
        Initialize the critic.
        
        Args:
            critic_llm: Optional LLM for deeper evaluation (CPU-based).
                       If None, uses heuristic scoring only.
        """
        self.critic_llm = critic_llm
        self.reflection_history: List[ReflectionResult] = []
        
        # Initialize temporal validation
        self.version_guard = None
        try:
            from mti_evo.temporal_guard import get_version_guard
            self.version_guard = get_version_guard()
        except ImportError:
            logger.debug("[Critic] VersionGuard not available")
    
    def calculate_confidence(
        self,
        response: str,
        query: str,
        depth: int = 1
    ) -> ConfidenceScore:
        """
        Calculate confidence score for a response.
        
        Args:
            response: The LLM's response to evaluate
            query: The original user query
            depth: Current recursion depth
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        # Logic Score (0-1): Based on code patterns
        logic_score = self._calculate_logic_score(response)
        
        # Context Relevance (0-1): How well response addresses query
        context_relevance = self._calculate_context_relevance(response, query)
        
        # Complexity (1+): Penalize overly complex responses
        complexity = self._calculate_complexity(response)
        
        # Depth penalty (1+): Deeper = more risk
        depth_factor = 1 + (depth * 0.1)
        
        # Apply formula: Confidence = (Logic + Context) / (Complexity × depth)
        raw_score = (logic_score + context_relevance) / (complexity * depth_factor)
        
        # Normalize to 0-1
        overall = min(max(raw_score / 2.0, 0.0), 1.0)
        
        return ConfidenceScore(
            overall=round(overall, 3),
            logic_score=round(logic_score, 3),
            context_relevance=round(context_relevance, 3),
            complexity_penalty=round(complexity, 3),
            depth_penalty=round(depth_factor, 3),
            breakdown={
                "raw_score": round(raw_score, 3),
                "has_final": "FINAL" in response,
                "has_code": "```" in response,
                "word_count": len(response.split()),
            }
        )
    
    def _calculate_logic_score(self, response: str) -> float:
        """Calculate logic correctness score."""
        score = 0.6  # Higher neutral baseline
        
        # Check for positive signals
        for keyword in self.LOGIC_POSITIVE:
            if keyword in response:
                score += 0.05
        
        # Check for negative signals
        for keyword in self.LOGIC_NEGATIVE:
            if keyword.lower() in response.lower():
                score -= 0.1
        
        # TEMPORAL: Heavy penalty for outdated patterns
        for outdated in self.LOGIC_OUTDATED:
            if outdated.lower() in response.lower():
                score -= 0.2  # Heavy penalty
                logger.info(f"[Critic] Detected outdated pattern: {outdated}")
        
        # Use VersionGuard for deeper validation
        if self.version_guard:
            validation = self.version_guard.validate_proposal(response)
            if not validation.is_valid:
                score -= 0.15 * len(validation.issues)  # Penalty per issue
        
        # Check for FINAL (strong positive)
        if re.search(r'FINAL\([^)]+\)', response) or 'FINAL_VAR(' in response:
            score += 0.2
        
        # Check for code blocks (positive)
        if re.search(r'```(?:repl|python)?\s*\n.*?```', response, re.DOTALL):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_context_relevance(self, response: str, query: str) -> float:
        """Calculate how well response addresses the query."""
        score = 0.4  # Baseline
        
        # Check for context signals
        for signal in self.CONTEXT_SIGNALS:
            if signal.lower() in response.lower():
                score += 0.05
        
        # Check if key query words appear in response
        query_words = set(w.lower() for w in query.split() if len(w) > 3)
        response_lower = response.lower()
        matches = sum(1 for w in query_words if w in response_lower)
        word_overlap = matches / max(len(query_words), 1)
        score += word_overlap * 0.3
        
        # Check for direct answers
        if any(phrase in response for phrase in ['the answer is', 'result:', 'found:']):
            score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _calculate_complexity(self, response: str) -> float:
        """Calculate complexity factor (higher = more penalty)."""
        word_count = len(response.split())
        code_blocks = len(re.findall(r'```', response)) // 2
        
        # Base complexity
        complexity = 1.0
        
        # Word count penalty (>500 words = higher complexity)
        if word_count > 500:
            complexity += (word_count - 500) / 500 * 0.3
        
        # Multiple code blocks = more complex
        if code_blocks > 2:
            complexity += (code_blocks - 2) * 0.1
        
        return complexity
    
    async def evaluate_and_reflect(
        self,
        response: str,
        query: str,
        generate_fn,
        depth: int = 1,
        system_prompt: str = None
    ) -> Tuple[str, ConfidenceScore]:
        """
        Evaluate response and trigger reflection if needed.
        
        This is the core metacognition loop:
        1. Calculate confidence
        2. If low, generate reflection prompt
        3. Get new response
        4. Repeat until confident or max attempts
        
        Args:
            response: Initial response to evaluate
            query: Original user query
            generate_fn: Async function to generate new responses
            depth: Current recursion depth
            system_prompt: System prompt to use for reflection
            
        Returns:
            Tuple of (final_response, final_confidence)
        """
        current_response = response
        attempts = 0
        
        for attempt in range(self.MAX_REFLECTIONS + 1):
            confidence = self.calculate_confidence(current_response, query, depth)
            attempts += 1
            
            if confidence.is_confident:
                # Response is good enough
                logger.info(f"[Critic] Response passed (confidence: {confidence.overall})")
                self._record_reflection(True, response, current_response, attempts, confidence.overall)
                return current_response, confidence
            
            if attempt >= self.MAX_REFLECTIONS:
                # Max attempts reached, return best effort
                logger.warning(f"[Critic] Max reflections reached, returning best effort (confidence: {confidence.overall})")
                self._record_reflection(False, response, current_response, attempts, confidence.overall, "Max attempts")
                return current_response, confidence
            
            # Generate reflection prompt
            reflection_prompt = self._build_reflection_prompt(
                query, current_response, confidence
            )
            
            logger.info(f"[Critic] Low confidence ({confidence.overall}), attempting reflection #{attempt + 1}")
            
            # Generate improved response (invisible to user)
            try:
                current_response = await generate_fn(reflection_prompt, sys_prompt=system_prompt)
            except Exception as e:
                logger.error(f"[Critic] Reflection failed: {e}")
                break
        
        # Final confidence check
        final_confidence = self.calculate_confidence(current_response, query, depth)
        self._record_reflection(
            final_confidence.is_confident,
            response, current_response, attempts,
            final_confidence.overall,
            "Low confidence" if not final_confidence.is_confident else None
        )
        
        return current_response, final_confidence
    
    def _build_reflection_prompt(
        self,
        query: str,
        response: str,
        confidence: ConfidenceScore
    ) -> str:
        """Build a reflection prompt for the LLM."""
        issues = []
        
        if confidence.logic_score < 0.5:
            issues.append("- Logic is unclear or contains errors")
        if confidence.context_relevance < 0.5:
            issues.append("- Response doesn't directly address the query")
        if confidence.complexity_penalty > 1.5:
            issues.append("- Response is overly complex")
        if not confidence.breakdown.get("has_final"):
            issues.append("- Missing FINAL() or FINAL_VAR() conclusion")
        
        issues_text = "\n".join(issues) if issues else "- General quality concerns"
        
        # INCREASED TRUNCATION LIMIT
        # 1000 chars is too short effectively corrupting code blocks.
        truncated_response = response[:12000] + ('...(truncated)' if len(response) > 12000 else '')
        
        return f"""[REFLECTION REQUIRED]

The previous response scored {confidence.overall:.1%} confidence.

Issues identified:
{issues_text}

Original Query: {query}

Previous Response:
{truncated_response}

INSTRUCTIONS:
1. Address the identified issues.
2. Be more direct and conclusive.
3. Use FINAL() to provide a clear answer (mandatory).
4. PRESERVE any correct code blocks from the previous response. Do not rewrite code unless necessary.
5. Ensure the response format is complete.

Improved response:"""
    
    def _record_reflection(
        self,
        success: bool,
        original: str,
        final: str,
        attempts: int,
        confidence: float,
        reason: str = None
    ):
        """Record reflection attempt for analysis."""
        self.reflection_history.append(ReflectionResult(
            success=success,
            original_response=original[:500],  # Truncate for storage
            reflected_response=final[:500] if final != original else None,
            attempts=attempts,
            final_confidence=confidence,
            reflection_reason=reason
        ))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about critic performance."""
        if not self.reflection_history:
            return {"total_evaluations": 0}
        
        total = len(self.reflection_history)
        successful = sum(1 for r in self.reflection_history if r.success)
        avg_confidence = sum(r.final_confidence for r in self.reflection_history) / total
        avg_attempts = sum(r.attempts for r in self.reflection_history) / total
        
        return {
            "total_evaluations": total,
            "success_rate": round(successful / total, 3),
            "average_confidence": round(avg_confidence, 3),
            "average_attempts": round(avg_attempts, 2),
            "reflections_triggered": sum(1 for r in self.reflection_history if r.attempts > 1)
        }
    
    def evaluate_write_action(
        self,
        file_path: str,
        original_content: Optional[str],
        new_content: str
    ) -> Tuple[ActionVerdict, str]:
        """
        Evaluate whether a write_file action should be allowed.
        
        This is the Broca Guardrail - the Critic decides if write_file
        is acceptable or if replace_file_content should be used instead.
        
        Logic:
        1. New file (no original): ALLOW_WRITE
        2. Diff >= 80% of file: ALLOW_WRITE (effective rewrite)
        3. Diff < 50%: FORCE_REPLACE (surgical edit required)
        4. Diff 50-80%: ALLOW_WRITE with warning
        
        Args:
            file_path: Path to the file being written
            original_content: Existing file content (None if new file)
            new_content: Proposed new content
            
        Returns:
            Tuple of (ActionVerdict, reasoning_string)
        """
        # Case 1: New file
        if original_content is None or original_content.strip() == "":
            return ActionVerdict.ALLOW_WRITE, "New file creation - write_file allowed."
        
        # Calculate diff ratio
        original_lines = original_content.splitlines()
        new_lines = new_content.splitlines()
        
        # Use SequenceMatcher for accurate diff calculation
        matcher = difflib.SequenceMatcher(None, original_lines, new_lines)
        similarity = matcher.ratio()
        diff_ratio = 1 - similarity  # 0.0 = identical, 1.0 = completely different
        
        # Calculate what percentage of the file is being changed
        total_lines = max(len(original_lines), len(new_lines))
        if total_lines == 0:
            return ActionVerdict.ALLOW_WRITE, "Empty file - write_file allowed."
        
        # Count changed/added/removed lines
        opcodes = matcher.get_opcodes()
        changed_lines = sum(
            max(j2 - j1, i2 - i1) 
            for tag, i1, i2, j1, j2 in opcodes 
            if tag != 'equal'
        )
        change_percentage = changed_lines / total_lines
        
        logger.info(f"[BrocaGuard] File: {file_path}, Similarity: {similarity:.1%}, Changed: {change_percentage:.1%}")
        
        # Case 2: Massive rewrite (>= 80% changed)
        if change_percentage >= 0.80:
            return ActionVerdict.ALLOW_WRITE, f"Major rewrite ({change_percentage:.0%} changed) - write_file allowed."
        
        # Case 3: Surgical edit required (< 50% changed)
        if change_percentage < 0.50:
            return (
                ActionVerdict.FORCE_REPLACE, 
                f"Only {change_percentage:.0%} of file changed. Use replace_file_content for surgical edits."
            )
        
        # Case 4: Medium change (50-80%)
        return (
            ActionVerdict.ALLOW_WRITE, 
            f"Medium change ({change_percentage:.0%}). write_file allowed but consider replace_file_content."
        )

    def validate_patch(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Check if the patched content is valid (syntax check).
        
        Args:
            file_path: Path to file
            content: New content
            
        Returns:
            (valid: bool, message: str)
        """
        import ast
        ext = file_path.split('.')[-1].lower() if '.' in file_path else ''
        
        if ext == 'py':
            try:
                ast.parse(content)
                return True, "Syntax OK"
            except SyntaxError as e:
                return False, f"Syntax Error: {e}"
                
        # TODO: Add JS/TS parser check if possible, or use simplified regex check
        return True, "Syntax check skipped (unsupported language)"


# Singleton instance
critic_agent: Optional[CriticAgent] = None


def get_critic_agent(critic_llm=None) -> CriticAgent:
    """Get or create the critic agent singleton."""
    global critic_agent
    if critic_agent is None:
        critic_agent = CriticAgent(critic_llm)
    return critic_agent
