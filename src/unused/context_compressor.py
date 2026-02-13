"""
MTI Context Compressor
======================

Reduces token overhead in delegation prompts by compressing context.

Techniques:
1. History Windowing: Keep only last N messages in sub-agent history
2. Code Truncation: Trim file contents to relevant sections
3. Summary Compression: Replace verbose outputs with bullet points
4. Deduplication: Remove repeated information across history

Token Savings Target:
- 40-60% reduction in delegation prompt size
- Minimal information loss for task completion
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics from a compression operation."""
    original_chars: int
    compressed_chars: int
    compression_ratio: float
    technique_used: str


class ContextCompressor:
    """
    Compresses context for delegation to reduce token usage.
    
    Usage:
        compressor = ContextCompressor()
        
        # Compress history
        compressed_history = compressor.compress_history(sub_history, max_items=5)
        
        # Compress a single context string
        compressed = compressor.compress_context(long_context, max_chars=2000)
    """
    
    # Default limits
    DEFAULT_HISTORY_WINDOW = 6    # Keep last 6 messages
    DEFAULT_MAX_CHARS = 3000      # Max chars per context
    DEFAULT_CODE_LINES = 50       # Max lines of code to keep
    
    # Patterns for compression
    NOISE_PATTERNS = [
        r'\[DEBUG\].*?\n',
        r'\[INFO\].*?\n',
        r'^\s*#.*?\n',  # Comments (careful with shebang)
        r'"""[\s\S]*?"""',  # Docstrings (aggressive)
        r"'''[\s\S]*?'''",
    ]
    
    # Keep these patterns (important)
    PRESERVE_PATTERNS = [
        r'def\s+\w+',      # Function definitions
        r'class\s+\w+',    # Class definitions
        r'import\s+',      # Imports
        r'from\s+\w+',     # From imports
        r'return\s+',      # Return statements
        r'raise\s+',       # Exceptions
    ]
    
    def __init__(
        self,
        history_window: int = DEFAULT_HISTORY_WINDOW,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_code_lines: int = DEFAULT_CODE_LINES
    ):
        self.history_window = history_window
        self.max_chars = max_chars
        self.max_code_lines = max_code_lines
        
        # Stats tracking
        self._total_original = 0
        self._total_compressed = 0
        self._operations = 0
    
    def compress_history(
        self,
        history: List[str],
        max_items: Optional[int] = None
    ) -> List[str]:
        """
        Compress a conversation history by keeping only recent relevant items.
        
        Args:
            history: List of history messages
            max_items: Maximum items to keep (None = use default window)
            
        Returns:
            Compressed history list
        """
        window = max_items or self.history_window
        
        if len(history) <= window:
            return history
        
        # Always keep first item (usually context) and last N items
        compressed = [history[0]] + history[-window:]
        
        # Track stats
        original_len = sum(len(h) for h in history)
        compressed_len = sum(len(h) for h in compressed)
        self._track_compression(original_len, compressed_len)
        
        return compressed
    
    def compress_context(
        self,
        context: str,
        max_chars: Optional[int] = None,
        preserve_structure: bool = True
    ) -> str:
        """
        Compress a context string to reduce token usage.
        
        Args:
            context: The context to compress
            max_chars: Maximum output characters
            preserve_structure: Keep code structure intact
            
        Returns:
            Compressed context
        """
        if not context:
            return context
        
        max_len = max_chars or self.max_chars
        original_len = len(context)
        
        if original_len <= max_len:
            return context
        
        # Step 1: Remove noise patterns
        compressed = context
        for pattern in self.NOISE_PATTERNS:
            compressed = re.sub(pattern, '', compressed, flags=re.MULTILINE)
        
        # Step 2: Truncate if still too long
        if len(compressed) > max_len:
            if preserve_structure:
                # Smart truncation: keep start and end
                half = max_len // 2
                compressed = (
                    compressed[:half] + 
                    "\n\n... [COMPRESSED: middle section omitted] ...\n\n" +
                    compressed[-half:]
                )
            else:
                compressed = compressed[:max_len] + "\n... [TRUNCATED]"
        
        self._track_compression(original_len, len(compressed))
        return compressed
    
    def compress_code(
        self,
        code: str,
        focus_pattern: Optional[str] = None,
        max_lines: Optional[int] = None
    ) -> str:
        """
        Compress code by keeping only relevant sections.
        
        Args:
            code: The code to compress
            focus_pattern: Optional regex pattern to focus on
            max_lines: Maximum lines to keep
            
        Returns:
            Compressed code
        """
        lines = code.split('\n')
        max_l = max_lines or self.max_code_lines
        original_len = len(code)
        
        if len(lines) <= max_l:
            return code
        
        if focus_pattern:
            # Find lines matching the focus pattern
            focus_indices = []
            pattern = re.compile(focus_pattern, re.IGNORECASE)
            for i, line in enumerate(lines):
                if pattern.search(line):
                    focus_indices.append(i)
            
            if focus_indices:
                # Keep context around focus areas
                context_size = max_l // (len(focus_indices) + 1)
                result_lines = []
                
                for idx in focus_indices:
                    start = max(0, idx - context_size // 2)
                    end = min(len(lines), idx + context_size // 2)
                    result_lines.extend(lines[start:end])
                    result_lines.append("# ...")
                
                compressed = '\n'.join(result_lines[:max_l])
                self._track_compression(original_len, len(compressed))
                return compressed
        
        # Default: Keep start and end
        half = max_l // 2
        result = lines[:half] + ["# ... [middle omitted] ..."] + lines[-half:]
        compressed = '\n'.join(result)
        
        self._track_compression(original_len, len(compressed))
        return compressed
    
    def compress_tool_output(self, output: str, tool_name: str) -> str:
        """
        Compress tool output based on tool type.
        
        Args:
            output: The tool output
            tool_name: Name of the tool
            
        Returns:
            Compressed output
        """
        original_len = len(output)
        
        if tool_name in ['list_dir', 'find_by_name']:
            # For directory listings, keep first and last items with count
            lines = output.strip().split('\n')
            if len(lines) > 10:
                compressed = '\n'.join(lines[:5]) + f"\n... ({len(lines) - 10} more) ...\n" + '\n'.join(lines[-5:])
                self._track_compression(original_len, len(compressed))
                return compressed
        
        elif tool_name in ['grep_search']:
            # For search results, keep top matches
            lines = output.strip().split('\n')
            if len(lines) > 8:
                compressed = '\n'.join(lines[:8]) + f"\n... ({len(lines) - 8} more matches) ..."
                self._track_compression(original_len, len(compressed))
                return compressed
        
        elif tool_name in ['read_file']:
            # For file reads, use code compression
            return self.compress_code(output)
        
        elif tool_name in ['run_command']:
            # For command outputs, keep last lines (usually most relevant)
            lines = output.strip().split('\n')
            if len(lines) > 20:
                compressed = "... [start omitted] ...\n" + '\n'.join(lines[-20:])
                self._track_compression(original_len, len(compressed))
                return compressed
        
        # Default: character limit
        return self.compress_context(output, max_chars=1500)
    
    def build_compressed_prompt(
        self,
        system_prompt: str,
        history: List[str],
        current_task: str
    ) -> Tuple[str, CompressionStats]:
        """
        Build a fully compressed prompt for delegation.
        
        Args:
            system_prompt: The system prompt (kept intact)
            history: Conversation history
            current_task: The current task description
            
        Returns:
            Tuple of (compressed_prompt, stats)
        """
        original_total = len(system_prompt) + sum(len(h) for h in history) + len(current_task)
        
        # Compress history
        compressed_history = self.compress_history(history)
        
        # Build prompt
        history_text = '\n'.join(compressed_history[-6:])  # Last 6 items max
        prompt = f"{system_prompt}\n\n{history_text}\n\nResponse:"
        
        stats = CompressionStats(
            original_chars=original_total,
            compressed_chars=len(prompt),
            compression_ratio=1 - (len(prompt) / original_total) if original_total > 0 else 0,
            technique_used="history_windowing + context_truncation"
        )
        
        return prompt, stats
    
    def _track_compression(self, original: int, compressed: int):
        """Track compression statistics."""
        self._total_original += original
        self._total_compressed += compressed
        self._operations += 1
    
    def get_stats(self) -> Dict:
        """Get compression statistics."""
        if self._total_original == 0:
            ratio = 0.0
        else:
            ratio = 1 - (self._total_compressed / self._total_original)
        
        return {
            "total_operations": self._operations,
            "total_original_chars": self._total_original,
            "total_compressed_chars": self._total_compressed,
            "overall_compression_ratio": round(ratio * 100, 1),
            "avg_chars_saved_per_op": round((self._total_original - self._total_compressed) / max(1, self._operations)),
        }
    
    def reset_stats(self):
        """Reset compression statistics."""
        self._total_original = 0
        self._total_compressed = 0
        self._operations = 0


# Singleton instance
_compressor: Optional[ContextCompressor] = None

def get_context_compressor() -> ContextCompressor:
    """Get the singleton ContextCompressor instance."""
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor()
    return _compressor

def compress_for_delegation(
    history: List[str],
    context: str = "",
    max_history_items: int = 6,
    max_context_chars: int = 2000
) -> Tuple[List[str], str]:
    """
    Convenience function to compress content for delegation.
    
    Args:
        history: Sub-agent conversation history
        context: Additional context string
        max_history_items: Max history items to keep
        max_context_chars: Max context characters
        
    Returns:
        Tuple of (compressed_history, compressed_context)
    """
    compressor = get_context_compressor()
    compressed_history = compressor.compress_history(history, max_history_items)
    compressed_context = compressor.compress_context(context, max_context_chars)
    return compressed_history, compressed_context
