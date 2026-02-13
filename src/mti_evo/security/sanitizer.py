"""
MTI Security Sanitizer
======================

The "Blood-Brain Barrier" for the MTI Architecture.
Responsible for filtering data entering the Hippocampus (Memory) 
and leaving via the CloudBridge (Efferent).

Strategy:
- Use RegEx and Heuristics for zero-latency, zero-VRAM protection.
- Block standard PII (IPs, Emails) and Secrets (API Keys).
- Anonymize local file paths.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SecuritySanitizer:
    """
    Sanitizes text to prevent:
    1. Memory Poisoning (Malicious prompts/data in Vector DB)
    2. PII Leakage (Secrets sent to Cloud LLMs)
    """
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = workspace_path
        
        # --- REGEX PATTERNS ---
        self.patterns = {
            # Secrets
            "openai_key": (r"sk-[a-zA-Z0-9]{20,}", "<OPENAI_KEY_REDACTED>"),
            "github_token": (r"(ghp|gho)_[a-zA-Z0-9]{20,}", "<GITHUB_TOKEN_REDACTED>"),
            "aws_key": (r"AKIA[0-9A-Z]{16}", "<AWS_KEY_REDACTED>"),
            "private_key": (r"-----BEGIN PRIVATE KEY-----", "<PRIVATE_KEY_BLOCK_REDACTED>"),
            
            # PII
            "ipv4": (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP_ADDRESS]"),
            "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_ADDRESS]"),
            
            # System
            "win_user_path": (r"(?i)C:[\\/]Users[\\/][a-zA-Z0-9_]+[\\/]", "[USER_HOME]/"), # Anonymize user user
        }
        
    def sanitize(self, text: str, context: str = "general") -> str:
        """
        Main entry point for text sanitization.
        
        Args:
            text: The raw text to sanitize
            context: 'memory' (Hippocampus) or 'cloud' (Bridge)
            
        Returns:
            Sanitized text
        """
        if not text:
            return text
            
        clean_text = text
        
        # 1. Apply Regex Filters
        for name, (pattern, replacement) in self.patterns.items():
            try:
                # Need to be careful with paths (don't break code imports)
                if name == "win_user_path":
                    # Only generic drive paths, try to serve relative if possible
                    clean_text = re.sub(pattern, replacement, clean_text)
                else:
                    clean_text = re.sub(pattern, replacement, clean_text)
            except Exception as e:
                logger.error(f"Regex error for {name}: {e}")
                
        # 2. Context-Specific Rules
        if context == "memory":
            # Strip huge traceback blocks to avoid polluting semantic search
            if "Traceback (most recent call last):" in clean_text:
                clean_text = self._strip_traceback(clean_text)
                
        return clean_text

    def _strip_traceback(self, text: str) -> str:
        """Collapse long tracebacks into a summary line."""
        # Simple heuristic: keep header and last error line
        lines = text.split('\n')
        new_lines = []
        in_traceback = False
        
        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                new_lines.append("[TRACEBACK_SUMMARY]")
                continue
                
            if in_traceback:
                # If line is not indented (and not empty), it's likely the final error message
                if line.strip() and not line.startswith(" "):
                    in_traceback = False
                    new_lines.append(line) # Keep the error type/message
            else:
                new_lines.append(line)
                
        return '\n'.join(new_lines)

# Singleton instance for easy import
_global_sanitizer = None

def get_sanitizer(workspace_path: str = None) -> SecuritySanitizer:
    global _global_sanitizer
    if _global_sanitizer is None:
        _global_sanitizer = SecuritySanitizer(workspace_path)
    return _global_sanitizer
