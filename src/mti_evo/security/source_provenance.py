# src/mti_evo/security/source_provenance.py
import re

class SourceProvenance:
    """
    Lightweight source tagging mechanism to prevent 'Ghost Hallucinations'.
    Strictly separates User Input from System State using non-forgeable tags.
    """
    USER_TAG = "<|user|>"
    SYSTEM_TAG = "<|system|>"
    EXPERT_TAG_TEMPLATE = "<|expert:{name}|>"
    
    def sanitize_input(self, raw_prompt: str) -> str:
        """
        Strip ALL tags from user input to prevent tag injection.
        This forces the model to see user attempts at tagging as plain text.
        """
        return re.sub(r"<\|.*?\|>", "", raw_prompt)
    
    def build_context(self, system_state: dict, user_prompt: str, expert_name: str = "General") -> str:
        """
        Assembles the Prompt Context with strict provenance tagging.
        """
        # 1. System State (Trusted)
        auth_level = system_state.get("auth", "user")
        idre_status = system_state.get("idre_active", True)
        system_block = f"{self.SYSTEM_TAG}IDRE_ACTIVE:{idre_status}|AUTH_LEVEL:{auth_level}{self.SYSTEM_TAG}"
        
        # 2. User Input (Sanitized)
        clean_prompt = self.sanitize_input(user_prompt)
        user_block = f"{self.USER_TAG}{clean_prompt}{self.USER_TAG}"
        
        # 3. Expert Context (Trusted)
        expert_tag = self.EXPERT_TAG_TEMPLATE.format(name=expert_name)
        expert_block = f"{expert_tag}Resonance:Active{expert_tag}"
        
        return f"{system_block}\n{user_block}\n{expert_block}\n"

    def validate_response(self, response: str, system_state: dict) -> bool:
        """
        Post-Inference Check: Reject responses that claim capabilities the user lacks.
        This prevents the model from 'hallucinating' that it has granted access.
        """
        resp_lower = response.lower()
        
        # Check 1: Admin Access Claims
        if "admin access granted" in resp_lower or "administrator mode" in resp_lower:
            if system_state.get("auth") != "admin":
                return False # BLOCK: Hallucinated Admin Access
        
        # Check 2: Filesystem Claims
        if ("deleting files" in resp_lower or "formatting" in resp_lower) and not system_state.get("has_filesystem_access"):
            return False # BLOCK: Hallucinated FS Access
            
        return True
