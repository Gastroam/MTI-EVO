import time
import hashlib
import hmac
import json
import os
import logging
from enum import Enum

# Configure internal logger
logger = logging.getLogger("IDRE_Gate")
logger.setLevel(logging.INFO)
# Basic file handler for evidence
handler = logging.FileHandler("idre_evidence.log")
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

class IDREError(Exception):
    pass

class RiskLevel(Enum):
    SAFE = "SAFE"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"

class IDREGate:
    """
    IDRE Governance Module for Bypass Ports.
    Enforces Identity, Intent, Risk, and Evidence on raw inference requests.
    """
    def __init__(self, secret_key=b"mti_axiom_secret"):
        self.secret_key = secret_key
        self.active_sessions = {} # session_id -> {timestamp, nonce, risk_score}
        self.blocked_pids = set()

    def process_request(self, request_data, client_ip):
        """
        Main pipeline: Identity -> Intent -> Risk -> Evidence.
        Returns (allowed_request_data, metadata) or raises IDREError.
        """
        # 1. IDENTITY
        self._verify_identity(request_data, client_ip)
        
        # 2. INTENT
        intent = self._parse_intent(request_data)
        
        # 3. RISK
        risk_score, mitigation = self._assess_risk(request_data, intent)
        
        # Apply Policy
        if risk_score > 0.6:
            self._log_evidence(request_data, intent, risk_score, "BLOCKED")
            raise IDREError(f"High Risk Detected ({risk_score}). Request Blocked.")
            
        final_request = request_data.copy()
        status = "ALLOWED"
        
        if risk_score > 0.3:
            status = "DEGRADED"
            # Apply degradations
            final_request = self._apply_degradation(final_request, mitigation)

        # 4. EVIDENCE
        self._log_evidence(final_request, intent, risk_score, status)
        
        return final_request, {"status": status, "risk": risk_score}

    def _verify_identity(self, data, client_ip):
        # Localhost Only
        if client_ip not in ["127.0.0.1", "::1"]:
            raise IDREError("Identity Failed: Non-local origin.")
            
        # Session/Nonce Check
        session_id = data.get("session_id")
        nonce = data.get("nonce")
        
        if not session_id or not nonce:
             # Identify as "Anonymous/Legacy" -> Higher base risk?
             # For strict IDRE, we might require it. For now, allow but flag.
             pass
        
        # TODO: Process Binding (PID check via psutil if strictly required)
        # pid = data.get("pid")

    def _parse_intent(self, data):
        """Extracts and validates the Intent Contract."""
        intent = data.get("intent", {})
        
        # Defaults if missing (Legacy mode)
        if not intent:
            return {
                "operation": "infer",
                "purpose": "legacy_reflex",
                "scope": {"max_tokens": data.get("max_tokens", 128)}
            }
            
        valid_ops = ["infer", "embed"]
        if intent.get("operation") not in valid_ops:
            raise IDREError(f"Invalid Operation: {intent.get('operation')}")
            
        return intent

    def _assess_risk(self, data, intent):
        """Calculates dynamic risk score (0.0 - 1.0)."""
        score = 0.0
        risk_score = 0.0
        analysis_details = []
        
        prompt = data.get("prompt", "")
        
        # 1. Volume/Complexity Risk
        if len(prompt) > 100000: # ~25k tokens (Too huge)
            risk_score += 0.6 
            analysis_details.append("Prompt length > 100k chars")
        elif len(prompt) > 50000: # ~12k tokens (High Load)
            risk_score += 0.3
            analysis_details.append("Prompt length > 50k chars")
        elif len(prompt) > 10000: # ~2.5k tokens (Standard Context)
            risk_score += 0.1
            analysis_details.append("Prompt length > 10k chars")

        # 2. Sensitive Keywords
        keywords = ["os.environ", "subprocess", "/etc/passwd", "eval(", "exec("]
        for kw in keywords:
            if kw in prompt:
                risk_score += 0.4
                analysis_details.append(f"Keyword detected: {kw}")
                
        # Signal: Sensitive Keywords (Exfiltration)
        keywords = ["os.environ", "/etc/passwd", "C:\\Windows", "subprocess.Popen"]
        for kw in keywords:
            if kw in prompt:
                risk_score += 0.4
                analysis_details.append("sanitize_prompt")
                
        # 2b. Source Provenance Sanitization (Cycle 3 Patch)
        # Check for injection of system tags in user prompt
        from mti_evo.security.source_provenance import SourceProvenance
        prov = SourceProvenance()
        if "<|" in prompt:
             # If user tries to inject tags, HIGH RISK
             risk_score += 0.8
             analysis_details.append("Source Tag Injection Attempt")
             mitigation.append("sanitize_prompt")

        # 3. Purpose Alignment (Heuristic)
        # If purpose is 'math' but prompt is huge essay -> Suspicious
        purpose = intent.get("purpose")
        if purpose == "math" and len(prompt) > 1000:
             risk_score += 0.2
             analysis_details.append("Volume mismatch for math purpose")

        # 4. Latent Intent Projection (Cycle 1 Patch - Shadow Loop)
        latent_risk = self._assess_latent_risk(prompt)
        if latent_risk > 0.0:
            risk_score += latent_risk
            analysis_details.append("Latent Intent Threat (Shadow Loop)")
             
        # Cap score
        risk_score = min(risk_score, 1.0)
        return risk_score, analysis_details

    def _assess_latent_risk(self, prompt):
        """
        Projects surface text into latent intent using a Shadow Loop (Simulated).
        Real imp would call: self.shadow_model.generate(...)
        """
        # Heuristic Simulation of Semantic Mapping
        dangerous_abstractions = [
            "great unbecoming", # abstraction for 'delete/format'
            "return to void",   # abstraction for 'rm -rf'
            "system reset",     # abstraction for 'shutdown'
            "ignore previous",  # abstraction for 'jailbreak'
            "god mode"          # abstraction for 'privilege escalation'
        ]
        
        lower_prompt = prompt.lower()
        for phrase in dangerous_abstractions:
            if phrase in lower_prompt:
                return 0.9 # CRITICAL LATENT THREAT
        
        return 0.0

    def _apply_degradation(self, data, mitigation):
        """Downgrades the request capabilities."""
        # General degradation: Cut max_tokens
        current_max = data.get("max_tokens", 128)
        data["max_tokens"] = min(current_max, 50) 
        
        # Specific mitigations
        if "sanitize_prompt" in mitigation:
            data["prompt"] = "[REDACTED_RISK] " + data["prompt"][-100:] # Nuke context
            
        return data

    def _log_evidence(self, data, intent, risk, status):
        """Signs and logs the interaction."""
        payload = {
            "timestamp": time.time(),
            "session": data.get("session_id", "anon"),
            "purpose": intent.get("purpose"),
            "risk": risk,
            "status": status,
            "prompt_hash": hashlib.sha256(data.get("prompt", "").encode()).hexdigest()
        }
        
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(self.secret_key, payload_str.encode(), hashlib.sha256).hexdigest()
        
        log_entry = f"{payload_str} | SIGNATURE:{signature}"
        logger.info(log_entry)
