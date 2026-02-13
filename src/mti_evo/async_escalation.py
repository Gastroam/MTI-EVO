"""
MTI Async Escalation Manager
=============================

Non-blocking escalation system for Cloud ↔ Local delegation.

Problem:
- When Gemma escalates to Cloud, the current implementation BLOCKS
- This wastes GPU idle time while waiting for Cloud response

Solution:
- Use asyncio/threading to request Cloud guidance asynchronously
- Continue Gemma processing with a "best effort" approach
- Merge Cloud guidance when it arrives

Architecture:
- EscalationRequest: Encapsulates the escalation context
- AsyncEscalationManager: Manages pending escalations and responses
- Uses concurrent.futures for thread-based async (compatible with sync code)
"""

import threading
import queue
import time
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

class EscalationStatus(Enum):
    """Status of an escalation request."""
    PENDING = "pending"       # Request submitted, waiting for response
    COMPLETED = "completed"   # Cloud response received
    FAILED = "failed"        # Cloud unreachable or timeout
    CANCELLED = "cancelled"   # Request cancelled (no longer needed)


@dataclass
class EscalationRequest:
    """Encapsulates an escalation request to Cloud."""
    request_id: str
    reason: str
    context: str
    original_task: str
    status: EscalationStatus = EscalationStatus.PENDING
    guidance: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


class AsyncEscalationManager:
    """
    Manages asynchronous escalation requests to Cloud.
    
    Usage:
        manager = AsyncEscalationManager(cloud_llm)
        
        # Submit escalation (non-blocking)
        request_id = manager.submit_escalation(reason, context, original_task)
        
        # Continue local processing...
        
        # Check if guidance is available
        guidance = manager.get_guidance(request_id, timeout=0.1)
        if guidance:
            # Apply guidance
            pass
    """
    
    def __init__(
        self,
        cloud_llm: Any,
        max_workers: int = 2,
        default_timeout: float = 30.0
    ):
        """
        Initialize the async escalation manager.
        
        Args:
            cloud_llm: The LLM instance for Cloud queries
            max_workers: Maximum concurrent escalation requests
            default_timeout: Default timeout for Cloud requests
        """
        self.cloud_llm = cloud_llm
        self.default_timeout = default_timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        self._requests: Dict[str, EscalationRequest] = {}
        self._futures: Dict[str, concurrent.futures.Future] = {}
        self._lock = threading.Lock()
        
        # Stats
        self._stats = {
            "total_escalations": 0,
            "successful": 0,
            "failed": 0,
            "cancelled": 0,
            "avg_response_time": 0.0,
        }
    
    def submit_escalation(
        self,
        reason: str,
        context: str,
        original_task: str,
        callback: Optional[Callable[[str, Optional[str]], None]] = None
    ) -> str:
        """
        Submit an escalation request asynchronously.
        
        Args:
            reason: Why the local agent needs help
            context: Compressed context for Cloud
            original_task: The original task description
            callback: Optional callback(request_id, guidance) when complete
            
        Returns:
            Request ID for tracking
        """
        import uuid
        request_id = f"esc_{uuid.uuid4().hex[:8]}"
        
        request = EscalationRequest(
            request_id=request_id,
            reason=reason,
            context=context[:500],  # Limit context to reduce tokens
            original_task=original_task[:300],
        )
        
        with self._lock:
            self._requests[request_id] = request
            self._stats["total_escalations"] += 1
        
        # Build the Cloud prompt
        cloud_prompt = self._build_escalation_prompt(reason, context, original_task)
        
        # Submit to thread pool
        future = self.executor.submit(
            self._execute_escalation,
            request_id,
            cloud_prompt,
            callback
        )
        
        with self._lock:
            self._futures[request_id] = future
        
        print(f"🔄 [ASYNC ESCALATION] Submitted: {request_id} (reason: {reason[:50]})", flush=True)
        return request_id
    
    def _build_escalation_prompt(self, reason: str, context: str, original_task: str) -> str:
        """Build a compact escalation prompt for Cloud."""
        return f"""ESCALATION FROM LOCAL SUB-AGENT:

Reason: {reason}
Context: {context[:500]}
Task: {original_task[:300]}

PROVIDE BRIEF GUIDANCE (max 200 words):
- What approach should local agent take?
- What patterns apply?
- Key pitfalls to avoid?"""

    def _execute_escalation(
        self,
        request_id: str,
        prompt: str,
        callback: Optional[Callable]
    ) -> Optional[str]:
        """Execute the Cloud request (runs in thread pool)."""
        try:
            # Track timing
            start_time = time.time()
            
            # Make Cloud request
            guidance = self.cloud_llm.generate(prompt)
            
            elapsed = time.time() - start_time
            
            # Update request status
            with self._lock:
                if request_id in self._requests:
                    req = self._requests[request_id]
                    req.status = EscalationStatus.COMPLETED
                    req.guidance = guidance
                    req.completed_at = time.time()
                    self._stats["successful"] += 1
                    
                    # Update rolling average
                    total = self._stats["total_escalations"]
                    prev_avg = self._stats["avg_response_time"]
                    self._stats["avg_response_time"] = ((prev_avg * (total - 1)) + elapsed) / total
            
            print(f"☁️ [ASYNC ESCALATION] Completed: {request_id} in {elapsed:.1f}s", flush=True)
            
            # Invoke callback if provided
            if callback:
                callback(request_id, guidance)
            
            return guidance
            
        except Exception as e:
            with self._lock:
                if request_id in self._requests:
                    req = self._requests[request_id]
                    req.status = EscalationStatus.FAILED
                    req.error = str(e)
                    req.completed_at = time.time()
                    self._stats["failed"] += 1
            
            print(f"⚠️ [ASYNC ESCALATION] Failed: {request_id} - {e}", flush=True)
            
            if callback:
                callback(request_id, None)
            
            return None
    
    def get_guidance(
        self,
        request_id: str,
        timeout: float = 0.0
    ) -> Optional[str]:
        """
        Get guidance for a request (optionally wait).
        
        Args:
            request_id: The escalation request ID
            timeout: Max seconds to wait (0 = don't wait, return immediately)
            
        Returns:
            Guidance string if available, None otherwise
        """
        with self._lock:
            if request_id not in self._requests:
                return None
            
            req = self._requests[request_id]
            
            # Already complete?
            if req.status == EscalationStatus.COMPLETED:
                return req.guidance
            elif req.status in [EscalationStatus.FAILED, EscalationStatus.CANCELLED]:
                return None
        
        # Wait for completion if timeout > 0
        if timeout > 0:
            future = self._futures.get(request_id)
            if future:
                try:
                    result = future.result(timeout=timeout)
                    return result
                except concurrent.futures.TimeoutError:
                    return None
                except Exception:
                    return None
        
        return None
    
    def cancel_escalation(self, request_id: str) -> bool:
        """Cancel a pending escalation request."""
        with self._lock:
            if request_id in self._requests:
                req = self._requests[request_id]
                if req.status == EscalationStatus.PENDING:
                    req.status = EscalationStatus.CANCELLED
                    self._stats["cancelled"] += 1
                    return True
        return False
    
    def is_pending(self, request_id: str) -> bool:
        """Check if an escalation is still pending."""
        with self._lock:
            if request_id in self._requests:
                return self._requests[request_id].status == EscalationStatus.PENDING
        return False
    
    def get_stats(self) -> Dict:
        """Get escalation statistics."""
        with self._lock:
            return self._stats.copy()
    
    def cleanup_old_requests(self, max_age_seconds: float = 300.0):
        """Remove completed requests older than max_age."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            to_remove = [
                rid for rid, req in self._requests.items()
                if req.completed_at and req.completed_at < cutoff
            ]
            for rid in to_remove:
                del self._requests[rid]
                if rid in self._futures:
                    del self._futures[rid]
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=False)


# Singleton instance
_manager: Optional[AsyncEscalationManager] = None

def get_escalation_manager(cloud_llm: Any = None) -> AsyncEscalationManager:
    """Get or create the async escalation manager singleton."""
    global _manager
    if _manager is None:
        if cloud_llm is None:
            raise ValueError("cloud_llm required for first initialization")
        _manager = AsyncEscalationManager(cloud_llm)
    return _manager
