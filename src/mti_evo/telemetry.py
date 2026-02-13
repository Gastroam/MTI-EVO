"""
MTI-EVO Telemetry Bus
=====================
Centralized event logging system for the MTI architecture.
Allows any module (Brain, API, LLM) to push events that the frontend can consume.
"""
import queue
import time
import logging

# Shared event queue
_event_queue = queue.Queue()

# Configure logging
logger = logging.getLogger("mti_telemetry")

def log_event(event_type: str, message: str, data: dict = None):
    """
    Log an event to the telemetry bus.
    
    Args:
        event_type: 'system', 'thought', 'error', 'metric'
        message: Human readable string
        data: Optional metadata dict
    """
    event = {
        "timestamp": time.time(),
        "type": event_type,
        "message": message,
        "data": data or {}
    }
    _event_queue.put(event)
    
    # Also log to stdout for terminal visibility
    print(f"[{event_type.upper()}] {message}")

def get_events(max_events=50):
    """Retrieve pending events from the queue."""
    events = []
    # Fetch up to max_events
    for _ in range(max_events):
        try:
            events.append(_event_queue.get_nowait())
        except queue.Empty:
            break
    return events

# Metric History Storage
_metric_history = []
MAX_HISTORY = 300 # Keep last 300 points (approx 15-25 min at 5s interval)

def log_metric(name: str, value: float):
    """
    Log a distinct metric point for time-series analysis.
    """
    point = {
        "timestamp": time.time(),
        "name": name,
        "value": value
    }
    _metric_history.append(point)
    
    # Prune
    if len(_metric_history) > MAX_HISTORY:
        _metric_history.pop(0)

def get_metric_history():
    """Return full metric history."""
    return _metric_history
