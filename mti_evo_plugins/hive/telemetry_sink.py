"""
MTI-HIVE Telemetry Sink
=======================
Consolidates IDRE telemetry, manages forensic bundles, and generates alerts.
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Deque
from collections import deque
from dataclasses import asdict

class TelemetryAlert(Exception):
    pass

class RotationStorm(TelemetryAlert): pass
class HighReplayRate(TelemetryAlert): pass

class TelemetrySink:
    def __init__(self, node_id: str, log_dir: str = "hive_audit"):
        self.node_id = node_id
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        self.current_bundle = []
        self.start_epoch = int(time.time())
        
        # Anomaly Detection Windows
        self.window_size = 60 # seconds
        self.rotation_events: Deque[float] = deque()
        self.replay_events: Deque[float] = deque()
        
    def ingest_event(self, event: Dict):
        """Ingest a raw event dictionary from IDRETelemetry."""
        # Add Node Context
        event['node_id'] = self.node_id
        
        self.current_bundle.append(event)
        
        # Analyze for Risks
        self._analyze_risk(event)
        
    def _analyze_risk(self, event):
        now = time.time()
        ev_type = event.get('event_type')
        
        # Cleaning old window
        while self.rotation_events and self.rotation_events[0] < now - self.window_size:
            self.rotation_events.popleft()
        while self.replay_events and self.replay_events[0] < now - self.window_size:
            self.replay_events.popleft()
            
        # Check specific types
        if ev_type == 'KEY_ROTATE':
            self.rotation_events.append(now)
            if len(self.rotation_events) > 3: # >3 per minute
                self._emit_alert("ROTATION_STORM", f"{len(self.rotation_events)} rotations in {self.window_size}s")
                
        elif ev_type == 'REPLAY_DETECTED':
            self.replay_events.append(now)
            # Threshold: simple count for now, say 5 per minute is suspicious for a canary
            if len(self.replay_events) > 5:
                self._emit_alert("HIGH_REPLAY_RATE", f"{len(self.replay_events)} replays in {self.window_size}s")
                
    def _emit_alert(self, alert_type, msg):
        alert = {
            "type": "ALERT",
            "alert_code": alert_type,
            "message": msg,
            "timestamp": time.time(),
            "node_id": self.node_id
        }
        print(f"[{self.node_id}] 🚨 ALERT: {alert_type} - {msg}")
        self.current_bundle.append(alert)

        
    def _summarize_metrics(self) -> Dict:
        """Calculates p50/p95 latency and avg coherence."""
        latencies = []
        coherences = []
        gpu_utils = []
        req_count = 0
        res_count = 0
        
        for ev in self.current_bundle:
            if ev.get("event_type") == "LLM_INFER":
                if "latency_ms" in ev: 
                    latencies.append(ev["latency_ms"])
                if "coherence" in ev: 
                    coherences.append(ev["coherence"])
                if "gpu_stats" in ev and ev["gpu_stats"]:
                    gpu_utils.append(ev["gpu_stats"].get("util", 0))
                req_count += 1
                
            if ev.get("event_type") == "LLM_RES_RECV":
                res_count += 1
                
        stats = {
            "total_requests": req_count,
            "total_responses": res_count,
            "avg_coherence": sum(coherences)/len(coherences) if coherences else 0.0,
            "avg_gpu_util": sum(gpu_utils)/len(gpu_utils) if gpu_utils else 0.0
        }
        
        if latencies:
            s = sorted(latencies)
            stats["p50_latency_ms"] = s[len(s)//2]
            stats["p95_latency_ms"] = s[int(len(s)*0.95)]
        else:
            stats["p50_latency_ms"] = 0.0
            stats["p95_latency_ms"] = 0.0
            
        return stats

    def flush_bundle(self, epoch: int) -> str:
        """Writes current events to a signed bundle."""
        if not self.current_bundle:
            return None
            
        bundle_id = f"bundle_{self.node_id}_{epoch}_{int(time.time())}"
        filename = os.path.join(self.log_dir, f"{bundle_id}.jsonl")
        
        # Calculate Metrics
        metrics = self._summarize_metrics()
        
        manifest = {
            "bundle_id": bundle_id,
            "node_id": self.node_id,
            "epoch": epoch,
            "event_count": len(self.current_bundle),
            "generated_at": time.time(),
            "metrics": metrics
        }
        
        # Calculate State Hash (Chain of custody)
        # Simple Merkle-like hash of all events
        bundle_str = json.dumps(self.current_bundle, sort_keys=True)
        manifest["content_hash"] = hashlib.sha256(bundle_str.encode()).hexdigest()
        
        # Sign Manifest (Simulated signature)
        manifest["signature"] = f"SIG_{self.node_id}_{manifest['content_hash'][:8]}"
        
        with open(filename, 'w') as f:
            # Write Manifest first
            f.write(json.dumps({"_manifest": manifest}) + "\n")
            # Write Events
            for ev in self.current_bundle:
                f.write(json.dumps(ev) + "\n")
                
        # Export Metrics Sidecar
        self._export_metrics_sidecar(filename, epoch)
        
        # Clear buffer
        self.current_bundle = []
        return filename

    def _export_metrics_sidecar(self, bundle_filename: str, epoch: int):
        """Exports raw metrics to a sidecar JSON for fusion."""
        latencies = []
        coherences = []
        gpu_stats_list = []
        total_req = 0
        total_res = 0
        
        for ev in self.current_bundle:
            if ev.get("event_type") == "LLM_INFER":
                latencies.append(ev.get("latency_ms", 0.0))
                coherences.append(ev.get("coherence", 0.0))
                if "gpu_stats" in ev:
                    gpu_stats_list.append(ev["gpu_stats"])
                total_req += 1
            if ev.get("event_type") == "LLM_RES_RECV":
                total_res += 1
                
        metrics = {
            "node_id": self.node_id,
            "epoch": epoch,
            "total_requests": total_req,
            "total_responses": total_res,
            "latencies_ms": latencies,
            "coherences": coherences,
            "gpu_stats": gpu_stats_list
        }
        
        # Calculate stats for verify script convenience
        if gpu_stats_list:
             utils = [g.get("util", 0) for g in gpu_stats_list]
             metrics["avg_gpu_util"] = sum(utils)/len(utils)
        
        mpath = bundle_filename.replace(".jsonl", "_metrics.json")
        with open(mpath, "w", encoding="utf-8") as mf:
            json.dump(metrics, mf, indent=2)
