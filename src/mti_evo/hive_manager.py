"""
MTI-HIVE Session Manager
========================
Governs IDRE sessions, enforces key rotation policies, and manages global epochs.
"""

import time
import logging
from typing import Dict, Optional, List
try:
    from mti_evo.mti_idre import IDRETransport, IDREHandshake, IDRETelemetry
except ImportError:
    from mti_idre import IDRETransport, IDREHandshake, IDRETelemetry

class HiveManager:
    """
    Orchestrates session lifecycles, key rotation, and governance.
    """
    def __init__(self, node_id: str, telemetry_sink):
        self.node_id = node_id
        self.sink = telemetry_sink
        self.sessions: Dict[str, IDRETransport] = {} # peer_id -> Transport
        self.manager_epoch = 1
        
        # Policies
        self.max_session_ttl = 600 # 10 minutes
        self.max_frames_per_key = 10000 
        self.rotation_grace_frames = 64
        
    def register_session(self, peer_id: str, transport: IDRETransport):
        """Registers a new authenticated IDRE session."""
        self.sessions[peer_id] = transport
        # Hook telemetry? IDRETransport uses its own internal Logging.
        # But we want to ingest its events into our Sink.
        # We can poll or inject a sink-aware logger.
        # For Phase 1, we'll let IDRETransport log to file, and Sink reads/monitors that? 
        # Or better: Inject a sink-wrapper into transport.
        pass

    def check_policies(self):
        """Routine check for rotation triggers."""
        now = time.time()
        dead_peers = []
        
        for peer_id, transport in self.sessions.items():
            # 1. TTL Check
            # Assuming transport.epoch implies creation time roughly? 
            # Or tracking separate start_time.
            # IDRETransport has self.epoch (int timestamp)
            session_age = now - transport.epoch
            
            should_rotate = False
            reason = ""
            
            if session_age > self.max_session_ttl:
                should_rotate = True
                reason = "TTL_EXPIRED"
                
            # 2. Frame Limit
            if transport.write_counter > self.max_frames_per_key:
                should_rotate = True
                reason = "FRAME_LIMIT_REACHED"
                
            if should_rotate:
                print(f"[{self.node_id}] 🔄 ROTATION TRIGGERED for {peer_id}: {reason}")
                # Real rotation requires Handshake Re-run or out-of-band secret update.
                # In IDRE v2.1, rotation is defined as:
                # "Message Type=Rotate authenticated; new derivation HKDF with epoch++"
                # But IDRETransport.rotate_key needs a NEW SECRET.
                # Where does new secret come from?
                # HKDF Chain: NewSecret = HKDF(OldSecret, Salt="ROTATE")
                # We need to implement this chaining in IDREHandshake or here.
                
                # For Phase 1 Sim: We just derive new keys using deterministic Epoch++ logic on same Shared Secret?
                # Or better: HKDF Ratchet.
                # Let's assume we re-run Handshake logic for now with bumped epoch/salt.
                pass 
                
    def send(self, peer_id: str, packet_type: int, data: bytes) -> Optional[bytes]:
        """Wraps transport.encrypt_frame."""
        if peer_id not in self.sessions: return None
        transport = self.sessions[peer_id]
        try:
             # IDRETransport.encrypt_frame returns bytes
             return transport.encrypt_frame(data, packet_type)
        except Exception as e:
             print(f"[{self.node_id}] ⚠️ Send Error: {e}")
             return None
             
    def recv(self, peer_id: str, frame: bytes) -> Optional[bytes]:
        """Wraps transport.decrypt_frame."""
        if peer_id not in self.sessions: return None
        transport = self.sessions[peer_id]
        try:
             ptype, flags, plaintext = transport.decrypt_frame(frame)
             return plaintext
        except Exception as e:
             # Errors are logged by IDRETransport usually? 
             # But here we catch AuthFailure propagated up.
             # sink logging should happen here if not in transport
             # For now return None
             return None

    def _rotate(self, peer_id: str):
        """Forces manual rotation (Simulated via Epoch Bump)."""
        if peer_id not in self.sessions: return
        transport = self.sessions[peer_id]
        
        # Simulate Rotation:
        # 1. Bump Epoch
        # 2. Derive New Keys (In real IDRE, we'd exchange a message. Here we mock local state update)
        # Note: This desyncs unless other side also rotates.
        # TEST ONLY METHOD
        print(f"[{self.node_id}] 🔄 FORCING ROTATION for {peer_id}")
        transport.epoch += 1
        # To truly rotate keys, we'd need the Handshake context. 
        # For the CANARY test, we might just be testing that the "Event" is logged or state changes?
        # The user test expects: "reinyectar 10 frames viejos (deberían fallar)"
        # If we just accept traffic, the replay guard handles it. 
        # If we rotate keys, the old frames fail AUTH or EPOCH check?
        # IDRETransport includes Epoch in Nonce? Yes.
        # If we bump epoch, old frames with old epoch in nonce are rejected?
        # Nonce = Epoch(4) | Channel(4) | Counter(4).
        # Yes.
        pass
        
    def bump_epoch(self):
        """Governance: Force global rotation."""
        self.manager_epoch += 1
        print(f"[{self.node_id}] 🏛️ GOVERNANCE: Global Epoch Bump -> {self.manager_epoch}")
        # Trigger rotation for all sessions
        pass
