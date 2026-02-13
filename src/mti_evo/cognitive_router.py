"""
Cognitive Router (IDRE)
=======================
Encapsulates LLM requests/responses in AEAD frames.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict
import json

try:
    from mti_evo.mti_idre import PacketType
except ImportError:
    from mti_idre import PacketType
# Note: PacketType.DATA is what we map TYPE_DATA to.
TYPE_DATA = PacketType.DATA

try:
    from mti_evo.hive_manager import HiveManager
    from mti_evo.telemetry_sink import TelemetrySink
    from mti_evo.llm_adapter import LLMAdapter, LLMResponse
except ImportError:
    from hive_manager import HiveManager
    from telemetry_sink import TelemetrySink
    from llm_adapter import LLMAdapter, LLMResponse

@dataclass
class CognitivePacket:
    kind: str  # "LLM_REQ" | "LLM_RES"
    payload: Dict

class CognitiveRouter:
    """
    Encapsula solicitudes/respuestas LLM en frames AEAD.
    """
    def __init__(self, node_id: str, manager: HiveManager, sink: TelemetrySink):
        self.node_id = node_id
        self.manager = manager
        self.sink = sink
        self.llm = LLMAdapter()

    def send_llm_request(self, peer_id: str, prompt: str) -> Optional[bytes]:
        pkt = CognitivePacket(kind="LLM_REQ", payload={"prompt": prompt})
        data = json.dumps(asdict(pkt)).encode("utf-8")
        if not hasattr(self.manager, 'send'):
             # Fallback if manager not updated yet (or use direct transport)
             # But we expect manager to have send()
             print(f"[{self.node_id}] ⚠️ Manager missing send() method")
             return None
        return self.manager.send(peer_id, TYPE_DATA, data)

    def handle_frame(self, peer_id: str, frame: bytes) -> Optional[bytes]:
        """
        Processes an incoming encrypted frame.
        Decrypts via Manager -> Routes Packet -> (Optional) Generates Response.
        """
        if not hasattr(self.manager, 'recv'):
             print(f"[{self.node_id}] ⚠️ Manager missing recv() method")
             return None
             
        # Manager recv returns plaintext bytes
        pt = self.manager.recv(peer_id, frame)
        if pt is None:
            # Auth fail or other error (already logged by manager/sink integration)
            return None
            
        try:
            obj = json.loads(pt.decode("utf-8"))
        except Exception:
            epoch = self.manager.sessions[peer_id].epoch if peer_id in self.manager.sessions else 0
            self.sink.ingest_event({"event_type": "DECODE_FAILURE", "state": "DROP", "epoch": epoch, "peer": peer_id})
            return None

        kind = obj.get("kind")
        
        if kind == "LLM_REQ":
            prompt = obj["payload"]["prompt"]
            res: LLMResponse = self.llm.infer(prompt)
            
            epoch = self.manager.sessions[peer_id].epoch
            self.sink.ingest_event({
                "event_type": "LLM_INFER", 
                "epoch": epoch,
                "latency_ms": res.latency_ms, 
                "tokens": res.tokens, 
                "coherence": res.coherence,
                "gpu_stats": res.gpu_stats
            })
            
            pkt = CognitivePacket(kind="LLM_RES", payload={
                "text": res.text, 
                "tokens": res.tokens,
                "latency_ms": res.latency_ms, 
                "coherence": res.coherence
                # Do NOT send GPU stats to peer? Or yes?
                # User asked to "Registrar gpu_util... por nodo" -> Log Locally.
                # No need to send over wire.
            })
            
            data = json.dumps(asdict(pkt)).encode("utf-8")
            return self.manager.send(peer_id, TYPE_DATA, data)

        elif kind == "LLM_RES":
            # Entrega al caller (quien envió la solicitud)
            # In simulation, we just log it. In real app, we'd fire callback.
            epoch = self.manager.sessions[peer_id].epoch
            self.sink.ingest_event({
                "event_type": "LLM_RES_RECV", 
                "epoch": epoch,
                "tokens": obj["payload"]["tokens"], 
                "coherence": obj["payload"]["coherence"]
            })
            return pt

        else:
            epoch = self.manager.sessions[peer_id].epoch
            self.sink.ingest_event({"event_type": "UNKNOWN_KIND", "state": "DROP", "epoch": epoch, "peer": peer_id})
            return None
