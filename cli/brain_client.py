"""
Brain Client (MoE Bridge Edition)
=================================
Adapts the VMT-CLI to talk to the MTI-Bridge (HTTP) instead of the RLM Server (WebSocket).
"""
import asyncio
import json
import os
from typing import AsyncGenerator, Optional, Dict, Any, List
import httpx

class BrainClient:
    """HTTP client for MTI-MoE Bridge."""
    
    def __init__(self, host: str = "localhost", port: int = 8766):
        self.base_url = f"http://{host}:{port}/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def connect(self) -> bool:
        """Verify connection to bridge."""
        try:
            # Simple health check (using parietal telemetry as ping)
            await self.client.post(f"{self.base_url}/parietal/telemetry", json={"payload": {}})
            return True
        except Exception:
            return False
    
    async def disconnect(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def _post(self, endpoint: str, payload: Dict = None) -> Dict:
        """Send POST request to bridge."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/{endpoint}", 
                json={"payload": payload or {}}
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as e:
            return {"error": f"Bridge Error: {e}"}
        except Exception as e:
            return {"error": str(e)}

    # ========================================================================
    # HIGH-LEVEL API (Mapped to Bridge Endpoints)
    # ========================================================================
    
    async def get_brain_status(self) -> Dict:
        """Get status of all 9 lobes (Mocked or aggregated)."""
        # In MoE mode, we might just return "active" for all mapped lobes
        return {
            "lobes": {
                "hippocampus": {"active": True, "activation": 1.0, "info": "Bridge Connected"},
                "wernicke": {"active": True, "activation": 1.0, "info": "Ready"},
                "parietal": {"active": True, "activation": 1.0, "info": "Monitoring"},
                "occipital": {"active": True, "activation": 1.0, "info": "Scanning"},
            }
        }
    
    async def query(self, prompt: str, depth: int = 1) -> Dict:
        """
        In MoE mode, the CLI shouldn't really be used for 'query' since Antigravity IS the thinker.
        But we can route it to a simple completion if needed, or return a meta-message.
        """
        return {"text": "⚠️ In MoE Mode, Antigravity (User) is the Agent. Use 'vmt search' or other tools directly."}
    
    async def query_stream(self, prompt: str, depth: int = 1) -> AsyncGenerator[Dict, None]:
        """Streaming not supported in simple bridge mode yet."""
        yield {"type": "error", "error": "Streaming not supported in MoE Bridge mode"}
    
    async def hippocampus_search(self, query: str, limit: int = 5) -> Dict:
        """Search vector memory."""
        # Maps to /v1/hippocampus/recall
        res = await self._post("hippocampus/recall", {"query": query, "limit": limit})
        if "results" in res:
             return {"memories": res["results"]}
        return res
    
    async def homeostasis_health(self, path: str = ".") -> Dict:
        """Get code health analysis."""
        return await self._post("homeostasis/scan", {"path": path})
    
    async def parietal_telemetry(self) -> Dict:
        """Get hardware telemetry."""
        return await self._post("parietal/telemetry")
    
    async def wernicke_index(self, path: str = ".", force: bool = False) -> Dict:
        """Index codebase."""
        # Maps to /v1/wernicke/graph (or we need an index endpoint)
        # Bridge.py didn't implement 'index' explicitly, let's assume 'graph' or add it.
        # Actually bridge.py needs an index endpoint. I'll map to graph for now or fail gracefully.
        return {"error": "Indexing not exposed in Bridge v1"}
    
    async def occipital_verify(self, file: str) -> Dict:
        """Verify file AST."""
        return await self._post("occipital/scan", {"file": file})

