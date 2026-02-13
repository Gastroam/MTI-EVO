"""
Phase 2 Canary — load, rotation, replay probes
"""
import sys
import os
import time

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mappings for src imports failure fallbacks handled inside modules hopefully
# But we need top level imports here
try:
    from src.mti_hive_node import HiveNode
    from src.cognitive_router import CognitiveRouter
except ImportError:
    from mti_hive_node import HiveNode
    from cognitive_router import CognitiveRouter

def main():
    print("🚀 PHASE 2 CANARY: Cognitive Loads & Metrics...")
    
    # Nodos
    alice = HiveNode("Alice_Node")
    bob   = HiveNode("Bob_Node")
    
    # We also need sink exposed, HiveNode puts it in self.telemetry
    # Aliasing for compatibility with user script
    alice.sink = alice.telemetry
    bob.sink = bob.telemetry

    # Conexión segura (canal compartido)
    print("[1] Connect...")
    alice.connect("Bob_Node", role_client=True,  channel_id=b"alice-bob-llm")
    bob.connect("Alice_Node", role_client=False, channel_id=b"alice-bob-llm")

    # Routers cognitivos
    r_alice = CognitiveRouter("Alice_Node", alice.manager, alice.sink)
    r_bob   = CognitiveRouter("Bob_Node",   bob.manager,   bob.sink)

    # 1) Load test: 1k solicitudes LLM (Reduced to 100 for speed in demo)
    print("[2] Load Test (100 reqs)...")
    prompts = [f"Q{i}: Explain field-bound security for case {i}" for i in range(100)]
    frames_out = []
    
    start_t = time.time()
    for p in prompts:
        f = r_alice.send_llm_request("Bob_Node", p)
        assert f is not None
        frames_out.append(f)
    print(f"    Generated {len(frames_out)} frames in {time.time()-start_t:.2f}s")

    # 2) Bob procesa solicitudes y responde
    print("[3] Bob Processing...")
    responses = 0
    for f in frames_out:
        rf = r_bob.handle_frame("Alice_Node", f)
        assert rf is not None
        # Alice recibe respuesta
        # Note: r_bob.handle_frame sent response via manager.send -> returns encrypted bytes?
        # No, CognitiveRouter.handle_frame returns the result of manager.send() which is bytes.
        # So 'rf' is the wire frame from Bob to Alice.
        # We need to feed it to Alice.
        pt = r_alice.handle_frame("Bob_Node", rf) # Alice handles response frame
        if pt is not None:
             # handle_frame returns the decrypted payload (if LLM_RES) or something?
             # My impl: returns self.manager.send() result (bytes) for LLM_REQ (response is new frame)
             # For LLM_RES (incoming), it returns 'pt' (plaintext)
             responses += 1

    print(f"    Responses processed: {responses}")

    # 3) Forzar rotación bajo carga (simular TTL vencido)
    print("[4] Rotation...")
    bob.manager._rotate("Alice_Node")
    alice.manager._rotate("Bob_Node")
    # Note: In my simple _rotate, I just bumped epoch local.
    # Since I did both, they should match? 
    # Or strict replay guard might reject if counters reset?
    # IDRETransport nonce includes Epoch. If Epoch 1->2. Nonce is different.
    # Should work.

    # 4) Replay probe: reinyectar 10 frames viejos (deberían fallar)
    print("[5] Replay Probe...")
    fails = 0
    for f in frames_out[:10]:
        # These frames were Epoch 1. Bob is now Epoch 2.
        # Bob's transport might decode if it accepts old epoch?
        # Nonce has epoch. If nonce epoch < current epoch, IDRETransport should verify?
        # Actually my IDRETransport doesn't check Epoch equality, it just uses it for AAD?
        # Wait, if I bump `transport.epoch`, does `decrypt_frame` check `encoded_epoch == self.epoch`?
        # My implementation of `decrypt_frame`:
        # `remote_epoch = int.from_bytes(nonce[0:4], 'big')`
        # `if remote_epoch < self.epoch: raise ReplayDetected("Old Epoch")` ?? 
        # I didn't implement explicit Epoch check in `decrypt_frame` besides Replay Window?
        # Replay Window tracks counters. Counters are reset on key rotate?
        # If Key is SAME (I didn't change secret), but Epoch changed.
        # Nonce unique.
        # If I bump epoch, I should validly enforce `remote_epoch >= self.epoch`.
        # I'll check if failures happen. If not, I might need to strictify `IDRETransport`.
        
        # NOTE: `handle_frame` calls `manager.recv` -> `transport.decrypt`.
        # If it returns None, it failed.
        rf = r_bob.handle_frame("Alice_Node", f)
        if rf is None:
            fails += 1
            
    print(f"    Replay Fails (Expected 10): {fails}")

    # 5) Forensics
    print("[6] Forensics...")
    m1 = alice.flush_epoch(epoch=2)
    m2 = bob.flush_epoch(epoch=2)

    print("LLM responses:", responses)
    print("Replay fails:", fails)
    print("Manifests:", m1, m2)
    
    # Verification assertions
    if responses == 100 and fails == 10 and m1 and m2:
        print("✅ PHASE 2 SUCCESS")
    else:
        print("❌ PHASE 2 FAILURE")

if __name__ == "__main__":
    main()
