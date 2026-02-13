"""
Demo: three‑node federated canary (Phase 3)
"""
import time
import statistics
import json
import os
import sys

# Update Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.mti_hive_node import HiveNode
    from src.cognitive_router import CognitiveRouter
except ImportError:
    from mti_hive_node import HiveNode
    from cognitive_router import CognitiveRouter

def merge_manifests(paths, out_path):
    print(f"Merging manifests: {paths}")
    merged = {
        "cluster": "HiveCluster_ABC",
        "epoch": None,
        "nodes": [],
        "alerts": [],
        "signatures": [],
        "metrics": {
            "total_requests": 0,
            "total_responses": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_coherence": 0.0
        }
    }
    latencies = []
    coherences = []
    for p in paths:
        if not p or not os.path.exists(p):
            print(f"Warning: Manifest {p} missing.")
            continue
            
        with open(p, "r", encoding="utf-8") as f:
            # First line is manifest wrapper in my implementation
            first_line = f.readline()
            try:
                m_wrapper = json.loads(first_line)
                m = m_wrapper.get("_manifest", m_wrapper) # Support wrapped or direct
            except:
                print(f"Error parsing manifest {p}")
                continue
                
        merged["nodes"].append(m.get("node_id"))
        merged["alerts"].extend(m.get("alerts", []))
        merged["signatures"].append(m.get("signature")) # Changed 'sig' to 'signature' to match sink
        
        # optional metrics file alongside manifest: node_metrics.json
        # My Sink produces: bundle_ID_metrics.json (replaced .jsonl with _metrics.json)
        # Verify the path logic
        mp = p.replace(".jsonl", "_metrics.json")
        
        if os.path.exists(mp):
            with open(mp, "r", encoding="utf-8") as mf:
                mm = json.load(mf)
            merged["metrics"]["total_requests"] += mm.get("total_requests", 0)
            merged["metrics"]["total_responses"] += mm.get("total_responses", 0)
            latencies.extend(mm.get("latencies_ms", []))
            coherences.extend(mm.get("coherences", []))
        
        if merged["epoch"] is None:
            merged["epoch"] = m.get("epoch")

    if latencies:
        s = sorted(latencies)
        merged["metrics"]["p50_latency_ms"] = s[len(s)//2]
        merged["metrics"]["p95_latency_ms"] = s[int(len(s)*0.95)]
    else:
        # Avoid crash on empty
        merged["metrics"]["p50_latency_ms"] = 0
        merged["metrics"]["p95_latency_ms"] = 0
        
    if coherences:
        merged["metrics"]["avg_coherence"] = sum(coherences) / len(coherences)

    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(merged, out, indent=2)
    return out_path

def main():
    print("🚀 PHASE 3 CANARY: Federation & Merged Forensics...")
    
    # Nodos
    alice   = HiveNode("Alice_Node")
    bob     = HiveNode("Bob_Node")
    charlie = HiveNode("Charlie_Node")

    # Access Sinks
    alice.sink = alice.telemetry
    bob.sink = bob.telemetry
    charlie.sink = charlie.telemetry

    # Conexiones seguras (dos dominios: AB y BC)
    # Using 'connect' alias which uses STATIC_SALT_PHASE2 if client.
    # But wait, 'channel_id' is used in my alias to derive deterministic integer.
    # Does 'STATIC_SALT' ignore 'channel_id'?
    # My connect alias impl:
    # if role == "client": hs = IDREHandshake(..., channel_id=cid); derive_keys(salt=STATIC_SALT)
    # So Client uses CID and Static Salt.
    # Server (else branch) uses CID and Static Salt.
    # So if CID differs, Keys differs.
    # Alice-Bob use "domain-ab". Bob-Charlie use "domain-bc".
    # Keys should be domain specific. Correct.
    
    print("[1] Federation Setup (Domains AB, BC)...")
    alice.connect("Bob_Node",     role_client=True,  channel_id=b"domain-ab")
    bob.connect("Alice_Node",     role_client=False, channel_id=b"domain-ab")

    bob.connect("Charlie_Node",   role_client=True,  channel_id=b"domain-bc")
    charlie.connect("Bob_Node",   role_client=False, channel_id=b"domain-bc")

    # Routers cognitivos
    r_alice   = CognitiveRouter("Alice_Node",   alice.manager,   alice.sink)
    r_bob     = CognitiveRouter("Bob_Node",     bob.manager,     bob.sink)
    r_charlie = CognitiveRouter("Charlie_Node", charlie.manager, charlie.sink)

    # Carga: Alice → Bob (AB) y Bob → Charlie (BC)
    # Reduced count for speed
    count = 50 
    prompts_ab = [f"AB:{i} Explain federated rotation safety {i}" for i in range(count)]
    prompts_bc = [f"BC:{i} Describe replay guard across domains {i}" for i in range(count)]

    latencies = []
    # responses counter
    responses = 0

    print(f"[2] Traffic Gen (AB: {count}, BC: {count})...")
    
    # Enviar y procesar AB
    for p in prompts_ab:
        t0 = time.perf_counter()
        f = r_alice.send_llm_request("Bob_Node", p) # A -> B
        
        # Bob receives from Alice
        rf = r_bob.handle_frame("Alice_Node", f) # B processing
        
        # B sends response to A (rf is the response frame from B->A)
        # Alice processes response
        # Note: handle_frame returns BYTES (encrypted frame) usually.
        # My cognitive logger returns decrypted payload for RES?
        # Let's check cognitive_router.py again.
        # handle_frame logic:
        # LLM_REQ -> returns self.manager.send() (BYTES) [Response Frame]
        # LLM_RES -> returns pt (PLAINTEXT)
        
        # So 'rf' is the Response Frame (Bytes) from Bob.
        # Alice needs to decrypt it.
        # But 'alice.manager.recv' takes 'frame'.
        # r_alice.handle_frame calls manager.recv.
        
        # WAIT. 'rf' from r_bob.handle_frame is Encrypted Frame B->A?
        # send_llm_request -> returns Encrypted Frame A->B.
        # r_bob.handle_frame(A, f) -> Decrypts f. Infers. Sends Response. Returns Encrypted Frame B->A.
        # So 'rf' is B->A frame.
        
        # Alice handles it.
        pt = r_alice.handle_frame("Bob_Node", rf) # Alice decrypts B->A
        if pt is not None:
             responses += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    # Enviar y procesar BC
    for p in prompts_bc:
        t0 = time.perf_counter()
        f = r_bob.send_llm_request("Charlie_Node", p) # B -> C
        rf = r_charlie.handle_frame("Bob_Node", f) # C processing, returns C->B frame
        pt = r_bob.handle_frame("Charlie_Node", rf) # B decrypts
        if pt is not None:
            responses += 1
        latencies.append((time.perf_counter() - t0) * 1000)

    # Rotación federada (elders bump): sincronizar epoch en AB y BC
    print("[3] Federated Rotation...")
    # Node managers rotate per peer
    bob.manager._rotate("Alice_Node")
    alice.manager._rotate("Bob_Node")
    
    bob.manager._rotate("Charlie_Node")
    charlie.manager._rotate("Bob_Node")

    # Replay cross‑domain: intentar reinyectar frames AB en BC (deben fallar)
    print("[4] Cross-Domain Replay Attack...")
    fails = 0
    
    # Generate new frames from A->B (Epoch 2)
    # Try to feed them to C as if coming from B? 
    # Or feed them to C as if coming from A? (C doesn't know A)
    # The attacks says: "reinyectar frames AB en BC"
    # Scenario: Attacker captures A->B and sends to C claiming to be B?
    # Or sends to C on B-C link.
    
    test_prompts = prompts_ab[:10]
    for p in test_prompts:
        # Alice generates frame for Bob (Domain AB, Key AB)
        f_ab = r_alice.send_llm_request("Bob_Node", p) 
        
        # Inject into Charlie (expecting from Bob)
        # Charlie tries to decrypt using Domain BC Key
        # Should fail Auth (Key Mismatch)
        
        # C handles frame claiming sender is "Bob_Node"
        # C uses B-C key. f_ab encrypt with A-B key.
        rf = r_charlie.handle_frame("Bob_Node", f_ab)
        if rf is None:
            fails += 1
            
    print(f"    Cross-Domain Replay Fails (Expected 10): {fails}")

    # Forensics por nodo (epoch=4)
    print("[5] Merged Forensics...")
    m_alice   = alice.flush_epoch(epoch=4)
    m_bob     = bob.flush_epoch(epoch=4)
    m_charlie = charlie.flush_epoch(epoch=4)
    
    # Note: flushing epoch 4? We are at epoch 2 (started 1, rotated once).
    # But flush_epoch just takes an epoch ID for labelling. 4 is fine.
    
    manifest_paths = [m_alice, m_bob, m_charlie]
    
    # Fusión de manifiestos
    merged_path = "cluster_ABC_merged_manifest.json"
    merge_manifests(manifest_paths, merged_path)

    # Reporte
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0.0
    print(f"Responses={responses}/{count*2}  ReplayCrossDomainFails={fails}  p50={p50:.2f}ms  p95={p95:.2f}ms")
    # check merged
    if os.path.exists(merged_path):
        with open(merged_path, 'r') as f:
            d = json.load(f)
            print("Merged Stats:", d.get("metrics"))

    if responses == 100 and fails == 10:
        print("✅ PHASE 3 SUCCESS")
    else:
        print("❌ PHASE 3 FAILURE")

if __name__ == "__main__":
    main()
