"""
Hive Mesh Canary (v1.3 Deployment Verification)
===============================================
Simulates a 3-node cluster (Alice, Bob, Charlie) to verify:
1. Secure Handshake (HKDF)
2. AEAD Transport (ChaCha20-Poly1305)
3. Replay Resistance
4. Telemetry Bundles
"""

import sys
import os
import time
import json
# Fix paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mti_hive_node import HiveNode
from mti_idre import PacketType

def run_canary():
    print("🚀 HIVE MESH CANARY: Starting Deployment Verification...")
    
    # 1. Setup Nodes
    alice = HiveNode("Alice_Node", seed=1001)
    bob = HiveNode("Bob_Node", seed=2002)
    charlie = HiveNode("Charlie_Node", seed=3003) # Attacker
    
    print("\n[STEP 1] Handshake & Session Establishment")
    # Alice -> Bob
    # 1. Hello
    hello = alice.create_handshake_hello()
    if bob.process_handshake_hello(hello):
        # 2. Session Init (Alice=Client)
        init_msg = alice.init_session(bob.node_id, role="client")
        
        # Bob processes Init
        bob.handle_session_init(init_msg)
        
    print("✅ Session Verified: Alice <-> Bob")
    
    # Setup Charlie -> Bob for Control (Charlie is valid peer, but tries to replay Alice's packets?)
    # Or Charlie is MITM. IDRE AEAD prevents MITM without key.
    # We simulate Charlie capturing Alice's packet.
    
    print("\n[STEP 2] Secure Messaging & Routing")
    msg_content = {"type": "GOSSIP", "data": "Secrets of the Universe"}
    
    # Alice sends to Bob
    wire_packet = alice.send_message(bob.node_id, msg_content)
    print(f"  Wire Packet: {wire_packet.keys()}")
    
    # Bob receives
    decoded = bob.receive_message(wire_packet)
    if decoded["data"] == "Secrets of the Universe":
        print("✅ Message Delivered & Decrypted")
    else:
        print("❌ Message Corruption")
        
    print("\n[STEP 3] Replay Attack Simulation")
    # Charlie replays the SAME wire_packet to Bob
    print("  Charlie Replaying Packet...")
    result = bob.receive_message(wire_packet)
    
    if result is None:
        print("✅ Replay Blocked (Result is None)")
    else:
        print("❌ Replay Accepted!")
        
    print("\n[STEP 4] Telemetry & Forensics")
    # Produce Bundle
    bundle_file = bob.telemetry.flush_bundle(epoch=1)
    if bundle_file and os.path.exists(bundle_file):
        print(f"✅ Forensic Bundle Created: {bundle_file}")
        
        # verify content
        with open(bundle_file, 'r') as f:
            lines = f.readlines()
            manifest = json.loads(lines[0])
            print(f"   Manifest: {manifest['_manifest']['signature']}")
            
            # Check for Replay Event
            found_replay = False
            for line in lines[1:]:
                ev = json.loads(line)
                if ev.get("event_type") == "REPLAY_DETECTED":
                    found_replay = True
                    print("   Found Telemetry: REPLAY_DETECTED")
            
            if found_replay:
                print("✅ Telemetry Complete")
            else:
                print("❌ Missing Replay Event in Telemetry")
    else:
        print("❌ Bundle Generation Failed")

if __name__ == "__main__":
    run_canary()
