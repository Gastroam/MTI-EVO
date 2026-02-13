"""
IDRE Protocol v2.1 Verification Suite
=====================================
Strict validation of IDRE hardening:
1. Handshake Derivation (HKDF)
2. Transport Security (AEAD, Framing)
3. Replay Protection (Sliding Window)
4. Telemetry & Audit
"""

import os
import sys
import unittest
import struct
import json
import logging
import time

# Adjust import path to include src directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mti_idre
from mti_idre import (
    IDREHandshake, IDRETransport, IDRETelemetry, 
    PacketType, PacketFlags, 
    AuthFailure, ReplayDetected, FrameMalformed, VersionMismatch
)
# Constants
from mti_idre import KEY_SIZE, SALT_SIZE

class TestIDREProtocol(unittest.TestCase):
    
    def setUp(self):
        self.shared_secret = b"SUPER_SECRET_PSI_CORE_KEY_007"
        self.channel_id = 1
        
        # Setup Handshake
        self.client_hs = IDREHandshake(self.shared_secret, role="client", channel_id=self.channel_id)
        self.server_hs = IDREHandshake(self.shared_secret, role="server", channel_id=self.channel_id)
        
        # Derive Keys (Simulate exchange of salt)
        self.salt = self.client_hs.salt
        
        # Both derive the SAME set of directional keys given the same salt
        # derive_keys returns (client_key, server_key)
        self.c_client_key, self.c_server_key = self.client_hs.derive_keys(self.salt)
        self.s_client_key, self.s_server_key = self.server_hs.derive_keys(self.salt)
        
        # Setup Transport
        # Client sends with ClientKey, Reads with ServerKey
        self.client_tx = IDRETransport(self.c_client_key, self.c_server_key, self.channel_id, role="client")
        
        # Server sends with ServerKey, Reads with ClientKey
        self.server_rx = IDRETransport(self.s_client_key, self.s_server_key, self.channel_id, role="server")

    def tearDown(self):
        # Cleanup Telemetry handles
        if hasattr(self.client_tx, 'telemetry'): self.client_tx.telemetry.close()
        if hasattr(self.server_rx, 'telemetry'): self.server_rx.telemetry.close()

    def test_handshake_derivation(self):
        """Verify HKDF derivation is deterministic and context-bound."""
        print("\n[TEST] Handshake Derivation")
        
        # Keys must match between peers
        self.assertEqual(self.c_client_key, self.s_client_key)
        self.assertEqual(self.c_server_key, self.s_server_key)
        
        # Client key != Server key (Context separation)
        self.assertNotEqual(self.c_client_key, self.c_server_key)
        print(f"  Key Match & Separation: OK")

    def test_transport_aead(self):
        """Verify AEAD Encryption and Integrity."""
        print("\n[TEST] Transport AEAD")
        
        payload = b"NEURAL_LINK_ACTIVE: GAMMA_WAVE_SYNC_100%"
        
        # Encrypt (Client -> Server)
        frame = self.client_tx.encrypt_frame(payload, PacketType.DATA)
        print(f"  Frame Size: {len(frame)} bytes")
        
        # Decrypt (Server receives)
        ptype, flags, msg = self.server_rx.decrypt_frame(frame)
        self.assertEqual(msg, payload)
        self.assertEqual(ptype, PacketType.DATA)
        print("  Decryption (C->S): OK")
        
        # Reverse (Server -> Client)
        s_payload = b"ACK_SYNC"
        s_frame = self.server_rx.encrypt_frame(s_payload, PacketType.DATA)
        ptype, flags, msg = self.client_tx.decrypt_frame(s_frame)
        self.assertEqual(msg, s_payload)
        print("  Decryption (S->C): OK")
        
        # Tamper Check (Bit flip in Ciphertext)
        # Use a NEW frame to avoid Replay Detection masking the Auth Failure
        frame_2 = self.client_tx.encrypt_frame(payload, PacketType.DATA)
        tampered_frame = frame_2[:-1] + bytes([frame_2[-1] ^ 0xFF]) # Flip last bit of validation tag
        
        with self.assertRaises(AuthFailure):
            self.server_rx.decrypt_frame(tampered_frame)
        print("  Tamper Resistance: OK (AuthFailure on tag mod)")
        
        # Header Tamper (Flip Type)
        # Use another new frame
        frame_3 = self.client_tx.encrypt_frame(payload, PacketType.DATA)
        header_byte_1 = frame_3[1]
        tampered_header = frame_3[:1] + bytes([header_byte_1 ^ 0x01]) + frame_3[2:]
        
        with self.assertRaises(AuthFailure):
            self.server_rx.decrypt_frame(tampered_header)
        print("  Header Integrity: OK (AAD check works)")

    def test_replay_guard(self):
        """Verify Sliding Window Replay Protection."""
        print("\n[TEST] Replay Guard")
        
        payload = b"PING"
        
        # Send 3 Frames
        f1 = self.client_tx.encrypt_frame(payload) # Counter 0
        f2 = self.client_tx.encrypt_frame(payload) # Counter 1
        f3 = self.client_tx.encrypt_frame(payload) # Counter 2
        
        # Receive Normal Order
        self.server_rx.decrypt_frame(f1)
        self.server_rx.decrypt_frame(f2)
        
        # Replay f1
        with self.assertRaises(ReplayDetected):
            self.server_rx.decrypt_frame(f1)
        print("  Simple Replay: BLOCKED")
        
        # Accept f3
        self.server_rx.decrypt_frame(f3)
        print("  Sequence Continue: OK")
        
        # Out of Order (but valid windows) - Simulate packet loss/reorder
        # Skipping ahead
        # Manually advance client counter
        for _ in range(10): 
            _ = self.client_tx.encrypt_frame(payload) # Burn counters 3-12
            
        f13 = self.client_tx.encrypt_frame(payload) # Counter 13
        
        self.server_rx.decrypt_frame(f13)
        print("  Window Jump (Gap): OK")
        
        # Let's test window edge
        # Current Max: 13. Window: 1024. 
        
    def test_telemetry(self):
        """Verify Telemetry Logging."""
        print("\n[TEST] Telemetry Checks")
        
        log_file = "idre_audit_test.jsonl"
        # Temporarily redirect telemetry
        self.server_rx.telemetry.close() # Close old
        self.server_rx.telemetry = IDRETelemetry(log_file)
        
        try:
            # Trigger Auth Failure
            bad_frame = os.urandom(50) 
            try:
                self.server_rx.decrypt_frame(bad_frame)
            except:
                pass
            
            self.server_rx.telemetry.close() # Flush and close
            
            # Check log
            with open(log_file, 'r') as f:
                logs = [json.loads(line) for line in f]
                
            self.assertTrue(len(logs) > 0)
            last_event = logs[-1]
            self.assertIn(last_event['event_type'], ["FRAME_ERROR", "AUTH_FAILURE", "VERSION_MISMATCH"])
            print("  Security Event Logged: OK")
            
        finally:
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except PermissionError:
                    print("  Warning: Could not remove test log file (still locked?)")


if __name__ == '__main__':
    unittest.main()
