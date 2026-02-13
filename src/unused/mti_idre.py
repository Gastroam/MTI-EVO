"""
MTI-EVO IDRE Channel (Hardened v2.1)
======================================
Implements IDRE-Angular Streaming Protocol (v2.1).
Security Level: Max/Hardened (AEAD, HKDF, ReplayGuards, ForensicLogs)
"""

import os
import sys
import time
import struct
import json
import hashlib
import binascii
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from enum import Enum, IntEnum

# Cryptography Primitives (Hardened)
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.exceptions import InvalidTag

# --- Constants & Enums ---

PROTOCOL_VERSION = 0x21  # v2.1
NONCE_SIZE = 12
TAG_SIZE = 16
HEADER_SIZE = 1 + 1 + 1 + 2 + NONCE_SIZE  # Ver(1)+Type(1)+Flags(1)+Len(2)+Nonce(12) = 17 bytes
SALT_SIZE = 32
KEY_SIZE = 32
REPLAY_WINDOW_SIZE = 1024
MAX_PAYLOAD_SIZE = 65535

class PacketType(IntEnum):
    HANDSHAKE = 0x01
    DATA = 0x02
    ROTATE = 0x03
    CLOSE = 0x04
    KEEPALIVE = 0x05

class PacketFlags(IntEnum):
    NONE = 0x00
    COMPRESSED = 0x01
    RESYNC = 0x02
    PRIORITY = 0x04

class IDREError(Exception):
    pass

class AuthFailure(IDREError): pass
class ReplayDetected(IDREError): pass
class NonceReuse(IDREError): pass
class FrameMalformed(IDREError): pass
class VersionMismatch(IDREError): pass
class KeyExpired(IDREError): pass


# --- Telemetry & Audit ---

@dataclass
class SecurityEvent:
    event_id: str
    timestamp: float
    event_type: str
    cause: str
    state_hash: str
    metadata: Dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

from .telemetry import log_event as capture_system_event

class IDRETelemetry:
    def __init__(self, log_file: str = "idre_audit.jsonl"):
        self.log_file = log_file
        self.logger = logging.getLogger(f"IDRE_AUDIT_{os.urandom(4).hex()}") # Unique logger per instance
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers if logger reused (though we randomize name now)
        if not self.logger.handlers:
            self.handler = logging.FileHandler(self.log_file)
            self.handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(self.handler)
        else:
            self.handler = self.logger.handlers[0]

    def log_event(self, event_type: str, cause: str, state: str, metadata: Dict = None):
        event = SecurityEvent(
            event_id=binascii.hexlify(os.urandom(8)).decode(),
            timestamp=time.time(),
            event_type=event_type,
            cause=cause,
            state_hash=hashlib.sha256(state.encode()).hexdigest(),
            metadata=metadata or {}
        )
        self.logger.info(event.to_json())
        
        # [PHASE 5] Pipe to Central Telemetry
        capture_system_event("security", f"IDRE: {event_type} - {cause}", data=asdict(event))
        
        return event

    def close(self):
        """Release file handles."""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# --- Core Protocol Types ---

class ReplayGuard:
    """Sliding Window Replay Protection."""
    def __init__(self, window_size: int = REPLAY_WINDOW_SIZE):
        self.window_size = window_size
        self.max_seen = -1
        self.window = 0 # Bitmap for checking packets within window [max_seen - window_size, max_seen]

    def check_and_update(self, counter: int) -> bool:
        if counter <= self.max_seen:
            # Check if too old
            diff = self.max_seen - counter
            if diff >= self.window_size:
                return False # Too old, replay
            
            # Check bitmap
            if (self.window >> diff) & 1:
                return False # Should not handle replay, already seen
            
            # Update bitmap
            self.window |= (1 << diff)
            return True
        else:
            # New max
            diff = counter - self.max_seen
            if diff >= self.window_size:
                self.window = 1 # Shift out everything
            else:
                self.window <<= diff
                self.window |= 1
            
            self.max_seen = counter
            return True

class IDREHandshake:
    def __init__(self, shared_secret: bytes, role: str = "client", channel_id: int = 1):
        self.shared_secret = shared_secret
        self.role = role
        self.channel_id = channel_id
        self.salt = os.urandom(SALT_SIZE)
        
    def derive_keys(self, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive (client_key, server_key) using HKDF-SHA256."""
        if salt:
            self.salt = salt
            
        def derive_for_context(ctx_label: str) -> bytes:
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=KEY_SIZE,
                salt=self.salt,
                info=ctx_label.encode(),
            )
            return hkdf.derive(self.shared_secret)

        c_ctx = f"IDRE/v2.1:handshake:client:{self.channel_id}"
        s_ctx = f"IDRE/v2.1:handshake:server:{self.channel_id}"
        
        client_key = derive_for_context(c_ctx)
        server_key = derive_for_context(s_ctx)
        
        return client_key, server_key

class IDRETransport:
    def __init__(self, client_key: bytes, server_key: bytes, channel_id: int = 1, role: str = "client"):
        self.channel_id = channel_id
        self.role = role
        
        if role == "client":
            self.tx_key = client_key
            self.rx_key = server_key
        elif role == "server":
            self.tx_key = server_key
            self.rx_key = client_key
        else:
            raise ValueError("Role must be 'client' or 'server'")

        self.tx_cipher = ChaCha20Poly1305(self.tx_key)
        self.rx_cipher = ChaCha20Poly1305(self.rx_key)
        
        # Nonce Discipline
        self.epoch = int(time.time())
        self.write_counter = 0
        self.read_guard = ReplayGuard()
        
        self.telemetry = IDRETelemetry()
        self.state = "ACTIVE"

    def _make_nonce(self, counter: int) -> bytes:
        # 12 bytes: Epoch(4) | Channel(4) | Counter(4)
        return struct.pack(">III", self.epoch, self.channel_id, counter)

    def encrypt_frame(self, data: bytes, packet_type: PacketType = PacketType.DATA, flags: PacketFlags = PacketFlags.NONE) -> bytes:
        if self.write_counter > 0xFFFFFFFF:
             raise KeyExpired("Counter wraparound imminent, rotate key")
        
        # Prepare Nonce
        nonce = self._make_nonce(self.write_counter)
        
        # Construct Header for AAD (Authenticated Additional Data)
        # Header: [Ver:1][Type:1][Flags:1][Len:2][Nonce:12]
        payload_len = len(data)
        if payload_len > MAX_PAYLOAD_SIZE:
            raise ValueError("Payload too large")

        header_no_nonce = struct.pack(">BBBH", PROTOCOL_VERSION, packet_type, flags, payload_len)
        header = header_no_nonce + nonce
        
        # Encrypt with TX Cipher
        ciphertext = self.tx_cipher.encrypt(nonce, data, header)
        
        frame = header + ciphertext
        
        self.write_counter += 1
        return frame

    def decrypt_frame(self, frame_bytes: bytes) -> Tuple[int, int, bytes]:
        """Returns (type, flags, payload)."""
        if len(frame_bytes) < HEADER_SIZE + TAG_SIZE:
             self.telemetry.log_event("FRAME_ERROR", "Frame too short", self.state)
             raise FrameMalformed("Frame too short")
             
        # Parse Header
        ver, p_type, flags, p_len = struct.unpack(">BBBH", frame_bytes[:5])
        nonce = frame_bytes[5:17]
        ciphertext_with_tag = frame_bytes[17:]
        
        # 1. Version Check
        if ver != PROTOCOL_VERSION:
            self.telemetry.log_event("VERSION_MISMATCH", f"Got {ver}", self.state)
            raise VersionMismatch(f"Expected {PROTOCOL_VERSION}, got {ver}")
            
        # 2. Extract Counter from Nonce
        remote_epoch, remote_channel, remote_counter = struct.unpack(">III", nonce)
        
        if remote_channel != self.channel_id:
             self.telemetry.log_event("CHANNEL_MISMATCH", f"{remote_channel} != {self.channel_id}", self.state)
             pass 

        # 3. Replay Check
        if not self.read_guard.check_and_update(remote_counter):
            self.telemetry.log_event("REPLAY_DETECTED", f"Counter {remote_counter} replayed/old", self.state)
            raise ReplayDetected(f"Counter {remote_counter} rejected")

        # 4. Decrypt with RX Cipher
        header = frame_bytes[:HEADER_SIZE]
        try:
            plaintext = self.rx_cipher.decrypt(nonce, ciphertext_with_tag, header)
        except InvalidTag:
            self.telemetry.log_event("AUTH_FAILURE", "Tag mismatch", self.state)
            raise AuthFailure("Integrity check failed")
            
        return p_type, flags, plaintext

    def rotate_key(self, new_secret: bytes):
        """Rotate key and increment epoch."""
        handshake = IDREHandshake(new_secret, self.role, self.channel_id)
        # Assuming rotation derives new keys for both directions
        c_key, s_key = handshake.derive_keys()
        
        if self.role == "client":
            self.tx_key = c_key
            self.rx_key = s_key
        else:
            self.tx_key = s_key
            self.rx_key = c_key
            
        self.tx_cipher = ChaCha20Poly1305(self.tx_key)
        self.rx_cipher = ChaCha20Poly1305(self.rx_key)
        
        self.epoch += 1
        self.write_counter = 0
        self.read_guard = ReplayGuard()
        self.telemetry.log_event("KEY_ROTATE", f"New Epoch {self.epoch}", self.state)

