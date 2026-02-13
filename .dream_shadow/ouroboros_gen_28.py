# Solution for: # Scan for TODOs

import hashlib
import os

EXPECTED_HASH = '62fb86c98ad8287a79ae404e090c3ba4558b6b1f3fafd4e7ede78d4093b01523'

def verify_self():
    """
    Verifies that the current file's SHA-256 hash matches the expected hash.
    """
    try:
        filepath = __file__
        with open(filepath, "rb") as f:
            file_content = f.read()
            sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Self-verification successful!")
            return True
        else:
            print(f"Self-verification failed. Expected: {EXPECTED_HASH}, Actual: {sha256_hash}")
            return False
    except Exception as e:
        print(f"Error during self-verification: {e}")
        return False

if __name__ == "__main__":
    if not verify_self():
        exit(1)  # Exit with an error code if verification fails