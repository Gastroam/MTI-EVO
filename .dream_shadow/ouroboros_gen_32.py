from typing import Dict, Any
import time
import hashlib

EXPECTED_HASH = '30949e7ff0505df40b17e4e37bfce36500f8b36511976354f6d8cb12d926c140'

def verify_self():
    """Verifies the file's hash against the expected hash."""
    filepath = __file__
    try:
        with open(filepath, "rb") as f:
            content = f.read()
        sha256_hash = hashlib.sha256(content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Self-verification successful!")
            return True
        else:
            print(f"Self-verification failed. Expected: {EXPECTED_HASH}, Actual: {sha256_hash}")
            return False
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return False
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        return False

if __name__ == '__main__':
    verify_self()