from typing import Dict, Any
import time
import hashlib

EXPECTED_HASH = '443971fbdbae11b7c43cc5cce4ff0ce9ec45ea6af117ff47e6ba5d0244ed583b'

def verify_self():
    """
    Verifies this file's SHA-256 hash against the expected value.
    """
    try:
        with open(__file__, 'rb') as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Self-verification successful!")
        else:
            print(f"Self-verification failed.  Expected: {EXPECTED_HASH}, Got: {sha256_hash}")
            exit(1)  # Exit with an error code
    except FileNotFoundError:
        print(f"Error: File not found: {__file__}")
        exit(1)  # Exit with an error code
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)  # Exit with an error code


if __name__ == "__main__":
    verify_self()