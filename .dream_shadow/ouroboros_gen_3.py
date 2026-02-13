# Solution for: # Scan for TODOs

import hashlib
import os

EXPECTED_HASH = '9aec83ea0917888b03c2cf414d0f8b5980a1ffbe73d9297c41f46ebf3080c0f1'

def verify_hash():
    """
    Verifies the SHA-256 hash of the current file.

    Returns:
        True if the hash matches the expected hash, False otherwise.
    """
    try:
        filepath = os.path.abspath(__file__)  # Get the absolute path of the file
        with open(filepath, 'rb') as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        return sha256_hash == EXPECTED_HASH
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return False
    except Exception as e:
        print(f"An error occurred during hash verification: {e}")
        return False


if __name__ == "__main__":
    if verify_hash():
        print("Hash verification successful!")
    else:
        print("Hash verification failed.")