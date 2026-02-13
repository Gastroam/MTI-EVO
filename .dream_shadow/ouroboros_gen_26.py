import hashlib
import os

EXPECTED_HASH = '9c105e0715475ae57be54bd6a16d0d90e493ed1f1e705f0d8e046a6f37392e24'

def verify_hash():
    """Verifies the SHA-256 hash of the current file."""
    filepath = __file__
    try:
        with open(filepath, "rb") as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Hash verification successful!")
            return True
        else:
            print(f"Hash verification failed. Expected: {EXPECTED_HASH}, Actual: {sha256_hash}")
            return False
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return False
    except Exception as e:
        print(f"An error occurred during hash verification: {e}")
        return False

if __name__ == "__main__":
    verify_hash()