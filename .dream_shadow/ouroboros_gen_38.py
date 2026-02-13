import os
import hashlib
import sys

EXPECTED_HASH = '5d883fdfb5ec728ff84871565306c231fe8641eaed07f495b306f3a6761a93e6'

def calculate_sha256(file_path):
    """Calculates the SHA-256 hash of a file."""
    try:
        with open(file_path, "rb") as f:
            sha256_hash = hashlib.sha256()
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None

def verify_self():
    """Verifies the integrity of the Python file."""
    file_path = __file__
    calculated_hash = calculate_sha256(file_path)

    if calculated_hash is None:
        return False

    return calculated_hash == EXPECTED_HASH

if __name__ == "__main__":
    if verify_self():
        print("Self-verification successful!")
    else:
        print("Self-verification failed!")
        sys.exit(1)