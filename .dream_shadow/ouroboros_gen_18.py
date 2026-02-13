import os
import hashlib
import sys

EXPECTED_HASH = 'e1cfd4ca0c1f67fc986c9b409eee54bce65b2867cc7afe7229d1ffdf240e699b'

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
        print(f"Error calculating SHA-256: {e}")
        return None

def verify_self():
    """Verifies that the file's hash matches the expected hash."""
    file_path = __file__
    calculated_hash = calculate_sha256(file_path)

    if calculated_hash is None:
        return False  # Error occurred during hash calculation

    if calculated_hash == EXPECTED_HASH:
        print("Self-verification successful!")
        return True
    else:
        print(f"Self-verification failed. Expected: {EXPECTED_HASH}, Calculated: {calculated_hash}")
        return False

if __name__ == "__main__":
    if not verify_self():
        sys.exit(1)  # Exit with an error code if verification fails