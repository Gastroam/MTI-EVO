import os
import hashlib
import sys

EXPECTED_HASH = '71508d63c89fdb6d5556c9bd9be79127f97c7e4028f29a4e20b394f6a2d4fd5b'

def calculate_sha256(filepath):
    """Calculates the SHA-256 hash of a file."""
    try:
        with open(filepath, 'rb') as f:
            sha256_hash = hashlib.sha256()
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error calculating SHA-256: {e}")
        return None

def verify_self():
    """Verifies that the file's hash matches the expected hash."""
    filepath = __file__
    calculated_hash = calculate_sha256(filepath)

    if calculated_hash:
        if calculated_hash == EXPECTED_HASH:
            print("Self-verification successful!")
        else:
            print(f"Self-verification failed. Expected: {EXPECTED_HASH}, Calculated: {calculated_hash}")
    else:
        print("Self-verification failed due to an error.")

if __name__ == "__main__":
    verify_self()