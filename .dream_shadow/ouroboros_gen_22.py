# Solution for: source="TODO",

import hashlib
import os

EXPECTED_HASH = 'd40d7c665e2a0c11b5a598479dca2fd8d2a239ac6df33f9c141b6ec7d2bdc272'

def calculate_sha256(filepath):
    """
    Calculates the SHA-256 hash of a given file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The SHA-256 hash as a hexadecimal string, or None if the file
             does not exist.
    """
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

def verify_self():
    """
    Verifies the self-contained hash by calculating the hash of the current file
    and comparing it to the EXPECTED_HASH.
    """
    filepath = __file__  # Path to the current file
    calculated_hash = calculate_sha256(filepath)

    if calculated_hash is None:
        return False

    return calculated_hash == EXPECTED_HASH

if __name__ == "__main__":
    if verify_self():
        print("Self-verification successful!")
    else:
        print("Self-verification failed!")