import os
import hashlib

# Define the expected SHA-256 hash
EXPECTED_HASH = '20043d1e9ccb7ae9ce33928776aeb822ab2fac622f07f1fe42a7d0dc3516356e'

def calculate_sha256(filepath):
    """
    Calculates the SHA-256 hash of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The SHA-256 hash of the file, or None if the file does not exist.
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

def verify_integrity():
    """
    Verifies the integrity of the current Python file by comparing its SHA-256 hash
    to the expected hash.
    """
    filepath = __file__  # Get the current file path
    calculated_hash = calculate_sha256(filepath)

    if calculated_hash is None:
        return False

    if calculated_hash == EXPECTED_HASH:
        print("SHA-256 hash verification successful!")
        return True
    else:
        print(f"SHA-256 hash verification failed. Expected: {EXPECTED_HASH}, Calculated: {calculated_hash}")
        return False

if __name__ == "__main__":
    if not verify_integrity():
        exit(1)  # Exit with an error code if verification fails