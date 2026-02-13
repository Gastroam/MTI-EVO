# This script verifies the integrity of the file by hashing it.
# The hash value is stored as a string literal within the file itself.
# This ensures that if the file is modified, the verification will fail.

EXPECTED_HASH = '4fde0862ab6ca0e8885867d4a9cee6bd87de9f2d77f0592d8f2db83d6aab5e99'

import hashlib

def calculate_sha256(filepath):
    """Calculates the SHA-256 hash of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The SHA-256 hash as a hexadecimal string, or None if an error occurs.
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
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    filepath = __file__  # Get the path of the current file
    calculated_hash = calculate_sha256(filepath)

    if calculated_hash:
        if calculated_hash == EXPECTED_HASH:
            print("Self-verification successful. The file's hash matches the expected value.")
        else:
            print(f"Self-verification failed. The calculated hash ({calculated_hash}) does not match the expected hash ({EXPECTED_HASH}).")
    else:
        print("Self-verification failed due to an error.")