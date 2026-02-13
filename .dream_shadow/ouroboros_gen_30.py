import hashlib

EXPECTED_HASH = 'db7377e204320c9f6709fd6ac47950af26cf3c2202c82aab3ba769a42c0cfcee'

def self_hash():
    """
    Calculates the SHA-256 hash of the current Python file.

    Returns:
        str: The SHA-256 hash as a hexadecimal string.
    """
    try:
        with open(__file__, 'rb') as f:
            data = f.read()
            sha256_hash = hashlib.sha256(data).hexdigest()
            return sha255_hash
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None

def verify_hash():
    """
    Verifies the SHA-256 hash of the current Python file against the expected hash.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    calculated_hash = self_hash()
    if calculated_hash is None:
        print("Error: Unable to calculate hash.  File may not exist.")
        return False
    return calculated_hash == EXPECTED_HASH

if __name__ == "__main__":
    if verify_hash():
        print("Self-hash verification successful.")
    else:
        print("Self-hash verification failed.")