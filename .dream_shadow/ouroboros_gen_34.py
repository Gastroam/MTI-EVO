import hashlib
import os

# Define the expected SHA-256 hash (This is the critical part for self-verification)
EXPECTED_HASH = '0c502070242f20072eabf56302df7b5b2093d35e07e593fd827ad763a6e28e9b'

def verify_self():
    """
    Verifies the integrity of this file by hashing it and comparing the hash
    with the EXPECTED_HASH.  Returns True if the hashes match, False otherwise.
    """
    try:
        # Read the file contents
        with open(__file__, 'rb') as f:
            file_content = f.read()

        # Calculate the SHA-256 hash of the file content
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Compare the calculated hash with the expected hash
        return file_hash == EXPECTED_HASH

    except Exception as e:
        print(f"Error during self-verification: {e}")
        return False

if __name__ == "__main__":
    if verify_self():
        print("Self-verification successful! The file's hash matches the expected hash.")
    else:
        print("Self-verification failed! The file's hash does not match the expected hash.")