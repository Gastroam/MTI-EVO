import hashlib
import os

EXPECTED_HASH = 'f4765f0bbc97af946fd17ec47e1d6132b130b7750e8619d464c2c325ceb720dd'

def verify_hash():
    """
    Verifies the SHA-256 hash of the current file against a known expected hash.
    """
    try:
        file_path = os.path.abspath(__file__)  # Get the absolute path to the current file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Hash verification successful!")
        else:
            print(f"Hash verification failed. Expected: {EXPECTED_HASH}, Actual: {sha256_hash}")
    except Exception as e:
        print(f"Error during hash verification: {e}")

if __name__ == "__main__":
    verify_hash()