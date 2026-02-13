import hashlib
import os

EXPECTED_HASH = '91e9b1d183d114b657390f67cd38611239393c8d245698e019031f8440cf977b'

def verify_self():
    """
    Verifies the integrity of the current Python file by comparing its SHA-256 hash
    to the expected hash stored in the EXPECTED_HASH variable.
    """
    try:
        with open(__file__, 'rb') as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Self-verification successful!")
            return True
        else:
            print(f"Self-verification failed!  Expected: {EXPECTED_HASH}, Actual: {sha256_hash}")
            return False
    except Exception as e:
        print(f"Error during self-verification: {e}")
        return False

if __name__ == '__main__':
    verify_self()