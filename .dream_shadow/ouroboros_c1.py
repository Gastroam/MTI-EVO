import hashlib
import os

def hash_self():
    """
    Calculates and returns the SHA-256 hash of the current source code.
    """
    try:
        source_code = inspect.getsource(hash_self)
        hashed_code = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
        return hashed_code
    except Exception as e:
        print(f"Error calculating hash: {e}")
        return None

if __name__ == '__main__':
    # Example usage - this will only run if the script is executed directly
    hash_value = hash_self()
    if hash_value:
        print(f"The SHA-256 hash of this script is: {hash_value}")