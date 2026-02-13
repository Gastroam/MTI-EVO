import hashlib
import os

def hash_self():
    """
    Calculates the SHA-256 hash of the current Python file.

    Returns:
        str: The SHA-256 hash as a hexadecimal string.
    """
    try:
        file_path = os.path.abspath(__file__)  # Get the absolute path of the current file
        with open(file_path, 'rb') as f:
            file_content = f.read()
            sha256_hash = hashlib.sha256(file_content).hexdigest()
            return sha256_hash
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"Error calculating hash: {e}"

if __name__ == '__main__':
    hash_value = hash_self()
    print(f"The SHA-256 hash of this file is: {hash_value}")