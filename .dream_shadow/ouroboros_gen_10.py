import hashlib

EXPECTED_HASH = '751366fca1cb09bed30de71cbac358c6e957b1cd9751adf91033894746906a7e'

def verify_hash():
    """
    Verifies the integrity of the current file by calculating its SHA-256 hash
    and comparing it to the expected hash.
    """
    try:
        with open(__file__, 'rb') as f:
            file_content = f.read()
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        if sha256_hash == EXPECTED_HASH:
            print("Hash verification successful. The file has not been tampered with.")
        else:
            print(f"Hash verification failed. Expected: {EXPECTED_HASH}, Calculated: {sha256_hash}")
    except FileNotFoundError:
        print(f"Error: File '{__file__}' not found.")
    except Exception as e:
        print(f"An error occurred during hash verification: {e}")

if __name__ == "__main__":
    verify_hash()