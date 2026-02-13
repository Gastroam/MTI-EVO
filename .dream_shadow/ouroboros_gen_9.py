import hashlib

EXPECTED_HASH = '1ddd3f14fc1802642f1ff7d49b881a62032b8d5fcd7fd21f615ab1851380b837'

def verify_self():
    """
    Verifies the Python script by hashing itself and comparing it to the expected hash.
    """
    try:
        script_path = __file__
        with open(script_path, 'rb') as f:
            script_content = f.read()
        hash_object = hashlib.sha256(script_content)
        calculated_hash = hash_object.hexdigest()

        if calculated_hash == EXPECTED_HASH:
            print("Self-verification successful!")
            return True
        else:
            print(f"Self-verification failed. Expected: {EXPECTED_HASH}, Calculated: {calculated_hash}")
            return False

    except FileNotFoundError:
        print(f"Error: File not found at {script_path}")
        return False
    except Exception as e:
        print(f"An error occurred during self-verification: {e}")
        return False

if __name__ == "__main__":
    verify_self()