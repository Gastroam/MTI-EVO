import os
import hashlib

EXPECTED_HASH = '2dc09dea7f30af3d78ae51df2a3c9224896c757b3c3ec33fad2aff314af54df2'

def verify_self():
    """
    Verifies the integrity of the current script by hashing its contents
    and comparing it to the expected hash.
    """
    try:
        with open(__file__, 'rb') as f:
            script_content = f.read()
        calculated_hash = hashlib.sha256(script_content).hexdigest()
        if calculated_hash == EXPECTED_HASH:
            print(f"Verification successful!  Hash matches: {calculated_hash}")
            return True
        else:
            print(f"Verification failed!  Expected: {EXPECTED_HASH}, Calculated: {calculated_hash}")
            return False
    except FileNotFoundError:
        print(f"Error: File not found: {__file__}")
        return False
    except Exception as e:
        print(f"An error occurred during verification: {e}")
        return False


if __name__ == "__main__":
    if verify_self():
        print("Script integrity verified.")
    else:
        print("Script integrity not verified.  Possible corruption.")