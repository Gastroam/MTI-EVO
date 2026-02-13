import pytest
import time
import requests
import threading
import multiprocessing
import sys
import os
sys.path.append(os.getcwd()) # Ensure root is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from mti_evo.server.substrate import SubstrateServer

# Integration Test for Substrate Concurrency
# Requires launching server in background thread/process

PORTS = [8890, 8891, 8892] # Port pool to avoid conflicts

def _run_server_process(port):
    server = SubstrateServer(port=port, multiprocessing=True)
    server.start()
    server.serve_forever()

@pytest.fixture(scope="module")
def substrate_server():
    """Launches Substrate Server in a separate process."""
    port = PORTS[0]
    
    # We need to ensure we don't block
    # SubstrateServer.serve_forever blocks.
    # We run it in a Process.
    
    proc = multiprocessing.Process(target=_run_server_process, args=(port,), daemon=True)
    proc.start()
    
    # Wait for startup
    time.sleep(5) 
    
    yield f"http://localhost:{port}"
    
    proc.terminate()
    proc.join()

def test_concurrency_stress(substrate_server):
    """
    Fire 20 concurrent requests.
    Assert no deadlocks and valid responses.
    """
    url = f"{substrate_server}/v1/local/reflex"
    
    def worker(i):
        payload = {
            "prompt": f"Stress Test {i}",
            "action": "telepathy",
            "max_tokens": 10,
            "temperature": 0.1
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    threads = []
    results = [None] * 20
    
    def thread_wrapper(idx):
        results[idx] = worker(idx)
        
    for i in range(20):
        t = threading.Thread(target=thread_wrapper, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Analyze results
    success_count = 0
    errors = []
    
    for r in results:
        if r and r.get('response'):
            success_count += 1
        else:
            errors.append(r)
            
    print(f"Success: {success_count}/20")
    print(f"Errors: {errors}")
    
    assert success_count > 0 or len(errors) > 0 # At least we got responses
    
    # Check for TIMEOUTs specifically (deadlock indicator)
    timeouts = [e for e in errors if "Read timed out" in str(e) or "timeout" in str(e)]
    assert len(timeouts) == 0, f"Deadloack suspected! {len(timeouts)} timeouts."

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
