import requests
import json
import hashlib
import sys
import os

# Add src to path for direct Broca math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from mti_evo.mti_broca import MTIBroca

def text_to_seed(token):
    hash_object = hashlib.sha256(token.encode())
    hex_dig = hash_object.hexdigest()
    return int(hex_dig[-8:], 16)

def main():
    print("🧬 Testing Synaptic Mutation...")
    
    # 1. Fetch Archetypes
    print("   > Fetching Dreams...")
    res = requests.get('http://localhost:8800/api/dreams/archetypes')
    if res.status_code != 200:
        print("❌ Failed to fetch archetypes")
        return
    
    archetypes = res.json()['archetypes']
    target = archetypes[0]
    print(f"   > Target: {target['name']} (ID {target['id']})")
    
    # 2. Trigger Mutation
    print(f"   > Injecting Mutation...")
    res = requests.post('http://localhost:8800/control/mutate', json={"id": target['id']})
    print(f"   > Status: {res.status_code}")
    print(f"   > Response: {res.json()}")
    
    # 3. Verify Neural Existence
    seed = text_to_seed(target['name'])
    print(f"   > Checking Seed {seed}...")
    
    # Probe via API
    try:
        res = requests.get(f'http://localhost:8800/api/probe?seed={seed}')
        if res.status_code == 200:
            data = res.json()
            # API returns {seed, tau, angles, responses...} or {error}
            if "error" not in data and "responses" in data:
                print(f"   ✅ SUCCESS! Neuron {seed} exists!")
                print(f"   > Label: {data.get('note', 'Instinct')}") 
            else:
                print(f"   ❌ Neuron {seed} not found (API returned error or missing data).")
                print(f"   > Payload: {data}")
        else:
            print(f"   ❌ Probe failed: {res.status_code}")
    except Exception as e:
        print(f"   ❌ Error probing: {e}")

if __name__ == "__main__":
    main()
