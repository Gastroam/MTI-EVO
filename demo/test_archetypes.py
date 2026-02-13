import requests
import json
import time

print("🔍 Testing /api/dreams/archetypes...")
try:
    # 1. Standard Request
    t0 = time.time()
    res = requests.get('http://localhost:8800/api/dreams/archetypes')
    latency = (time.time() - t0) * 1000
    
    print(f"⏱️ Latency: {latency:.2f}ms")
    print(f"status: {res.status_code}")
    
    if res.status_code == 200:
        data = res.json()
        print(f"\n🧩 Recieved {len(data.get('archetypes', []))} Archetypes:")
        for arch in data['archetypes']:
            print(f"   [{arch['id']}] {arch['name']} (Count: {arch['count']})")
            print(f"      Anxiety: {arch['avg_anxiety']} | Vividness: {arch['avg_vividness']}")
            print(f"      Sample: {arch['sample_text']}")
    else:
        print(res.text)

except Exception as e:
    print(f"❌ Failed: {e}")
