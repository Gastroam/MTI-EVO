
"""
Oneiric Archetype Detection (Prototype)
=======================================
Demonstrates the pipeline for clustering dream reports into archetypes
using Semantic Embeddings from the Local LLM.

Pipeline:
1. Data: Synthetic Dream Reports
2. Embedding: LLMAdapter (Gemma/Llama) -> 2048d Vectors
3. Clustering: Density-Based (simplified DBSCAN logic)
4. Post-Hoc Analysis: Archetype Naming
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mti_evo.llm_adapter import LLMAdapter

# 1. Synthetic Dream Journal
dreams = [
    # Archetype A: Flying / Freedom
    {"text": "I was flying over a vast city, the wind in my hair, feeling absolute freedom.", "mood": "Ecstatic", "vividness": 9},
    {"text": "Soaring above the clouds, looked down at the tiny ocean below. No gravity.", "mood": "Peaceful", "vividness": 8},
    {"text": "Levitating just a few feet off the ground, moving effortlessly through the streets.", "mood": "Curious", "vividness": 7},
    
    # Archetype B: Coding / Logic
    {"text": "Debugging a recursive function that kept calling itself into infinity.", "mood": "Frustrated", "vividness": 5},
    {"text": "I was inside the computer, watching data streams flow like neon rivers. I fixed a syntax error in the sky.", "mood": "Focused", "vividness": 8},
    {"text": "Writing python code on a blackboard but the chalk kept changing colors.", "mood": "Confused", "vividness": 4},
    
    # Archetype C: Anxiety / Loss
    {"text": "My teeth started crumbling and falling out one by one.", "mood": "Terrified", "vividness": 10},
    {"text": "Running down a hallway that never ends, being chased by a shadow.", "mood": "Anxious", "vividness": 8},
    {"text": "I lost my wallet and couldn't find my way home. Everything was dark.", "mood": "Panic", "vividness": 6},
    
    # Noise (Unrelated)
    {"text": "Eating a sandwich made of glass.", "mood": "Weird", "vividness": 3},
    {"text": "A giant cat was explaining quantum physics to me.", "mood": "Amused", "vividness": 7}
]

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def main():
    print("🧠 Initializing LLM Adapter (with Embeddings)...")
    adapter = LLMAdapter()
    
    # Wait for model load if concurrent
    time.sleep(2)
    
    print(f"📉 Generating Embeddings for {len(dreams)} dreams...")
    
    embeddings = []
    valid_indices = []
    
    for idx, d in enumerate(dreams):
        try:
            vec = adapter.embed(d['text'])
            arr = np.array(vec)
            print(f"   > Dream {idx}: Shape {arr.shape}")
            
            # Filter bad shapes (e.g. if one is just a token list of size 18)
            if len(arr.shape) == 1 and arr.shape[0] > 100: 
                embeddings.append(arr)
                valid_indices.append(idx)
            else:
                print(f"   ⚠️ Skipping Dream {idx} (Bad Shape: {arr.shape})")
                
        except Exception as e:
            print(f"   ❌ Error embedding dream {idx}: {e}")

    print(f"✨ Generation Complete. Valid Vectors: {len(embeddings)}")
    
    if not embeddings:
        print("No valid embeddings. Exiting.")
        return

    # 2. Similarity Matrix
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    
    print("\n🔗 Calculating Similarity Matrix...")
    for i in range(n):
        for j in range(n):
            if i == j: 
                sim_matrix[i][j] = 1.0
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                sim_matrix[i][j] = sim
                
    # 3. Density Clustering (Simplified)
    THRESHOLD = 0.60
    MIN_CLUSTER_SIZE = 2
    
    labels = [-1] * n
    cluster_id = 0
    visited = [False] * n
    
    for i in range(n):
        if visited[i]: continue
        visited[i] = True
        
        neighbors = [j for j in range(n) if sim_matrix[i][j] >= THRESHOLD]
        
        if len(neighbors) < MIN_CLUSTER_SIZE:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = [x for x in neighbors if x != i]
            while queue:
                neighbor_idx = queue.pop(0)
                if labels[neighbor_idx] == -1: labels[neighbor_idx] = cluster_id
                if labels[neighbor_idx] != -1 and labels[neighbor_idx] != cluster_id: continue
                labels[neighbor_idx] = cluster_id
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    new_neighbors = [k for k in range(n) if sim_matrix[neighbor_idx][k] >= THRESHOLD]
                    if len(new_neighbors) >= MIN_CLUSTER_SIZE:
                        queue.extend([x for x in new_neighbors if x not in queue])
            cluster_id += 1
            
    # 4. Results & Post-Hoc Analysis
    print(f"\n🧩 Clustering Complete. Found {cluster_id} Archetypes.\n")
    
    # Map valid_indices back to original dreams
    clustered_dreams = {} # Label -> List of Dreams
    
    for i, label in enumerate(labels):
        original_idx = valid_indices[i]
        d = dreams[original_idx]
        if label not in clustered_dreams: clustered_dreams[label] = []
        clustered_dreams[label].append(d)
        
    # Analyze Clusters
    for label in sorted(clustered_dreams.keys()):
        group = clustered_dreams[label]
        
        if label == -1:
            name = "NOISE (Unclassified)"
        else:
            name = f"ARCHETYPE {label+1}"
            
        # [USER REQUEST] Post-Hoc Stats Integration
        avg_vivid = np.mean([d['vividness'] for d in group])
        
        # Simulate Anxiety from Mood (Simple mapping for demo)
        anxiety_map = {"Terrified": 1.0, "Panic": 0.9, "Anxious": 0.8, "Frustrated": 0.6, "Confused": 0.5, "Weird": 0.4, "Curious": 0.2, "Peaceful": 0.1, "Ecstatic": 0.0, "Focused": 0.2, "Amused": 0.1}
        avg_anxiety = np.mean([anxiety_map.get(d['mood'], 0.5) for d in group])
        
        print(f"=== {name} ===")
        print(f"  📊 Stats: Anxiety={avg_anxiety:.2f} | Vividness={avg_vivid:.1f}")
        print("  📝 Dreams:")
        for d in group:
            print(f"    - [{d['mood']}] {d['text']}")
        print()

if __name__ == "__main__":
    main()
