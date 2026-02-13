
"""
Dream Analyzer (Oneiric Archetype Engine)
=========================================
Production logic for clustering dream reports into archetypes.
Integrates with LLMAdapter for semantic embeddings.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from mti_evo.llm_adapter import LLMAdapter
from mti_evo.projector import SynapticProjector

logger = logging.getLogger(__name__)

class DreamAnalyzer:
    def __init__(self, adapter: LLMAdapter = None, broca=None):
        self.adapter = adapter or LLMAdapter() # Reuse existing or create new
        self.broca = broca
        self.projector = SynapticProjector() # One global projector

    def mutate_archetype(self, archetype: Dict[str, Any]) -> str:
        """
        Inject an Archetype into Broca as an Instinct.
        
        Args:
            archetype: Dictionary from analyze_archetypes output
            
        Returns:
            Success message or Error
        """
        if not self.broca:
            return "Error: Broca not connected to DreamAnalyzer."
            
        # 1. Calculate Centroid (High Dim)
        dreams = archetype.get('dreams', [])
        saved_vectors = archetype.get('vectors', [])
        
        if not dreams:
             return "Error: Empty archetype."
             
        if saved_vectors:
            # Use cached embeddings (FAST)
            vectors = [np.array(v) for v in saved_vectors]
        else:
            # Re-embed (SLOW - Fallback)
            logger.info("⚠️ Mutation: Re-embedding dreams (Slow Path)")
            vectors = []
            for d in dreams:
                vec = self.adapter.embed(d['text'])
                vectors.append(np.array(vec))
        
        centroid = np.mean(vectors, axis=0)
        
        # 2. Project to Low Dim (64d)
        instinct_vector = self.projector.project(centroid)
        
        # 3. Create Concept Seed
        name = archetype['name']
        seed = self.broca.text_to_seed(name)
        
        # print(f"[MUTATION DEBUG] Injecting '{name}' -> Seed {seed}")
        # print(f"[MUTATION DEBUG] Cortex Size Before: {len(self.broca.cortex.active_tissue)}")

        class InstinctNeuron:
            def __init__(self, weights, bias, label):
                self.weights = weights
                self.bias = bias
                self.label = label
                self.weight_avg = float(np.mean(np.abs(weights)))
                
        # Inject into Active Tissue
        self.broca.cortex.active_tissue[seed] = InstinctNeuron(
            weights=instinct_vector,
            bias=5.0, # Strong Bias for Instincts
            label=f"[MUTATION] {name}"
        )
        
        # print(f"[MUTATION DEBUG] Cortex Size After: {len(self.broca.cortex.active_tissue)}")
        # print(f"[MUTATION DEBUG] Verified in dict: {seed in self.broca.cortex.active_tissue}")
        
        logger.info(f"🧬 Mutation Complete: Injected {name} (Seed {seed}) into Cortex.")
        return f"Success: Injected {name} (Seed {seed})."

    def analyze_archetypes(self, dreams: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a list of dreams and return Archetypes.
        """
        if not dreams:
            return []

        logger.info(f"zzZ Analyzing {len(dreams)} dreams for archetypes...")
        
        # 1. Embed
        embeddings = []
        valid_indices = []
        
        for idx, d in enumerate(dreams):
            try:
                vec = self.adapter.embed(d['text'])
                arr = np.array(vec)
                
                # Filter malformed vectors
                if len(arr.shape) == 1 and arr.shape[0] > 100:
                    embeddings.append(arr)
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Skipping malformed embedding for dream {idx}: {arr.shape}")
            except Exception as e:
                logger.error(f"Embedding failed for dream {idx}: {e}")
                
        if not embeddings:
            return []

        # 2. Similarity Matrix
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    sim_matrix[i][j] = 1.0
                else:
                    norm1 = np.linalg.norm(embeddings[i])
                    norm2 = np.linalg.norm(embeddings[j])
                    if norm1 > 0 and norm2 > 0:
                        sim_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (norm1 * norm2)
                    else:
                        sim_matrix[i][j] = 0.0

        # 3. Density Clustering (Simplified DBSCAN)
        THRESHOLD = 0.65
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

        # 4. Construct Output
        archetypes = []
        clustered_dreams = {} 
        clustered_vectors = {} # Keep vectors for later use
        
        for i, label in enumerate(labels):
            original_idx = valid_indices[i]
            d = dreams[original_idx]
            vec = embeddings[i] # This corresponds to i-th valid embedding
            
            if label not in clustered_dreams: 
                clustered_dreams[label] = []
                clustered_vectors[label] = []
                
            clustered_dreams[label].append(d)
            clustered_vectors[label].append(vec.tolist())
            
        anxiety_map = {
            "Terrified": 1.0, "Panic": 0.9, "Anxious": 0.8, 
            "Frustrated": 0.6, "Confused": 0.5, "Weird": 0.4, 
            "Curious": 0.2, "Peaceful": 0.1, "Ecstatic": 0.0, 
            "Focused": 0.2, "Amused": 0.1
        }
            
        for label in sorted(clustered_dreams.keys()):
            if label == -1: continue 
            
            group = clustered_dreams[label]
            vectors_group = clustered_vectors[label]
            
            name_candidate = group[0]['text'][:30] + "..."
            avg_vivid = float(np.mean([d.get('vividness', 0) for d in group]))
            avg_anxiety = float(np.mean([anxiety_map.get(d.get('mood', ''), 0.5) for d in group]))
            
            archetypes.append({
                "id": int(label),
                "name": f"Archetype {label+1}",
                "sample_text": name_candidate,
                "count": len(group),
                "avg_vividness": round(avg_vivid, 2),
                "avg_anxiety": round(avg_anxiety, 2),
                "dominant_moods": list(set([d.get('mood', 'Unknown') for d in group])),
                "dreams": group,
                "vectors": vectors_group # [PHASE 52] Store embeddings for cached mutation
            })
            
        return archetypes
        """
        Analyze a list of dreams and return Archetypes.
        
        Args:
            dreams: List of dicts with 'text', 'mood', 'vividness'
            
        Returns:
            List of Archetype dicts
        """
        if not dreams:
            return []

        logger.info(f"zzZ Analyzing {len(dreams)} dreams for archetypes...")
        
        # 1. Embed
        embeddings = []
        valid_indices = []
        
        for idx, d in enumerate(dreams):
            try:
                vec = self.adapter.embed(d['text'])
                arr = np.array(vec)
                
                # Filter malformed vectors
                if len(arr.shape) == 1 and arr.shape[0] > 100:
                    embeddings.append(arr)
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Skipping malformed embedding for dream {idx}: {arr.shape}")
            except Exception as e:
                logger.error(f"Embedding failed for dream {idx}: {e}")
                
        if not embeddings:
            return []

        # 2. Similarity Matrix
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j: 
                    sim_matrix[i][j] = 1.0
                else:
                    # Cosine Similarity
                    norm1 = np.linalg.norm(embeddings[i])
                    norm2 = np.linalg.norm(embeddings[j])
                    if norm1 > 0 and norm2 > 0:
                        sim_matrix[i][j] = np.dot(embeddings[i], embeddings[j]) / (norm1 * norm2)
                    else:
                        sim_matrix[i][j] = 0.0

        # 3. Density Clustering (Simplified DBSCAN)
        THRESHOLD = 0.65 # Slightly stricter for production
        MIN_CLUSTER_SIZE = 2
        labels = [-1] * n
        cluster_id = 0
        visited = [False] * n
        
        for i in range(n):
            if visited[i]: continue
            visited[i] = True
            
            neighbors = [j for j in range(n) if sim_matrix[i][j] >= THRESHOLD]
            
            if len(neighbors) < MIN_CLUSTER_SIZE:
                labels[i] = -1 # Noise
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

        # 4. Construct Output
        archetypes = []
        clustered_dreams = {} 
        
        # Group dreams by label
        for i, label in enumerate(labels):
            original_idx = valid_indices[i]
            d = dreams[original_idx]
            if label not in clustered_dreams: clustered_dreams[label] = []
            clustered_dreams[label].append(d)
            
        # Anxiety Mapping
        anxiety_map = {
            "Terrified": 1.0, "Panic": 0.9, "Anxious": 0.8, 
            "Frustrated": 0.6, "Confused": 0.5, "Weird": 0.4, 
            "Curious": 0.2, "Peaceful": 0.1, "Ecstatic": 0.0, 
            "Focused": 0.2, "Amused": 0.1
        }
            
        for label in sorted(clustered_dreams.keys()):
            if label == -1: continue # Skip noise in output for now? Or group as 'Unconscious Static'
            
            group = clustered_dreams[label]
            
            # Text Summarization (Mock: Just take first dream, or common words?)
            # Ideally we'd ask LLM to name it. For now, use the first dream's snippet.
            name_candidate = group[0]['text'][:30] + "..."
            
            avg_vivid = float(np.mean([d.get('vividness', 0) for d in group]))
            avg_anxiety = float(np.mean([anxiety_map.get(d.get('mood', ''), 0.5) for d in group]))
            
            archetypes.append({
                "id": int(label),
                "name": f"Archetype {label+1}",
                "sample_text": name_candidate,
                "count": len(group),
                "avg_vividness": round(avg_vivid, 2),
                "avg_anxiety": round(avg_anxiety, 2),
                "dominant_moods": list(set([d.get('mood', 'Unknown') for d in group])),
                "dreams": group # Include full dreams? Maybe too heavy.
            })
            
        return archetypes

