
import pandas as pd
import faiss
import numpy as np
import os

# Configuration anti-conflit
os.environ["OMP_NUM_THREADS"] = "1"

class HealthcareSearch:
    def __init__(self, mapping_path, index_path):
        # Chargement direct
        self.df = pd.read_csv(mapping_path)
        self.index = faiss.read_index(index_path)
        
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(1)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
    def search(self, query, k=4):
        query_lower = query.lower()
        
        # 1. Identifier le médicament mentionné
        all_meds = self.df['medicament'].unique()
        target_med = next((med for med in all_meds if med.lower() in query_lower), None)
        
        # 2. Embedding de la query
        import torch
        with torch.no_grad():
            vector = self.model.encode([query]).astype(np.float32)
        
        distances, indices = self.index.search(vector, 100 if target_med else k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1: continue
            row = self.df.iloc[idx]
            
            if target_med and row['medicament'] != target_med:
                continue
                
            results.append({
                "medicament": row["medicament"],
                "avis": row["avis_for_embedding"],
                "score": float(dist)
            })
            if len(results) >= k: break 
            
        return results
