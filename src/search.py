
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
        import torch
        
        # Embedding
        with torch.no_grad():
            vector = self.model.encode([query]).astype(np.float32)
        
        # Recherche
        distances, indices = self.index.search(vector, k)
        
        # Résultats
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            row = self.df.iloc[idx]
            results.append({
                "medicament": row["medicament"],
                "avis": row["avis_for_embedding"],
                "score": float(dist)
            })
        
        return results

if __name__ == "__main__":
    # Chemins
    mapping_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
    index_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"
    
    # Initialisation et recherche
    search = HealthcareSearch(mapping_path, index_path)
    results = search.search(" Résume moi les avis sur Abasaglar", k=4)
    
    # Affichage
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['medicament']}] (score: {r['score']:.3f})")
        print(f"   {r['avis']}")
        print()