'''import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


df_mapping = pd.read_csv("/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index")


def embed_query(query):      # embeding de la requete user

    embedding = model.encode([query])
    return np.array(embedding)

def search(query,k):

    query_vector = embed_query(query)

    distance, indices = index.search(query_vector,k)

    # récupérer les résultats depuis le mapping medicaments-avis
    results = []
    for idx, dist in zip(indices[0], distance[0]):
        row = df_mapping.iloc[idx]  # récupérer la ligne correspondante
        results.append({
            "avis": row["avis_clean"],
            "medicament": row["medicament"],
            "score": float(dist)        # score de similarité
        })

    return results


if __name__ == "__main__":
    query = "Quels sont les effets secondaires du Doliprane ?"
    results = search(query, k=3)

    print("\nRésultats de recherche :\n")
    for r in results:
        print(f"- [{r['medicament']}] {r['avis']} (score={r['score']:.4f})")
'''



'''
import pandas as pd
import faiss
import numpy as np
import os
import sys
import gc

# Configuration pour éviter les conflits de threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

class IsolatedHealthcareSearch:
    def __init__(self):
        self.df_mapping = None
        self.index = None
        self.model = None
        
    def load_mapping(self, mapping_path):
        """Charge seulement le mapping"""
        print("Chargement du mapping...")
        self.df_mapping = pd.read_csv(mapping_path)
        print(f"Mapping chargé: {len(self.df_mapping)} entrées")
        return True
        
    def load_index(self, index_path):
        """Charge seulement l'index FAISS"""
        print("Chargement de l'index FAISS...")
        try:
            self.index = faiss.read_index(index_path)
            print(f"Index FAISS chargé: {self.index.ntotal} vecteurs, dimension {self.index.d}")
            return True
        except Exception as e:
            print(f"Erreur chargement index: {e}")
            return False
    
    def load_model_safely(self):
        """Charge le modèle avec précautions"""
        print("Chargement du modèle SentenceTransformer...")
        try:
            # Import retardé pour éviter les conflits
            from sentence_transformers import SentenceTransformer
            
            # Forcer l'utilisation du CPU et désactiver le parallélisme
            import torch
            torch.set_num_threads(1)
            
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
            self.model.eval()  # Mode évaluation
            
            print("Modèle chargé avec succès")
            return True
        except Exception as e:
            print(f"Erreur chargement modèle: {e}")
            return False
    
    def embed_query_safe(self, query):
        """Embedding sécurisé de la requête"""
        try:
            import torch
            with torch.no_grad():  # Désactiver le gradient
                embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=False)
                return embedding.astype(np.float32)
        except Exception as e:
            print(f"Erreur embedding: {e}")
            return None
    
    def search_safe(self, query, k=3):
        """Recherche sécurisée"""
        try:
            print(f"Recherche pour: '{query}'")
            
            # 1. Embedding de la requête
            query_vector = self.embed_query_safe(query)
            if query_vector is None:
                return []
            
            print(f"Vecteur requête généré: {query_vector.shape}")
            
            # 2. Recherche FAISS avec gestion d'erreurs
            try:
                distances, indices = self.index.search(query_vector, k)
                print(f"Recherche FAISS réussie")
            except Exception as e:
                print(f"Erreur recherche FAISS: {e}")
                return []
            
            # 3. Récupération des résultats
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.df_mapping):
                    row = self.df_mapping.iloc[idx]
                    results.append({
                        "avis": row["avis_clean"],
                        "medicament": row["medicament"],
                        "score": float(dist)
                    })
                else:
                    print(f"Index invalide ignoré: {idx}")
            
            return results
            
        except Exception as e:
            print(f"Erreur lors de la recherche: {e}")
            import traceback
            traceback.print_exc()
            return []

def main():
    """Fonction principale avec chargement séquentiel"""
    try:
        # Chemins
        mapping_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
        index_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"
        
        # Initialisation
        search_engine = IsolatedHealthcareSearch()
        
        # Chargement séquentiel avec nettoyage mémoire
        print("=== CHARGEMENT SÉQUENTIEL ===")
        
        # 1. Mapping
        if not search_engine.load_mapping(mapping_path):
            sys.exit(1)
        gc.collect()
        
        # 2. Index FAISS
        if not search_engine.load_index(index_path):
            sys.exit(1)
        gc.collect()
        
        # 3. Modèle (le plus risqué)
        print("\n⚠️  Chargement du modèle (étape critique)...")
        if not search_engine.load_model_safely():
            sys.exit(1)
        gc.collect()
        
        print("\n✅ Tous les composants chargés avec succès!")
        
        # Test de recherche
        print("\n=== TEST DE RECHERCHE ===")
        query = "Quels sont les effets secondaires du Doliprane ?"
        results = search_engine.search_safe(query, k=3)
        
        if results:
            print(f"\n📋 Résultats pour: '{query}'")
            print("-" * 50)
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['medicament']}] (score: {result['score']:.4f})")
                print(f"   {result['avis'][:150]}...")
                print()
        else:
            print("❌ Aucun résultat trouvé")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interruption utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

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
        
    def search(self, query, k=3):
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
                "avis": row["avis_clean"],
                "score": float(dist)
            })
        
        return results

if __name__ == "__main__":
    # Chemins
    mapping_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
    index_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"
    
    # Initialisation et recherche
    search = HealthcareSearch(mapping_path, index_path)
    results = search.search("Quels sont les effets secondaires du Doliprane ?", k=3)
    
    # Affichage
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['medicament']}] (score: {r['score']:.3f})")
        print(f"   {r['avis'][:100]}...")
        print()