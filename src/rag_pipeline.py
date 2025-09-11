# rag_pipeline.py

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Configuration anti-conflit MacOS / threads
os.environ["OMP_NUM_THREADS"] = "1"

# --- Classe RAG
class HealthcareRAG:
    def __init__(self, mapping_path, index_path, top_k=5):
        # Charger mapping et index FAISS
        self.df = pd.read_csv(mapping_path)
        self.index = faiss.read_index(index_path)
        self.top_k = top_k
        
        # Charger modèle embeddings sur CPU
        import torch
        torch.set_num_threads(1)
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        
        # Préparer LLM LangChain (OpenAI GPT)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        # Prompt template pour contextualiser la réponse
        self.prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Tu es un assistant santé. Voici les avis patients pertinents :\n"
                "{context}\n\n"
                "Question utilisateur : {query}\n"
                "Réponds de manière concise et claire, en te basant sur ces avis."
            )
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    # --- Encodage de la requête
    def embed_query(self, query):
        import torch
        with torch.no_grad():
            vector = self.model.encode([query]).astype(np.float32)
        return vector

    # --- Recherche FAISS
    def search(self, query):
        vector = self.embed_query(query)
        distances, indices = self.index.search(vector, self.top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            row = self.df.iloc[idx]
            results.append({
                "medicament": row["medicament"],
                "avis": row["avis_clean"],
                "score": float(dist)
            })
        return results

    # --- Générer réponse LLM avec top-k avis
    def generate_answer(self, query):
        search_results = self.search(query)
        context_text = "\n".join([f"- [{r['medicament']}] {r['avis']}" for r in search_results])
        response = self.chain.run({"query": query, "context": context_text})
        return response

# --- Exemple d'utilisation
if __name__ == "__main__":
    mapping_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
    index_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"
    
    rag = HealthcareRAG(mapping_path, index_path)
    query = "Quels sont les effets secondaires du Doliprane ?"
    
    answer = rag.generate_answer(query)
    print("\nRéponse générée :\n")
    print(answer)
