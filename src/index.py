from sentence_transformers import SentenceTransformer 
import pandas as pd
import numpy as np
import faiss
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def build_index():

    df = pd.read_csv("/Users/arthurpelong/healthcare-llm-assistant/data/processed/avis_clean.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = df["avis_for_embedding"].tolist()

    df_index = df[['medicament','avis_for_embedding']].copy()
    df_index.reset_index(inplace=True)  # colonne 'index' correspond aux vecteurs FAISS

    embeddings = model.encode(texts,show_progress_bar=True,convert_to_numpy=True)

    faiss.normalize_L2(embeddings)  # normaliser pour cosine similarity

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index")

    df_index.to_csv("/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv", index=False)

    return index, df_index

if __name__ == "__main__":
    build_index()