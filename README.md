# Healthcare LLM Assistant

**Domain-specific LLM assistant for patient forums** (Educational project – not medical advice)

> Ce projet vise à créer un assistant capable de répondre à des questions sur des médicaments à partir d'avis patients.  
> Les données ont été scrappées sur Carenity à but purement éducatif et non lucratif.

> On passe une requête à l'assistant qui génère une réponse basée sur les avis les plus pertinents liés à cette requête.

---

## Pipeline / Cartographie des technos

1. **Scraping / Collecte des données**  
   - **Outils :** `requests`, `BeautifulSoup`, `selenium`, `pandas`  
   - **Rôle :** récupérer les médicaments et les avis patients depuis Carenity  
   - **Output attendu :** `data/raw/medicaments_carenity.csv`, `data/raw/comments_final.csv`  

2. **Prétraitement (NLP classique)**  
   - **Outils :** `pandas`, `regex`  
   - **Rôle :** nettoyer et uniformiser les textes patients  
   - **Output attendu :** `data/processed/avis_clean.csv`  

3. **Embeddings (Vectorisation sémantique)**  
   - **Outils :** `sentence-transformers` (ex. `all-MiniLM-L6-v2`)  
   - **Rôle :** transformer chaque avis en vecteur dense pour capturer le sens et le contexte  
   - **Output attendu :** matrice de vecteurs + sauvegarde indexable  

4. **Base vectorielle / Indexation**  
   - **Outils :** FAISS  
   - **Rôle :** stocker les embeddings et retrouver rapidement les plus proches pour la recherche sémantique  
   - **Output attendu :** `indexes/faiss_index`  

5. **RAG (Retrieval-Augmented Generation)** 
   - **Outils :** FAISS, SentenceTransformer, Groq API, Llama 3.3 70b
   - **Rôle :** combiner **recherche sémantique** et **génération de texte** :  
     - Transformer la requête utilisateur en vecteur  
     - Retrouver les avis patients pertinents via FAISS  (filtrer la recherche sur seulement les avis sur le médicament étant présent dans la user query)
     - Fournir ce **contexte** au LLM choisi : ici utilisation de Llama 3.3 70b à travers l'API Groq
     - LLM choisi : Llama 3.3 70b
     - Générer une réponse finale (synthèse des avis)  

6. **Application (UI + Visualisation)**  
   - **Outils :** Streamlit 
   - **Rôle :** interface pour interagir avec l’assistant  
   - **Fonctions prévues :** zone de recherche / champ Q&A, résumés automatiques par médicament,


---

## État actuel

- ✅ `scraping1.py` et `scraping2.py` : récupération des médicaments et commentaires fonctionnelle  
- ✅ `search.py` : recherche vectorielle FAISS fonctionnelle → **retrieval opérationnel**  
- ✅ `rag_pipeline.py` **génération finale via RAG** 
- ✅ `app.py`: UI fonctionnelle
---

## A améliorer

- Qualité des embeddings dans un contexte de santé 

---

## Installation

```bash
git clone https://github.com/Arthur-plg/healthcare-llm-assistant.git
cd healthcare-llm-assistant
pip install -r requirements.txt




