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
   - **Outils :** FAISS, SentenceTransformer, HuggingFace Transformers 
   - **Rôle :** combiner **recherche sémantique** et **génération de texte** :  
     - Transformer la requête utilisateur en vecteur  
     - Retrouver les avis patients pertinents via FAISS  
     - Fournir ce **contexte** au LLM choisi, objectif : choisir un causal LM pas trop gros pouvant être run sur CPU
     - LLM choisi : bigscience/bloomz-1b7
     - Générer une réponse finale (Q&A, résumé, analyse de sentiments)  

6. **Application (UI + Visualisation)**  
   - **Outils :** Streamlit ou Gradio + Plotly, Wordcloud  
   - **Rôle :** interface pour interagir avec l’assistant  
   - **Fonctions prévues :** zone de recherche / champ Q&A, résumés automatiques par médicament, visualisation sentiments & thèmes  

7. **Déploiement**  
   - **Outils :** Streamlit Cloud, Hugging Face Spaces, ou Docker + Cloud provider  
   - **Rôle :** rendre l’assistant accessible publiquement  

---

## État actuel

- ✅ `scraping1.py` et `scraping2.py` : récupération des médicaments et commentaires fonctionnelle  
- ✅ `search.py` : recherche vectorielle FAISS fonctionnelle → **retrieval déjà opérationnel**  
- ⚠️ `rag_pipeline.py`, UI : en cours de développement → incluront la **génération finale via RAG** (Q&A, résumé, tagging)  
- Embeddings et index FAISS préparés, LLM intégré mais pas encore assez bien géré. 

> Le projet est donc **partiellement fonctionnel** : la collecte et la recherche vectorielle sont opérationnelles, la génération de la réponse finale reste à améliorer car les modèles open source pouvant être run sur CPU sont très basique.
> L'application reste à développer.

---

## Installation

```bash
git clone https://github.com/Arthur-plg/healthcare-llm-assistant.git
cd healthcare-llm-assistant
pip install -r requirements.txt



