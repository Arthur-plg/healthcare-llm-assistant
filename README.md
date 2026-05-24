# Healthcare LLM Assistant

**Domain-specific LLM assistant for patient forums** (Educational project – not medical advice)

> This project aims to build an assistant capable of answering questions about medications based on patient reviews.  
> The data was scraped from Carenity for purely educational, non-profit purposes.

> The user submits a query to the assistant, which generates a response based on the most relevant patient reviews retrieved for that query.

---

## Tech Stack & Pipeline Map

1. **Scraping / Data Collection**  
   - **Tools:** `requests`, `BeautifulSoup`, `selenium`, `pandas`  
   - **Role:** Retrieve medications and patient reviews from Carenity  
   - **Expected Output:** `data/raw/medicaments_carenity.csv`, `data/raw/comments_final.csv`  

2. **Preprocessing (Classic NLP)**  
   - **Tools:** `pandas`, `regex`  
   - **Role:** Clean and standardize patient reviews  
   - **Expected Output:** `data/processed/avis_clean.csv`  

3. **Embeddings (Semantic Vectorization)**  
   - **Tools:** `sentence-transformers` (specifically `all-MiniLM-L6-v2`)  
   - **Role:** Transform each review into a dense vector to capture meaning and context  
   - **Expected Output:** Dense vector matrix + indexable file  

4. **Vector Database / Indexing**  
   - **Tools:** FAISS  
   - **Role:** Store embeddings and quickly retrieve the nearest neighbors for semantic search  
   - **Expected Output:** `indexes/faiss_index`  

5. **RAG (Retrieval-Augmented Generation)**  
   - **Tools:** FAISS, SentenceTransformer, Groq API, Llama-3.3-70b-versatile  
   - **Role:** Combine **semantic search** and **text generation**:  
     - Vectorize the user's query  
     - Retrieve relevant patient reviews via FAISS (filtering specifically for the drug mentioned in the query)  
     - Provide this **context** to the LLM (Llama-3.3-70b via Groq API)  
     - The LLM processes the French context and generates a synthesized response **in English**  

6. **Application (UI & Visualization)**  
   - **Tools:** Streamlit  
   - **Role:** User-friendly interface to interact with the assistant  
   - **Features:** Q&A search bar, instant automatic summaries by medication  

---

## Current Status

- ✅ `scraping1.py` and `scraping2.py`: Medication and review collection is fully functional  
- ✅ `search.py`: FAISS semantic search is functional $\rightarrow$ **retrieval is operational**  
- ✅ `rag_pipeline.py`: **final generation via RAG is fully operational**  
- ✅ `app.py`: Streamlit UI is fully functional  
- ✅ `eval_deepeval.py`: Scientific evaluation with DeepEval is fully operational  

---

## Areas for Improvement

- Semantic embedding quality tuned specifically for the healthcare context.

---

## Installation

1. Clone the repository and navigate into the folder:
   ```bash
   git clone https://github.com/Arthur-plg/healthcare-llm-assistant.git
   cd healthcare-llm-assistant
   ```

2. Activate the Conda environment:
   ```bash
   conda activate rag_env
   ```

3. Install required dependencies (including the evaluation framework):
   ```bash
   pip install -r requirements.txt
   pip install streamlit groq python-dotenv deepeval
   ```

---

## How to Run the Project

### 1. Launch the Web Interface (Streamlit UI) ⚕️
To run and interact with the user interface:
```bash
streamlit run src/app.py
```

### 2. Run RAG Evaluation 📊

We have two ways to evaluate the quality of the generated answers:

#### A. Advanced Evaluation using **DeepEval** (Recommended) 🏆
This script runs the **DeepEval** evaluation framework on top of our Groq model (`llama-3.3-70b-versatile`) to calculate two key RAG metrics:
* **Faithfulness** (Fidelity / No Hallucinations): Checks if the assistant's response is strictly based on the retrieved context from FAISS.
* **Answer Relevancy** (Relevance): Checks if the assistant's response directly and usefully answers the user's query.

To execute this evaluation:
```bash
python src/eval_deepeval.py
```
Detailed scores and explanations will be exported to `deepeval_results.csv`.

#### B. Simple Evaluation using a Custom LLM Judge
A lightweight evaluation script that uses a custom judge prompt to rate faithfulness and relevance:
```bash
python src/eval.py
```
Results will be saved in `evaluation_results.csv`.
