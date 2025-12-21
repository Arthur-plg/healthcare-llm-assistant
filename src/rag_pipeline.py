import os
import torch
from groq import Groq
from search import HealthcareSearch
from dotenv import load_dotenv

load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAPPING_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
INDEX_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

class GroqRAGAssistant:
    def __init__(self, mapping_path, index_path, api_key):
        """
        Initialise le moteur de recherche local et le client API Groq.
        """
        self.search_engine = HealthcareSearch(mapping_path, index_path)
        self.client = Groq(api_key=api_key)

    def build_prompt(self, query, retrieved_docs):
        """
        Construit un prompt structuré avec les avis trouvés localement.
        """
        context_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            clean_avis = doc['avis'].split(':', 1)[-1].strip()
            context_text += f"Avis {i}: {clean_avis}\n"

        prompt = f"""Tu es un assistant médical expert en analyse de témoignages de patients.
Résume les avis suivants concernant la question posée en quelques phrases de manière fluide.

CONSIGNES STRICTES :
1. Rappelle que tu ne fais que résumé des avis patients et que ta réponse n'est en rien celle d'un professionnel de santé.

AVIS PATIENTS :
{context_text}

QUESTION : {query}

RÉPONSE (en français) :"""
        return prompt

    def generate_answer(self, prompt):

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {"role": "system", "content": "Tu es un assistant médical factuel."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  
                max_tokens=300,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Erreur lors de la génération avec Groq : {str(e)}"

    def run(self, query):

        retrieved = self.search_engine.search(query, k=5)

        if not retrieved:
            return "Désolé, je n'ai trouvé aucun avis patient correspondant à votre demande.", []

        prompt = self.build_prompt(query, retrieved)
        answer = self.generate_answer(prompt)

        return answer, retrieved

if __name__ == "__main__":
   
    assistant = GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)
    
    user_query = "Traitement Fivasa ?"
    
    reponse, sources = assistant.run(user_query)

    print("\n" + "="*30)
    print("SYNTHÈSE MÉDICALE")
    print("="*30)
    print(reponse)
    print("\n" + "="*30)
    print("SOURCES UTILISÉES")
    for s in sources:
        print(f"- [Score: {s['score']:.3f}] [{s['medicament'].upper()}] {s['avis']}")