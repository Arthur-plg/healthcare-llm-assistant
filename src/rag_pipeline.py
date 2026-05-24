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
        Initializes the local search engine and the Groq API client.
        """
        self.search_engine = HealthcareSearch(mapping_path, index_path)
        self.client = Groq(api_key=api_key)

    def build_prompt(self, query, retrieved_docs):
        """
        Constructs a structured prompt containing the retrieved local patient reviews.
        Note: The source comments are in French, but the LLM is instructed to output in English.
        """
        context_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            clean_avis = doc['avis'].split(':', 1)[-1].strip()
            context_text += f"Review {i}: {clean_avis}\n"

        prompt = f"""You are a medical assistant expert in analyzing patient testimonials.
Summarize the following patient reviews concerning the question in a few smooth sentences.

CRITICAL NOTE:
The patient reviews provided below are written in French. You MUST read and analyze them in French, but write your final summary response entirely in English.

STRICT GUIDELINES:
1. Clearly remind the user that you are only summarizing patient reviews and that your response is by no means professional medical advice.

PATIENT REVIEWS (in French):
{context_text}

QUESTION: {query}

RESPONSE (in English):"""
        return prompt

    def generate_answer(self, prompt):
        """
        Calls the Groq API to generate the final response using Llama-3.3-70b.
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile", 
                messages=[
                    {"role": "system", "content": "You are a factual medical assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  
                max_tokens=300,
                top_p=1,
                stream=False
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error during generation with Groq: {str(e)}"

    def run(self, query):
        """
        Main runner: retrieves documents and generates a response.
        """
        retrieved = self.search_engine.search(query, k=10)

        if not retrieved:
            return "Sorry, I could not find any patient reviews matching your request.", []

        prompt = self.build_prompt(query, retrieved)
        answer = self.generate_answer(prompt)

        return answer, retrieved


if __name__ == "__main__":
    assistant = GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)
    
    user_query = "Traitement Fivasa ?"
    
    response, sources = assistant.run(user_query)

    print("\n" + "="*30)
    print("MEDICAL SUMMARY")
    print("="*30)
    print(response)
    print("\n" + "="*30)
    print("SOURCES USED")
    for s in sources:
        print(f"- [Score: {s['score']:.3f}] [{s['medicament'].upper()}] {s['avis']}")