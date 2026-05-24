import os
import json
import pandas as pd
from groq import Groq
from rag_pipeline import GroqRAGAssistant
from dotenv import load_dotenv

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAPPING_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
INDEX_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

class RAGJudge:
    def __init__(self, api_key):
        """
        Initializes the Groq client and sets the evaluation model.
        """
        self.client = Groq(api_key=api_key)
        # Use Groq's most capable model to judge
        self.model = "llama-3.3-70b-versatile"

    def judge_faithfulness(self, context, answer):
        """Evaluates whether the response contains hallucinations or invented information."""
        prompt = f"""
        ROLE: You are an expert medical auditor.
        TASK: Verify if the GENERATED RESPONSE is strictly based on the provided CONTEXT.
        
        CONTEXT (Patient reviews):
        {context}
        
        GENERATED RESPONSE:
        {answer}
        
        CRITERION: If the response mentions a side effect, a benefit, or a figure that is NOT in the context, the score is 0. Otherwise, it is 1.
        
        RESPOND ONLY IN THE FOLLOWING JSON FORMAT:
        {{"score": 0 or 1, "reason": "short explanation"}}
        """
        return self._get_json_response(prompt)

    def judge_relevance(self, query, answer):
        """Evaluates whether the response directly and usefully answers the user's question."""
        prompt = f"""
        TASK: Evaluate if the RESPONSE directly answers the QUESTION.
        QUESTION: {query}
        RESPONSE: {answer}
        
        CRITERION: Is the response useful and does it address the topic? (Score from 0 to 5)
        
        RESPOND ONLY IN THE FOLLOWING JSON FORMAT:
        {{"score": int, "reason": str}}
        """
        return self._get_json_response(prompt)

    def _get_json_response(self, prompt):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            return {"score": 0, "reason": f"Judge error: {e}"}


def run_evaluation():
    # 1. Initialization
    assistant = GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)
    judge = RAGJudge(GROQ_API_KEY)
    
    # 2. Test dataset (drug targets remain in French to match index terms)
    test_queries = [
        "Quels sont les effets secondaires du Fivasa ?",
        "Le médicament A-313 est-il efficace pour la peau ?",
        "Avis sur l'efficacité du Doliprane ?",
        "Quels sont les effets secondaires sur le médicament abilify ?",
        "Quels sont les avis des patients sur le médicament serc ?",
        "Est ce que le médicament actiskenan soulage les douleurs sciatiques/lombaires ?",
        "Est ce que les rhumatologues conseillent l’actonel à leur patient? ",
        "Avis négatif sur alprazolam ? ",
        "Avis amlodipine ? ",
        "Possible de prendre arava en complément ou association avec un autre médicament ? ",
        "Médicaments pour BPCO ? "
    ]
    
    results = []

    print(f"Launching evaluation on {len(test_queries)} queries...\n")

    for query in test_queries:
        print(f"Test query: {query}")
        
        # Execute RAG pipeline
        answer, sources = assistant.run(query)
        
        # Prepare context for the judge (merging retrieved patient comments)
        full_context = "\n".join([s['avis'] for s in sources])
        
        # Judge scores
        faith_res = judge.judge_faithfulness(full_context, answer)
        rel_res = judge.judge_relevance(query, answer)
        
        results.append({
            "query": query,
            "answer": answer,
            "faithfulness_score": faith_res['score'],
            "faithfulness_reason": faith_res['reason'],
            "relevance_score": rel_res['score'],
            "relevance_reason": rel_res['reason']
        })

    # 3. Export and summary report
    df_eval = pd.DataFrame(results)
    df_eval.to_csv("evaluation_results.csv", index=False)
    
    print("\n--- SCORE SUMMARY ---")
    print(f"Average Faithfulness: {df_eval['faithfulness_score'].mean():.2f}/1")
    print(f"Average Relevance: {df_eval['relevance_score'].mean():.2f}/5")
    print("\nResults saved in 'evaluation_results.csv'")


if __name__ == "__main__":
    run_evaluation()