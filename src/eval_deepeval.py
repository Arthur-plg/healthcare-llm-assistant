import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from deepeval import evaluate
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from rag_pipeline import GroqRAGAssistant

# Configuration
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAPPING_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
INDEX_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not defined in the .env file")

class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """
    Custom wrapper to integrate Groq's Llama-3.3 model into DeepEval.
    """
    def __init__(self, api_key, model_name="llama-3.3-70b-versatile"):
        self.model_name = model_name
        self.client = Groq(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = completion.choices[0].message.content.strip()
            
            # Extract JSON block using regex if wrapped in code blocks or conversational prefixes
            import re
            
            # Check for ```json ... ```
            json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
            if json_match:
                return json_match.group(1).strip()
            
            # Check for ``` ... ```
            code_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            # Extract first outer JSON object {...}
            bracket_match = re.search(r"(\{.*\})", content, re.DOTALL)
            if bracket_match:
                return bracket_match.group(1).strip()
                
            return content
        except Exception as e:
            return f"Groq generation error: {e}"

    async def a_generate(self, prompt: str) -> str:
        # DeepEval calls a_generate when possible, we provide the sync fallback
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name


def run_deepeval_evaluation():
    # 1. RAG Assistant Initialization
    print("Initializing the RAG assistant...")
    assistant = GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)
    
    # 2. Custom Evaluation Model Initialization
    print("Initializing the DeepEval judge model (Groq)...")
    eval_model = GroqDeepEvalLLM(api_key=GROQ_API_KEY)

    # 3. Test dataset (drug targets remain in French to match index terms)
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

    test_cases = []
    results_raw = []

    print(f"\nGenerating RAG responses for {len(test_queries)} queries...")
    for query in test_queries:
        print(f"-> Processing query: '{query}'")
        
        # Execute RAG pipeline
        answer, sources = assistant.run(query)
        
        # Extract FAISS search context
        retrieval_context = [s['avis'] for s in sources]
        
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)
        
        # Temporary storage for our final CSV summary
        results_raw.append({
            "query": query,
            "answer": answer,
            "context": "\n".join(retrieval_context)
        })

    # 4. Configure evaluation metrics
    print("\nConfiguring evaluation metrics...")
    
    # Faithfulness: evaluates whether the response is strictly based on context (no hallucinations)
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model=eval_model, async_mode=False)
    
    # Relevancy: evaluates whether the response directly answers the user's question
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model=eval_model, async_mode=False)

    # 5. Launch global evaluation
    print("\nLaunching global evaluation with DeepEval...")
    from deepeval.evaluate.configs import ErrorConfig
    
    metrics_results = evaluate(
        test_cases=test_cases,
        metrics=[faithfulness_metric, relevancy_metric],
        error_config=ErrorConfig(ignore_errors=True)
    )

    # 6. Structure and export the results to CSV
    print("\nExporting evaluation results...")
    final_results = []
    
    for i, tr in enumerate(metrics_results.test_results):
        # Retrieve scores calculated by DeepEval
        faithfulness_score = None
        faithfulness_reason = None
        relevancy_score = None
        relevancy_reason = None
        
        if tr.metrics_data:
            for md in tr.metrics_data:
                if md.name == "Faithfulness":
                    faithfulness_score = md.score
                    faithfulness_reason = md.reason
                elif md.name == "Answer Relevancy":
                    relevancy_score = md.score
                    relevancy_reason = md.reason
                
        final_results.append({
            "query": results_raw[i]["query"],
            "answer": results_raw[i]["answer"],
            "faithfulness_score": faithfulness_score,
            "faithfulness_reason": faithfulness_reason,
            "relevance_score": relevancy_score,
            "relevance_reason": relevancy_reason
        })

    df = pd.DataFrame(final_results)
    df.to_csv("deepeval_results.csv", index=False)
    
    print("\n" + "="*40)
    print("DeepEval Global Evaluation Summary")
    print("="*40)
    print(f"Average Faithfulness: {df['faithfulness_score'].mean():.2f}/1")
    print(f"Average Relevance: {df['relevance_score'].mean():.2f}/1")
    print("="*40)
    print("Results successfully exported to 'deepeval_results.csv'")


if __name__ == "__main__":
    run_deepeval_evaluation()
