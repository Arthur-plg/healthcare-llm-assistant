import os
from search import HealthcareSearch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

def init_causal_llm_bloom(model_name="bigscience/bloomz-1b7", device="cpu"):
    """
    Initialisation causal LM léger (Bloom 1B7)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    return generator

def build_structured_context(query, retrieved_docs):
    """
    Construit un prompt structuré pour guider la génération
    """
    medicaments = {}
    for doc in retrieved_docs:
        med = doc['medicament']
        if med not in medicaments:
            medicaments[med] = []
        medicaments[med].append(doc['avis'])

    context_parts = []
    for med, avis_list in medicaments.items():
        avis_text = "\n- ".join([avis.replace(med, '').strip() for avis in avis_list])
        context_parts.append(f"{med}:\n- {avis_text}")

    context = "\n\n".join(context_parts)

    prompt = f"""
Voici les avis patients concernant différents médicaments:
{context}

Question: {query}

Réponds en 3 phrases complètes en français, en synthétisant et reformulant les avis de manière naturelle. Ne répète pas les avis tels quels.
"""
    return prompt

def rag_pipeline_bloom(query, mapping_path, index_path, k=3, device="cpu"):
    """
    Pipeline RAG 
    """
    search_engine = HealthcareSearch(mapping_path, index_path)
    retrieved = search_engine.search(query, k=k)
    if not retrieved:
        return "Aucun avis patient trouvé.", []

    context = build_structured_context(query, retrieved)
    generator = init_causal_llm_bloom(device=device)

    outputs = generator(
        context,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        return_full_text=False
    )

    answer = outputs[0].get("generated_text") or outputs[0].get("text") or "Aucune réponse générée"
    return answer, retrieved

if __name__ == "__main__":
    mapping_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
    index_path = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

    query = "Donne les effets indésirables liés au Fivasa (s'il y en a)"

    # Exécution du pipeline
    answer, retrieved = rag_pipeline_bloom(query, mapping_path, index_path, k=3, device="cpu")

    print("==== Réponse générée  ====")
    print(answer)

    print("\n==== Avis récupérés ====")
    for i, r in enumerate(retrieved, 1):
        print(f"{i}. [{r['medicament']}] (score: {r['score']:.3f})")
        print(f"   {r['avis']}\n")
