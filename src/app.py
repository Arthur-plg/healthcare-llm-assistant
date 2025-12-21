import streamlit as st
from rag_pipeline import GroqRAGAssistant
import os
from dotenv import load_dotenv

# Configuration de la page
st.set_page_config(
    page_title="H-LLM Assistant",
    page_icon="medical_symbol",
    layout="centered"
)

# Chargement des variables d'environnement et constantes
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAPPING_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
INDEX_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

# Initialisation de l'assistant (mise en cache pour éviter de recharger le modèle à chaque clic)
@st.cache_resource
def load_assistant():
    return GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)

assistant = load_assistant()

# --- Interface Utilisateur ---
st.title("⚕️ Assistant Médical (Avis Patients)")
st.markdown("""
Cette application résume les témoignages de patients sur des médicaments spécifiques. 
*Note : Ceci est une synthèse d'avis et ne remplace pas un avis médical.*
""")

# Barre de recherche
query = st.text_input("Posez votre question sur un traitement (ex: 'Effets secondaires du Fivasa ?')", placeholder="Rechercher...")

if query:
    with st.spinner("Analyse des avis en cours..."):
        # Appel de votre pipeline existant
        reponse, sources = assistant.run(query)
        
        # Affichage de la réponse du LLM
        st.subheader("🤖 Synthèse de l'Assistant")
        st.info(reponse)
        
        # Affichage des sources (expander pour ne pas encombrer l'écran)
        with st.expander("🔍 Voir les avis sources utilisés"):
            for i, s in enumerate(sources):
                st.markdown(f"**Avis {i+1}** - *Médicament : {s['medicament'].upper()}* (Score: {s['score']:.3f})")
                st.write(s['avis'])
                st.divider()

# Pied de page
st.caption("Développé avec Streamlit & Groq Llama-3.3")