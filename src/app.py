import streamlit as st
from rag_pipeline import GroqRAGAssistant
import os
from dotenv import load_dotenv

# Page Configuration
st.set_page_config(
    page_title="H-LLM Assistant",
    page_icon="medical_symbol",
    layout="centered"
)

# Load Environment Variables and Constants
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAPPING_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/mapping_index.csv"
INDEX_PATH = "/Users/arthurpelong/healthcare-llm-assistant/indexes/faiss_index"

# Initialize the assistant (cached to avoid reloading the model on every click)
@st.cache_resource
def load_assistant():
    return GroqRAGAssistant(MAPPING_PATH, INDEX_PATH, GROQ_API_KEY)

assistant = load_assistant()

# --- User Interface ---
st.title("⚕️ Medical Assistant (Patient Reviews)")
st.markdown("""
This application summarizes patient reviews about specific medications. 
*Note: This is a summary of patient feedback and does not replace professional medical advice.*
""")

# Search Bar
query = st.text_input("Ask a question about a treatment (e.g., 'Side effects of Fivasa?')", placeholder="Search...")

if query:
    with st.spinner("Analyzing patient reviews..."):
        # Call the existing RAG pipeline
        answer, sources = assistant.run(query)
        
        # Display the LLM summary response
        st.subheader("🤖 Assistant Summary")
        st.info(answer)
        
        # Display the source reviews (in an expander to keep the UI clean)
        with st.expander("🔍 View patient review sources used"):
            for i, s in enumerate(sources):
                st.markdown(f"**Review {i+1}** - *Medication: {s['medicament'].upper()}* (Score: {s['score']:.3f})")
                st.write(s['avis'])
                st.divider()

# Footer
st.caption("Developed with Streamlit & Groq Llama-3.3")