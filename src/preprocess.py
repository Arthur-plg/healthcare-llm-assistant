import pandas as pd
import re
import unidecode

df = pd.read_csv("/Users/arthurpelong/healthcare-llm-assistant/data/raw/comments_final.csv")

def get_med(x):
    """
    Extracts the medication name from a URL string.
    """
    y = x.split('/')

    if y[-1].lower().startswith("a-313"):
        return "a-313"
    
    else: 
        z = y[-1].split('-')
        return z[0]
    

def clean_text(text):
    """
    Cleans text by lowering, removing accents, and stripping special characters.
    """
    # Lower case
    text = text.lower()

    # Remove accents
    text = unidecode.unidecode(text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text) 

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean(df): 
    """
    Performs data cleaning pipeline and exports clean patient comments CSV.
    """
    df['medicament'] = df['url'].apply(get_med)
    df['avis_clean'] = df['comment'].apply(clean_text)

    df = df.dropna(subset=['avis_clean'])
    df = df[df['avis_clean'] != ""]

    df['avis_for_embedding'] = df.apply(lambda row: f"{row['medicament']}: {row['avis_clean']}", axis=1)

    output_path = "/Users/arthurpelong/healthcare-llm-assistant/data/processed/avis_clean.csv"
    df.to_csv(output_path, index=False)
    
    return df 
    
if __name__ == "__main__":
    df = pd.read_csv("/Users/arthurpelong/healthcare-llm-assistant/data/raw/comments_final.csv")
    clean(df)
