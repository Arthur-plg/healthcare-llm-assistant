import pandas as pd
import re
import unidecode

df = pd.read_csv("/Users/arthurpelong/healthcare-llm-assistant/data/raw/comments_final.csv")

def get_med(x):

    y = x.split('/')

    if y[-1].lower().startswith("a-313"):
        return "a-313"
    
    else: 
        z = y[-1].split('-')
        return z[0]
    

def clean_text(text):
    
    # Minuscules
    text = text.lower()

    # Supprimer accents
    text = unidecode.unidecode(text)

    # Supprimer ponctuation et caracteres speciaux 
    text = re.sub(r"[^a-z0-9\s]", " ", text) 

    # Supprimer espaces multiples
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean(df): 
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

