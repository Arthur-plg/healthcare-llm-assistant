import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import string

BASE_URL = "https://www.carenity.com"
OUT_PATH = "/Users/arthurpelong/healthcare-llm-assistant/data/raw/medicaments_carenity.csv"

def scrape_carenity_medicaments():
    """Scrape la liste des médicaments Carenity (A-Z) et sauvegarde CSV."""
    medicaments_dict = {}

    for LETTER in string.ascii_uppercase:
        print(f"=== Lettre {LETTER} ===")
        page = 1
        while True:
            url = f"{BASE_URL}/donner-mon-avis/index-medicaments/{LETTER}"
            if page > 1:
                url += f"?page={page}"

            response = requests.get(url)
            if response.status_code != 200:
                print(f"⚠️ Erreur {response.status_code} sur {url}")
                break

            soup = BeautifulSoup(response.text, "html.parser")
            page_meds = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                name = a.get_text(strip=True)
                if href.startswith("/donner-mon-avis/medicaments/") and name.lower() != "en savoir plus":
                    full_url = BASE_URL + href
                    if full_url not in medicaments_dict:
                        medicaments_dict[full_url] = name
                        page_meds += 1

            if page_meds == 0:
                break
            page += 1
            time.sleep(0.5)

        print(f"Total médicaments collectés jusque-là : {len(medicaments_dict)}")

    # Transformer en DataFrame
    medicaments = pd.DataFrame([{"name": name, "url": url} for url, name in medicaments_dict.items()])
    print(f"\n✅ Nombre total de médicaments collectés : {len(medicaments)}")

    # Sauvegarde CSV
    medicaments.to_csv(OUT_PATH, index=False)
    print(f"💾 Liste des médicaments sauvegardée dans {OUT_PATH}")

    return medicaments


if __name__ == "__main__":
    scrape_carenity_medicaments()
