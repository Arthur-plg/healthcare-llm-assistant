import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import string

BASE_URL = "https://www.carenity.com"
OUT_PATH = "/Users/arthurpelong/healthcare-llm-assistant/data/raw/medicaments_carenity.csv"

def scrape_carenity_medicaments():
    """Scrapes the list of Carenity medications (A-Z) and saves it to a CSV."""
    medicaments_dict = {}

    for LETTER in string.ascii_uppercase:
        print(f"=== Letter {LETTER} ===")
        page = 1
        while True:
            url = f"{BASE_URL}/donner-mon-avis/index-medicaments/{LETTER}"
            if page > 1:
                url += f"?page={page}"

            response = requests.get(url)
            if response.status_code != 200:
                print(f"⚠️ Error {response.status_code} on {url}")
                break

            soup = BeautifulSoup(response.text, "html.parser")
            page_meds = 0
            for a in soup.find_all("a", href=True):
                href = a["href"]
                name = a.get_text(strip=True)
                # Keep "en savoir plus" since it's a French HTML text match on the scraped site
                if href.startswith("/donner-mon-avis/medicaments/") and name.lower() != "en savoir plus":
                    full_url = BASE_URL + href
                    if full_url not in medicaments_dict:
                        medicaments_dict[full_url] = name
                        page_meds += 1

            if page_meds == 0:
                break
            page += 1
            time.sleep(0.5)

        print(f"Total medications collected so far: {len(medicaments_dict)}")

    # Convert to DataFrame
    medicaments = pd.DataFrame([{"name": name, "url": url} for url, name in medicaments_dict.items()])
    print(f"\n✅ Total number of medications collected: {len(medicaments)}")

    # CSV Save
    medicaments.to_csv(OUT_PATH, index=False)
    print(f"💾 Medication list saved to {OUT_PATH}")

    return medicaments


if __name__ == "__main__":
    scrape_carenity_medicaments()
