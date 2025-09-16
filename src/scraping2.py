from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd

def scrape_carenity_comments():
    # --- Charger la liste des médicaments ---
    df_med = pd.read_csv('/Users/arthurpelong/healthcare-llm-assistant/data/raw/medicaments_carenity.csv')
    list_url = df_med.url.to_list()

    # --- Lancer le navigateur ---
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Décommente pour du headless
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)

    # --- Aller sur la page de connexion ---
    driver.get("https://www.carenity.com/connexion")
    wait = WebDriverWait(driver, 15)

    # --- Accepter les cookies ---
    try:
        accept_cookies = wait.until(
            EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button"))
        )
        accept_cookies.click()
        print("✅ Cookies acceptés")
    except Exception:
        print("⚠️ Pas de popup cookies trouvée (peut-être déjà accepté).")

    # --- Remplir email et mot de passe ---
    email_input = wait.until(
        EC.presence_of_element_located((By.ID, "username"))
    )
    password_input = wait.until(
        EC.presence_of_element_located((By.ID, "password"))
    )
    email_input.send_keys("...")    # Caché pour github
    password_input.send_keys("...") # Caché pour github
    password_input.send_keys(Keys.RETURN)

    print("✅ Connexion réussie (si identifiants corrects)")

    comments_data = []

    # --- Scraper les commentaires ---
    for url in list_url:
        print("Scraping:", url)
        page = 1

        # Convertir l'URL www → membre pour la pagination des commentaires
        base_comment_url = url.replace("www.carenity.com", "membre.carenity.com")

        while True:
            url_paged = base_comment_url if page == 1 else f"{base_comment_url}?page={page}#comments"
            print("Page URL:", url_paged)
            driver.get(url_paged)
            time.sleep(0.5)  # Laisser le temps à la page de charger

            # Récupérer tous les commentaires
            comments = driver.find_elements(By.CSS_SELECTOR, "div.content-align-left")
            comments_text = [c.text.strip() for c in comments if c.text.strip()]

            # Condition d'arrêt : pas de commentaires
            if not comments_text:
                print("✅ Fin des pages pour ce médicament.")
                break

            for text in comments_text:
                comments_data.append({"url": url, "comment": text})

            page += 1
            time.sleep(0.2)

    # --- Sauvegarde ---
    df_comments = pd.DataFrame(comments_data)
    df_comments.to_csv("/Users/arthurpelong/healthcare-llm-assistant/data/raw/comments_final.csv", index=False)

    driver.quit()
    print("💾 Scraping terminé et CSV sauvegardé.")
    return df_comments

if __name__ == "__main__":
    scrape_carenity_comments()