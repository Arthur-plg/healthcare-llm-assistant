from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd

def scrape_carenity_comments():
    # --- Load the list of medications ---
    df_med = pd.read_csv('/Users/arthurpelong/healthcare-llm-assistant/data/raw/medicaments_carenity.csv')
    list_url = df_med.url.to_list()

    # --- Launch the browser ---
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Uncomment for headless mode
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)

    # --- Navigate to the login page ---
    driver.get("https://www.carenity.com/connexion")
    wait = WebDriverWait(driver, 15)

    # --- Accept cookies ---
    try:
        accept_cookies = wait.until(
            EC.element_to_be_clickable((By.ID, "didomi-notice-agree-button"))
        )
        accept_cookies.click()
        print("✅ Cookies accepted")
    except Exception:
        print("⚠️ No cookie popup found (might have already been accepted).")

    # --- Fill email and password ---
    email_input = wait.until(
        EC.presence_of_element_located((By.ID, "username"))
    )
    password_input = wait.until(
        EC.presence_of_element_located((By.ID, "password"))
    )
    email_input.send_keys("...")    # Hidden for GitHub
    password_input.send_keys("...") # Hidden for GitHub
    password_input.send_keys(Keys.RETURN)

    print("✅ Login successful (if credentials are correct)")

    comments_data = []

    # --- Scrape patient reviews ---
    for url in list_url:
        print("Scraping:", url)
        page = 1

        # Convert URL www -> membre for comments pagination
        base_comment_url = url.replace("www.carenity.com", "membre.carenity.com")

        while True:
            url_paged = base_comment_url if page == 1 else f"{base_comment_url}?page={page}#comments"
            print("Page URL:", url_paged)
            driver.get(url_paged)
            time.sleep(0.5)  # Allow time for the page to load

            # Retrieve all comments
            comments = driver.find_elements(By.CSS_SELECTOR, "div.content-align-left")
            comments_text = [c.text.strip() for c in comments if c.text.strip()]

            # Termination condition: no comments
            if not comments_text:
                print("✅ End of pages for this medication.")
                break

            for text in comments_text:
                comments_data.append({"url": url, "comment": text})

            page += 1
            time.sleep(0.2)

    # --- Saving data ---
    df_comments = pd.DataFrame(comments_data)
    df_comments.to_csv("/Users/arthurpelong/healthcare-llm-assistant/data/raw/comments_final.csv", index=False)

    driver.quit()
    print("💾 Scraping complete and CSV saved.")
    return df_comments

if __name__ == "__main__":
    scrape_carenity_comments()