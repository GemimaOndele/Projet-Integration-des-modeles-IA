import os
import time
import feedparser
import pandas as pd
import joblib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from newspaper import Article, Config
from preprocess import clean_text

# Chargement du modèle et du vectoriseur
model = joblib.load("model/fake_news_random_forest_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Configuration du user-agent pour newspaper3k
config = Config()
config.browser_user_agent = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/91.0.4472.124 Safari/537.36'
)

# === SELENIUM ===

def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# 📰 Extraction d’articles à partir des flux RSS
def get_rss_articles(feed_url, max_articles=5):
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_articles]
    return [entry.link for entry in entries]

# 🧠 Chargement d’un article avec le HTML depuis Selenium (pour RSS)
def fetch_article_with_selenium(driver, url):
    try:
        driver.get(url)
        time.sleep(2)
        html = driver.page_source
        article = Article(url, config=config)
        article.set_html(html)
        article.parse()
        if len(article.text) > 100:
            return {
                "title": article.title,
                "text": article.text,
                "url": url
            }
    except Exception as e:
        print(f"Erreur avec Selenium : {e}")
    return None

# 🌐 Extraction de liens "article" depuis une homepage
def get_article_urls_with_selenium(driver, url, max_links=5):
    driver.get(url)
    time.sleep(3)
    links = []
    elems = driver.find_elements(By.TAG_NAME, "a")
    for elem in elems:
        href = elem.get_attribute("href")
        if href and "article" in href and href not in links:
            links.append(href)
        if len(links) >= max_links:
            break
    return links

# 📄 Scraper les textes des liens détectés manuellement (pas via RSS)
def scrape_articles_from_urls(article_urls):
    articles_list = []
    for url in article_urls:
        try:
            article = Article(url, config=config)
            article.download()
            article.parse()
            if len(article.text) > 100:
                articles_list.append({
                    "title": article.title,
                    "text": article.text,
                    "url": article.url
                })
        except Exception as e:
            print(f"Erreur sur {url} : {e}")
        time.sleep(1)
    return articles_list

# === MAIN SCRIPT ===

driver = init_driver()
all_articles = []

# 🎯 Étape 1 : Articles via RSS
rss_feeds = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=politics"
]

for feed_url in rss_feeds:
    urls = get_rss_articles(feed_url)
    print(f"🔗 {len(urls)} liens extraits du flux : {feed_url}")
    for link in urls:
        article_data = fetch_article_with_selenium(driver, link)
        if article_data:
            print(f"✅ Article RSS extrait : {article_data['title']}")
            all_articles.append(article_data)
        time.sleep(1)

# 🎯 Étape 2 : Articles via homepage directe (optionnel, bonus)
homepage_sources = [
    "https://www.france24.com/en/",  # Tu peux en ajouter d'autres
]

for homepage_url in homepage_sources:
    try:
        print(f"🌍 Scraping de la page : {homepage_url}")
        article_links = get_article_urls_with_selenium(driver, homepage_url, max_links=5)
        homepage_articles = scrape_articles_from_urls(article_links)
        print(f"✅ {len(homepage_articles)} articles extraits depuis la homepage.")
        all_articles.extend(homepage_articles)
    except Exception as e:
        print(f"Erreur lors du scraping homepage : {e}")

driver.quit()

# === ANALYSE IA ===
if not all_articles:
    print("❌ Aucun article valide trouvé.")
    exit()

df = pd.DataFrame(all_articles)
df["cleaned_text"] = df["text"].apply(clean_text)
X = vectorizer.transform(df["cleaned_text"])
predictions = model.predict(X)
probas = model.predict_proba(X)

df["prediction"] = ["Fake" if pred == 1 else "Real" for pred in predictions]
df["proba_fake"] = [round(prob[1] * 100, 2) for prob in probas]
df["proba_real"] = [round(prob[0] * 100, 2) for prob in probas]

os.makedirs("results", exist_ok=True)
df.to_csv("results/combined_scraping_results.csv", index=False)
print("📁 Résultats enregistrés dans 'results/combined_scraping_results.csv'")
print("✅ Scraping & prédiction terminés avec succès !")
