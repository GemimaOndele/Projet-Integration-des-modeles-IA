import os
import time
import feedparser
import pandas as pd
import joblib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
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

# Setup Selenium en headless
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)

def get_rss_articles(feed_url, max_articles=5):
    """
    Utilise feedparser pour récupérer les URLs des articles à partir d’un flux RSS.
    """
    feed = feedparser.parse(feed_url)
    entries = feed.entries[:max_articles]
    return [entry.link for entry in entries]

def fetch_article_with_selenium(url):
    """
    Charge une page web avec Selenium et passe le HTML à newspaper3k.
    """
    try:
        driver.get(url)
        time.sleep(2)  # attendre le chargement du contenu
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

# URLs RSS
rss_feeds = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.reutersagency.com/feed/?best-topics=politics"  # Exemple alternatif
]

all_articles = []
for feed_url in rss_feeds:
    urls = get_rss_articles(feed_url)
    print(f"🔗 {len(urls)} liens extraits du flux : {feed_url}")
    for link in urls:
        article_data = fetch_article_with_selenium(link)
        if article_data:
            print(f"✅ Article extrait : {article_data['title']}")
            all_articles.append(article_data)
        time.sleep(1)

driver.quit()

# Analyse avec ton modèle IA
if not all_articles:
    print("Aucun article valide trouvé.")
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
df.to_csv("results/rss_selenium_articles.csv", index=False)
print("📁 Résultats enregistrés dans 'results/rss_selenium_articles.csv'")
print("✅ Scrapping terminée !")