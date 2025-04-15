import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def clean_text(text):
    """Fonction de prétraitement basique : 
       conversion en minuscules, suppression des URLs et caractères spéciaux."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Supprime les URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Ne garde que les lettres et espaces
    return text

# Chargez votre dataset nettoyé (all_news_cleaned.csv)
df = pd.read_csv("data/all_news_cleaned.csv")

# Appliquer le prétraitement sur la colonne des textes
df['text'] = df['text'].apply(clean_text)

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Initialiser le TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)

# Transformation des textes en vecteurs TF-IDF
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraînement du modèle Gradient Boosting
print("Entraînement du modèle Gradient Boosting...")
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train_tfidf, y_train)
print("Entraînement terminé (Gradient Boosting).")

# Prédictions
gb_pred = gb_clf.predict(X_test_tfidf)
print("Accuracy (Gradient Boosting):", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred))

# Sauvegarde du modèle et du vectorizer
joblib.dump(gb_clf, "model/gradient_boosting_fake_news.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
