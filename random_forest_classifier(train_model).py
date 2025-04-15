import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Fonction de nettoyage
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Chargement du dataset
df = pd.read_csv("data/all_news_cleaned.csv")
df.columns = df.columns.str.strip()

print("Colonnes :", df.columns)
print("Répartition des classes :", df['label'].value_counts())

# Nettoyage du texte
df['text'] = df['text'].apply(clean_text)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Entraînement Random Forest
print("Entraînement du modèle Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_tfidf, y_train)
print("Entraînement terminé.")

# Évaluation
rf_pred = rf_clf.predict(X_test_tfidf)
print("Accuracy (Random Forest):", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
