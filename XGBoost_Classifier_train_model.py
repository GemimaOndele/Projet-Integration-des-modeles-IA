import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

# ----------- Prétraitement basique -----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ----------- Chargement du dataset -----------
df = pd.read_csv("data/all_news_cleaned.csv")
df.columns = df.columns.str.strip()
print("Colonnes du dataset : ", df.columns)

# Nettoyage du texte
df['text'] = df['text'].apply(clean_text)

# Séparation features / target
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# ----------- TF-IDF Vectorisation -----------
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------- Modèle XGBoost -----------
print("Entraînement du modèle XGBoost...")
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(X_train_tfidf, y_train)
print("Entraînement terminé (XGBoost).")

# ----------- Évaluation -----------
xgb_pred = xgb_clf.predict(X_test_tfidf)
print("Accuracy (XGBoost):", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# ----------- Sauvegarde du modèle et du vectorizer -----------
joblib.dump(xgb_clf, "model/xgboost_fake_news.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("Modèle et vectorizer sauvegardés avec succès dans le dossier 'model'.")
