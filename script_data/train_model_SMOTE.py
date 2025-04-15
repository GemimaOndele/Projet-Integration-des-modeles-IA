import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE

def clean_text(text):
    """Prétraitement : minuscule, suppression URLs, caractères spéciaux."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ========== 1. Chargement du dataset ==========
print("Chargement du dataset all_news_cleaned.csv...")
df = pd.read_csv("data/all_news_cleaned.csv")
df.columns = df.columns.str.strip()
print("Colonnes du dataset : ", df.columns.tolist())

# ========== 2. Optionnel : Sous-échantillonnage pour tests rapides ==========
# Décommente pour activer (ex: pour tests rapides)
df = df.sample(frac=0.2, random_state=42)
print(f"Dataset réduit pour test rapide : {df.shape[0]} lignes")

# ========== 3. Vérification classes ==========
print("Répartition des classes dans 'label' :")
print(df['label'].value_counts())

# ========== 4. Nettoyage texte ==========
print("Nettoyage des textes...")
df['text'] = df['text'].astype(str).apply(clean_text)

# ========== 5. Split des données ==========
print("Séparation en train/test...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print("Répartition des classes dans y_train :")
print(y_train.value_counts())

# ========== 6. Vectorisation TF-IDF ==========
print("Vectorisation TF-IDF en cours...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorization terminée.")

# ========== 7. Application de SMOTE (optionnel) ==========
USE_SMOTE = True  # Passe à False si tu veux désactiver SMOTE

if USE_SMOTE and len(y_train.value_counts()) > 1:
    print("Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
    print("SMOTE terminé.")
else:
    print("SMOTE désactivé ou classes insuffisantes.")
    X_train_smote, y_train_smote = X_train_tfidf, y_train

# ========== 8. Entraînement du modèle ==========
print("Entraînement du modèle LogisticRegression...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_smote, y_train_smote)
print("Entraînement terminé.")

# ========== 9. Évaluation ==========
print("Évaluation du modèle...")
y_pred = clf.predict(X_test_tfidf)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Rapport de classification :\n", classification_report(y_test, y_pred))

# ========== 10. Sauvegarde ==========
print("Sauvegarde du modèle et du vectorizer...")
joblib.dump(clf, "model/fake_news_classifier.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("Modèle et vectorizer sauvegardés dans /model.")
