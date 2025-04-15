import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from imblearn.over_sampling import SMOTE

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# ========== 1. Chargement ==========
print("Chargement du dataset all_news_cleaned.csv...")
df = pd.read_csv("data/all_news_cleaned.csv")
df.columns = df.columns.str.strip()
print("Colonnes disponibles : ", df.columns.tolist())

# Optionnel : Réduction pour test rapide
# df = df.sample(frac=0.2, random_state=42)

# ========== 2. Vérification classes ==========
print("Répartition des classes dans 'label' :")
print(df['label'].value_counts())

# ========== 3. Nettoyage texte ==========
print("Nettoyage des textes...")
df['text'] = df['text'].astype(str).apply(clean_text)

# ========== 4. Split des données ==========
print("Séparation en train/test...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print("Répartition des classes dans y_train :")
print(y_train.value_counts())

# ========== 5. Vectorisation ==========
print("Vectorisation TF-IDF...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF terminé.")

# ========== 6. SMOTE (optionnel) ==========
USE_SMOTE = True

if USE_SMOTE and len(y_train.value_counts()) > 1:
    print("Application de SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)
    print("SMOTE terminé.")
else:
    print("SMOTE désactivé.")
    X_train_smote, y_train_smote = X_train_tfidf, y_train

# ========== 7. Entraînement Random Forest ==========
print("Entraînement du modèle RandomForestClassifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_smote, y_train_smote)
print("Entraînement terminé.")

# ========== 8. Évaluation ==========
print("Évaluation...")
y_pred = clf.predict(X_test_tfidf)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Rapport :\n", classification_report(y_test, y_pred))

# ========== 9. Sauvegarde ==========
print("Sauvegarde du modèle et du vectorizer...")
joblib.dump(clf, "model/fake_news_random_forest_classifier.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
print("Modèle Random Forest sauvegardé.")
