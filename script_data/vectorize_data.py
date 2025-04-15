# vectorize_data.py

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

# Chemins des fichiers
input_csv_path = "data/all_news_cleaned.csv"
vectorizer_path = "model/tfidf_vectorizer.pkl"

# Chargement des données nettoyées
df = pd.read_csv(input_csv_path)
print(f"{len(df)} lignes chargées depuis {input_csv_path}")

# Vérification des NaN dans la colonne 'cleaned_text' et suppression ou remplacement
df['cleaned_text'].fillna('', inplace=True)  # Remplacer les NaN par une chaîne vide
# df = df.dropna(subset=['cleaned_text'])  # Alternative : supprimer les lignes avec NaN

# Chargement du vectorizer TF-IDF
vectorizer = joblib.load(vectorizer_path)

# Transformation des textes
X = vectorizer.transform(df['cleaned_text'])
y = df['label'].values

print(f"Forme des données vectorisées : {X.shape}")
print(f"Exemple de label : {y[:5]}")

# Optionnel : division en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarde des jeux de données
joblib.dump((X_train, X_test, y_train, y_test), "model/tfidf_data_split.pkl")
print("Enregistrement terminé dans 'model/tfidf_data_split.pkl'")
