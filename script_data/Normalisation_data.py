#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prétraitement pour la détection de fake news.
Ce script réalise les opérations de nettoyage et de normalisation des textes contenus dans plusieurs datasets (ici fake.csv et true.csv).
Il combine ensuite les deux ensembles pour générer un fichier final nettoyé et peut créer un TF-IDF vectorizer pour préparer les données à l'entraînement.

Ajouts :
- Ajout de logs pour le suivi des étapes.
- Possibilité de sous-échantillonner pour des tests rapides.
- Équilibrage de classes désactivé par défaut pour accélérer.

Auteur: Votre nom
Date: YYYY-MM-DD
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)       # Supprime les URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # Conserve seulement les lettres et les espaces
    return text


def preprocess_dataset(input_csv, label_value=None):
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {input_csv} : {e}")
        return None

    if 'text' not in df.columns:
        print(f"Le dataset {input_csv} doit contenir une colonne 'text'.")
        return None

    print(f"Nettoyage des textes du fichier {input_csv} en cours...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    if label_value is not None:
        df['label'] = label_value

    return df


def combine_datasets(dataset_info, output_csv, vectorizer_path=None, sample_frac=None):
    dfs = []
    for input_csv, label in dataset_info:
        df = preprocess_dataset(input_csv, label_value=label)
        if df is not None:
            if sample_frac:
                df = df.sample(frac=sample_frac, random_state=42)
                print(f"Sous-échantillonnage à {sample_frac*100:.0f}% pour {input_csv}")
            dfs.append(df)

    if not dfs:
        print("Aucun dataset n'a pu être chargé.")
        return

    combined_df = pd.concat(dfs, ignore_index=True)

    print("\nStatistiques sur les données combinées :")
    print("Colonnes :", combined_df.columns)
    print("Distribution des classes :\n", combined_df['label'].value_counts())

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    combined_df.to_csv(output_csv, index=False)
    print(f"\nDataset combiné nettoyé sauvegardé dans '{output_csv}'.")

    if vectorizer_path is not None:
        print("\nVectorisation TF-IDF en cours...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        vectorizer.fit(combined_df['cleaned_text'])
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"TF-IDF vectorizer sauvegardé dans '{vectorizer_path}'.")


if __name__ == '__main__':
    dataset_info = [
        ("fake.csv", 1),
        ("true.csv", 0)
    ]
    output_csv_path = "data/all_news_cleaned.csv"
    vectorizer_save_path = "model/tfidf_vectorizer.pkl"

    # Test rapide avec 20% des données pour accélérer la vectorisation si besoin
    combine_datasets(
        dataset_info,
        output_csv=output_csv_path,
        vectorizer_path=vectorizer_save_path,
        sample_frac=0.2  # Commenter ou mettre None pour utiliser 100%
    )


