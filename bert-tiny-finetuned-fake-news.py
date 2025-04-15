import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# 1. Charger le dataset
df = pd.read_csv("data/all_news_cleaned.csv")

# 2. Vérifier les colonnes et afficher un aperçu
print("Colonnes disponibles :", df.columns.tolist())
print(df[['title', 'text']].head())

# 3. Préparer les textes (fusionner titre + contenu)
df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')

# 4. Charger le pipeline BERT rapide pour fake news
print("Chargement du modèle BERT compact...")
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

# 5. Prédictions sur tous les textes
print("Prédiction en cours...")
tqdm.pandas()  # Progress bar
df['prediction'] = df['full_text'].progress_apply(lambda x: classifier(x[:512])[0]['label'])  # max 512 tokens

# 6. Afficher les résultats
print(df[['title', 'prediction']].head())

# 7. Sauvegarder le fichier avec les prédictions
df.to_csv("results/bert_fake_news_predictions.csv", index=False)
print("Prédictions enregistrées dans 'results/bert_fake_news_predictions.csv'")
