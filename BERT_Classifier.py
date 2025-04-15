import pandas as pd
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# Fonction de nettoyage du texte
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

# Charger le tokenizer de BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenisation des textes
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=512)

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=512)

# Convertir en format adapté pour PyTorch
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encodings, y_train)
test_dataset = NewsDataset(test_encodings, y_test)

# Charger le modèle BERT pour la classification de séquences
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Arguments pour l'entraînement
training_args = TrainingArguments(
    output_dir='./results',          # Dossier de sortie
    num_train_epochs=3,              # Nombre d'époques
    per_device_train_batch_size=16,  # Taille des batchs pour l'entraînement
    per_device_eval_batch_size=64,   # Taille des batchs pour l'évaluation
    warmup_steps=500,                # Nombre d'étapes de warmup
    weight_decay=0.01,               # Poids de régularisation
    logging_dir='./logs',            # Dossier de logs
    logging_steps=10                 # Log tous les 10 steps
)

# Initialiser le Trainer
trainer = Trainer(
    model=model,                         # Le modèle
    args=training_args,                  # Les arguments d'entraînement
    train_dataset=train_dataset,         # Dataset d'entraînement
    eval_dataset=test_dataset,           # Dataset d'évaluation
    compute_metrics=lambda p: {
        'accuracy': accuracy_score(p.predictions.argmax(axis=-1), p.label_ids)
    }  # Calcul des métriques
)

# Entraînement du modèle
print("Entraînement du modèle BERT...")
trainer.train()
print("Entraînement terminé (BERT).")

# Sauvegarder le modèle et le tokenizer
model.save_pretrained("model/bert_fake_news")
tokenizer.save_pretrained("model/bert_tokenizer")

# Évaluation du modèle
predictions = trainer.predict(test_dataset)
print("Accuracy (BERT):", accuracy_score(y_test, predictions.predictions.argmax(axis=-1)))
print(classification_report(y_test, predictions.predictions.argmax(axis=-1)))
