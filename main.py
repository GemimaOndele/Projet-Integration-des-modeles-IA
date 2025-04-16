from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import uvicorn
from preprocess import clean_text
import numpy as np
import os
import sys
from transformers import pipeline

# Charger les modèles et vectoriseur
# BERT via Hugging Face
bert_model = pipeline("text-classification", model="bert-base-uncased")
rf_model = joblib.load("model/fake_news_random_forest_classifier.pkl")
xgb_model = joblib.load("model/xgboost_fake_news.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
gradient_boosting = joblib.load("model/gradient_boosting_fake_news.pkl")

app = FastAPI()

class NewsInput(BaseModel):
    text: str
    model: str  # "bert", "randomforest", "xgboost"

@app.get("/")
def home():
    return {"message": "Fake News Detection API is running 🎯"}

@app.post("/predict")
def predict_news(input_data: NewsInput):
    cleaned = clean_text(input_data.text)

    if input_data.model.lower() == "bert":
        prediction = bert_model(cleaned)[0]
        proba = prediction['score']
        return {
            "prediction": "FAKE" if prediction['label'] == "LABEL_1" else "REAL",
            "probabilities": {"FAKE": round(proba, 3), "REAL": round(1 - proba, 3)}
        }

    elif input_data.model.lower() == "randomforest":
        vectorized = vectorizer.transform([cleaned])
        prediction = rf_model.predict(vectorized)[0]
        proba = rf_model.predict_proba(vectorized)[0].tolist()
        return {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "probabilities": {"FAKE": round(proba[1], 3), "REAL": round(proba[0], 3)}
        }

    elif input_data.model.lower() == "xgboost":
        vectorized = vectorizer.transform([cleaned])
        prediction = xgb_model.predict(vectorized)[0]
        proba = xgb_model.predict_proba(vectorized)[0].tolist()
        return {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "probabilities": {"FAKE": round(proba[1], 3), "REAL": round(proba[0], 3)}
        }

    else:
        return {"error": "Unknown model selected"}
