import streamlit as st
import joblib
import requests
import re  # Import ajouté pour le nettoyage du texte

# Charger le vectorizer TF-IDF
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

st.title("Détection de Fake News")
st.write("Entrez un texte pour savoir s'il est Fake ou Non-Fake.")

# Interface utilisateur : champ de texte
user_input = st.text_area("Texte à analyser", "")

# Fonction de nettoyage du texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Supprime les URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Conserve uniquement les lettres et les espaces
    return text

# Fonction pour prédire le texte via l'API FastAPI
def predict(text, model_choice):
    url = "http://127.0.0.1:8000/predict"
    data = {
        "text": text,
        "model": model_choice
    }
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        st.error("Erreur lors de l'appel à l'API: " + str(e))
        return {}

# Choisir le modèle à utiliser
model_choice = st.selectbox("Choisissez le modèle de détection", ["randomforest", "xgboost", "bert", "gradient_boosting"])

# Affichage des résultats
if st.button("Analyser"):
    if user_input.strip():
        # Nettoyage du texte avant de l'envoyer à l'API
        cleaned_input = clean_text(user_input)
        result = predict(cleaned_input, model_choice)
        if "prediction" in result:
            st.success(f"Résultat de l'analyse : **{result['prediction']}**")
            st.write(f"Confiance : FAKE = {result['probabilities']['FAKE']}% | REAL = {result['probabilities']['REAL']}%")
        else:
            st.error("La réponse de l'API n'a pas le format attendu.")
    else:
        st.warning("Veuillez entrer un texte.")
