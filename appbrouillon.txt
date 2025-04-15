from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)

def clean_text(text):
    """
    Fonction de nettoyage du texte :
      - Conversion en minuscules,
      - Suppression des URLs,
      - Suppression des caractères non alphabétiques.
    """
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Charger le modèle et le vectorizer sauvegardés par Esther.
# Assurez-vous que les fichiers "model/fake_news_classifier.pkl" et "model/tfidf_vectorizer.pkl"
# existent et ont été créés par la partie d'Esther.
classifier = joblib.load("model/fake_news_classifier.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint pour prédire la classification d'un texte envoyé en JSON.
    Exemples de JSON attendus : {"text": "Votre texte ici"}
    """
    try:
        data = request.get_json(force=True)
        if "text" not in data:
            return jsonify({"error": "La clé 'text' est requise"}), 400

        text = data["text"]
        # Pré-traiter le texte
        cleaned_text = clean_text(text)
        # Transformer le texte avec le vectorizer
        vect_text = vectorizer.transform([cleaned_text])
        # Prédiction avec le modèle
        prediction = classifier.predict(vect_text)
        # Interprétation de la prédiction (0 : vrai, 1 : fake)
        result = {"fake_news": bool(prediction[0])}

        return jsonify(result)
    except Exception as e:
        # Gestion des erreurs
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
