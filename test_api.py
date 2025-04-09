import requests
import json

# URL de l'API en local
url = "http://127.0.0.1:5000/predict"

# Exemple de payload pour tester l'API
payload = {
    "text": "Ceci est un exemple de texte pour tester la détection des fake news."
}

try:
    response = requests.post(url, json=payload)
    print("Statut de la réponse :", response.status_code)
    print("Contenu de la réponse :", response.json())
except Exception as e:
    print("Erreur lors de l'appel API :", str(e))
