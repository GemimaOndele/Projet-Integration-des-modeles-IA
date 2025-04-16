import streamlit as st
import pandas as pd
import os
import datetime

# Définir le chemin de stockage des feedbacks
feedback_file = "results/feedback.csv"

# Charger les feedbacks existants s'ils existent, sinon créer un DataFrame vide
if os.path.exists(feedback_file):
    feedback_df = pd.read_csv(feedback_file)
else:
    feedback_df = pd.DataFrame(columns=["timestamp", "model", "text", "predicted", "feedback", "comment"])

st.title("Système de Feedback - Détection de Fake News")

# Section pour envoyer un feedback
st.header("Envoyer un signalement")
with st.form("feedback_form"):
    st.write("Veuillez signaler si la prédiction affichée semble incorrecte.")
    
    model_used = st.selectbox("Modèle utilisé", ["bert", "randomforest", "xgboost", "gradient_boosting"])
    original_text = st.text_area("Texte analysé", "")
    predicted_label = st.selectbox("Prédiction affichée", ["Fake", "Real"])
    feedback_correct = st.radio("Le résultat était-il correct ?", ("Oui", "Non"))
    comment = st.text_area("Commentaire (facultatif)", "")
    
    submit_button = st.form_submit_button("Envoyer le feedback")
    
    if submit_button:
        timestamp = datetime.datetime.now().isoformat()
        # Créer une nouvelle entrée de feedback
        new_feedback = {
            "timestamp": timestamp,
            "model": model_used,
            "text": original_text,
            "predicted": predicted_label,
            "feedback": "Correct" if feedback_correct == "Oui" else "Incorrect",
            "comment": comment
        }
        # ✅ Remplace append obsolète par concat
        feedback_df = pd.concat([feedback_df, pd.DataFrame([new_feedback])], ignore_index=True)
        
        # Sauvegarder dans un fichier CSV
        os.makedirs("results", exist_ok=True)
        feedback_df.to_csv(feedback_file, index=False)
        st.success("Feedback enregistré avec succès !")


# Section Dashboard pour visualiser les feedbacks
st.header("Dashboard de Feedback")

if feedback_df.empty:
    st.write("Aucun feedback enregistré pour le moment.")
else:
    st.subheader("Feedbacks récents")
    st.dataframe(feedback_df.head(10))
    
    st.subheader("Statistiques des feedbacks")
    feedback_counts = feedback_df["feedback"].value_counts()
    st.write(feedback_counts)
    st.bar_chart(feedback_counts)
