**Projet-Intégration-des-modèles-IA**

# Fake News Detector 📰🚫

Ce projet permet de détecter si une news est vraie ou fausse à l’aide d’un modèle IA entraîné sur un corpus nettoyé de fausses et vraies actualités.

## 📁 Structure

- `data/`: Données nettoyées
- `model/`: Modèle entraîné et TF-IDF vectorizer
- `scripts/`: Scripts de préparation
- `interface.py`: Interface Streamlit pour la détection des fakes news et le choix des modèles de test.
- `feedback_dashboard.py`: Interface Streamlit du tableau de board pour le sytèmes des feedbacks
- `main.py`: Fichier de l'api FastAPI

## ⚙️ Lancer l’application

```bash

activate ou .\.venv\Scripts\activate #Commandes windows dans le terminal powershell de vscode du projet pour activer l'environement virtuel du dossier ".venv"

pip install -r requirements.txt #Pour installer toutes librairies du projet

uvicorn main:app --reload #commmande pour Lancer l'api FastAPI, il joue aussi le role d'un serveur pour les deux interfaces.

http://127.0.0.1:8000/ #Après avoir lancer l'API, il tourne dans le port 80, c'est lien de l'api FastAPI

streamlit run interface.py #commande pour lancer l'interface pour la détection des fakesnews

http://localhost:8501/ #Après avoir lancer l'interface de détection des fakes news, il tourne dans port 85, c'est lien de l'interface interface.py

streamlit run feeback_dashboard.py #commande pour lancer l'interface du dashboard des feedback

http://localhost:8502/ #Après avoir lancer l'interface du système des feebacks, il tourne dans port 8502, c'est lien de l'interface feeback_dashboard.py
 

🔗 https://sites.google.com/chromium.org/driver/ #Télécharger le web driver, cette extension est utile pour faire le scrapping des nouveaux articles afin de tester les modèles.



**Détails du projet ** :
**Un projet autour de la matière intégration des modèles IA**

        Sujet: Détection de fake news avec IA.

**1. Objectifs du Projet**
   
But principal : Développer une solution intelligente capable de détecter les fake news à partir des textes issus des actualités.

**Sous-objectifs :**

Définir des critères précis pour distinguer les fake news des informations vérifiées.

Entraîner un modèle NLP robuste basé sur un corpus de fact-checking ou des datasets publics existants.

Mettre en place un processus de prétraitement et de vectorisation des données textuelles.

Intégrer la solution dans un outil pratique (plugin navigateur ou bot Twitter).

Tester et généraliser la solution sur diverses sources d’actualité avec une boucle de feedback pour l’amélioration continue.

**2. Décomposition par Étapes du Projet**
   
**a. Prompter : Définir les Critères de Classification**

Établir une Taxonomie :

Identifier les éléments caractéristiques des fake news (par exemple, absence de sources fiables, tonalité sensationnaliste, incohérences factuelles, etc.).

Définir des indicateurs quantitatifs et qualitatifs (nombre de sources citées, mots-clés suspects, tonalité émotionnelle, etc.).

Recherche Documentaire :

Consulter des études de cas et des articles scientifiques sur la détection de fake news pour s’inspirer des critères retenus par la communauté (cette étape peut inclure la consultation de ressources comme des revues académiques et des tutoriels en ligne).

**b. Modéliser : Entraînement d’un Modèle NLP**

Choix du Dataset :

Identifier ou constituer un corpus de données de fact-checking et de nouvelles vérifiées.

Quelques datasets connus : FakeNewsNet, LIAR dataset ou d’autres corpus disponibles sur Kaggle.

Sélection de l’Architecture :

Utiliser des modèles pré-entraînés tels que BERT, RoBERTa ou GPT-2 adaptés aux tâches de classification textuelle.

Envisager le fine-tuning de ces modèles sur le dataset choisi pour améliorer leur précision dans le contexte spécifique de la détection des fake news.

Entraînement :

Configurer un pipeline d’entraînement avec validation croisée pour éviter le sur-ajustement.

Mettre en place des métriques d’évaluation pertinentes (précision, rappel, F1-score) afin de suivre les performances du modèle.

**c. Processer : Extraction et Vectorisation des Textes de News**

Prétraitement des Données :

Nettoyer les textes (suppression des balises HTML, des stop-words, des signes de ponctuation inutiles, etc.).

Traiter les textes pour harmoniser la casse, éliminer les doublons, et corriger les erreurs de typographie.

Feature Engineering et Vectorisation :

Convertir les textes en vecteurs à l’aide de techniques comme TF-IDF, word embeddings (Word2Vec, GloVe) ou directement avec l’encodeur du modèle pré-entraîné utilisé.

Explorer d’éventuelles enrichissements tels que l’extraction de caractéristiques sentimentales ou syntactiques.

**d. Robotiser : Intégration dans une Application**

Choix de la Plateforme :

Plugin navigateur : Développer une extension (par exemple en JavaScript) qui envoie les articles à analyser à l’API de ton modèle.

Bot Twitter : Créer un bot qui analyse en temps réel les tweets ou les articles partagés sur Twitter en s’appuyant sur l’API Twitter et sur ton modèle de détection.

Développement de l’API :

Construire une API REST ou GraphQL qui encapsule l’appel au modèle NLP.

Utiliser des frameworks tels que Flask, FastAPI ou Django (Python) pour déployer l’API.

Assurer une bonne gestion des erreurs et de la sécurité lors de l’intégration.

Interface Utilisateur :

Concevoir une interface simple pour visualiser la probabilité qu’un article soit une fake news (par exemple, un score ou une alerte en temps réel).

**e. Généraliser : Tester sur Différentes Sources d’Actualité**

Évaluation sur des Jeux de Données Diversifiés :

Tester le modèle avec des articles provenant de différents sites d’information, blogs ou réseaux sociaux.

Évaluer la performance du modèle pour s’assurer qu’il n’est pas trop spécialisé sur un seul type de contenu ou sur une seule source d’actualité.

Itérations et Ajustements :

En fonction des tests, affiner les critères définis en phase de prompter et réajuster le modèle ou le prétraitement si nécessaire.

**f. Data-driven : Amélioration Continue via Feedback**

Mise en place d’un Système de Feedback :

Intégrer un mécanisme permettant aux utilisateurs finaux (ou aux testeurs) de signaler de fausses classifications, d’ajouter des commentaires ou de fournir des évaluations.

Mettre à jour périodiquement le dataset avec de nouvelles données et feedback pour réentraîner le modèle.

Suivi et Analyse de la Performance :

Mettre en place des dashboards pour suivre les indicateurs de performance en temps réel.

Évaluer l’impact des mises à jour et mettre en place des A/B tests pour comparer les versions successives du modèle.

Ce projet a été réalisé par **@Hubert CHAVASSE, @Hadrien LENNON, @Mohamed GHARMAOUI, @Gemima ONDELE POUROU et @Niangoran Esther BOKA.**
