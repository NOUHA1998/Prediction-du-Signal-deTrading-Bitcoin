import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import io

# CSS pour personnaliser l'interface
st.markdown(
    """
    <style>
        .stApp {
            background-image: url('/content/Arrière_plan.jpg');  /* Chemin local */
            background-size: cover;
            background-position: center;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Charger le modèle
try:
    model = joblib.load('bitcoin_signal_model.pkl')
except FileNotFoundError:
    model = None
    st.error("Le modèle 'bitcoin_signal_model.pkl' est introuvable. Assurez-vous qu'il est dans le même dossier que 'app.py'.")

# Sommaire interactif
st.sidebar.image("/content/Logo.png", caption="Bitcoin Market", use_container_width=True)


st.sidebar.title("Sommaire")
menu = st.sidebar.radio(
    "Naviguer vers :",
    ["Introduction", "Compréhension des données" , "Préparation des données", "Visualisation des données", "Modélisation et Évaluation"]
)

# Section : Introduction
if menu == "Introduction":
    st.title("📈 Prédiction du Signal de Trading Bitcoin")
    st.write(
        """
        Bienvenue dans cette application interactive dédiée à la prédiction des signaux de trading pour le Bitcoin.
        
        Cette application utilise des données historiques sur les prix et les volumes échangés pour prédire si un signal de marché représente un **achat** ou une **vente**.
        
        L'application vous permet d'explorer différentes étapes du processus de prédiction :
        - **Compréhension des données** : Explorez les données utilisées dans cette application.
        - **Préparation des données** : Découvrez les étapes de nettoyage et de transformation des données avant l'entraînement des modèles.
        - **Visualisation des données** : Visualisez les tendances des prix et des volumes échangés au fil du temps.
        - **Modélisation et Évaluation** : Apprenez comment les modèles sont créés et évalués pour prédire les signaux de trading.
        
        Cliquez sur les autres sections pour en savoir plus sur chaque étape du processus.
        """
    )
# Section : Compréhension des données
elif menu == "Compréhension des données":
    st.title("📊 Compréhension des données")
    st.write(
        """
        Cette section fournit une vue d'ensemble des données utilisées dans cette application pour prédire les signaux de trading Bitcoin.
        
        Les données comprennent :
        - **Prix d'ouverture** : Le prix auquel le Bitcoin a été échangé lors de l'ouverture de la période.
        - **Prix de clôture** : Le prix auquel le Bitcoin a été échangé à la fin de la période.
        - **Volume échangé** : Le nombre total de Bitcoins échangés pendant cette période.
        
        Ces informations sont utilisées pour prédire si un signal de marché correspond à un **achat** ou à une **vente**.
        """
    )
    
    # Affichage des premières lignes du jeu de données
    st.subheader("Aperçu des données")
    st.write("Voici un aperçu des premières lignes du jeu de données :")
    st.dataframe(data.head())
    
    # Dimensions et informations des données
    st.subheader("Dimensions et informations")
    st.write(f"Le jeu de données contient **{data.shape[0]} lignes** et **{data.shape[1]} colonnes**.")
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write("Résumé statistique des colonnes numériques :")
    st.dataframe(data.describe())
    
    # Répartition des classes
    st.subheader("Répartition des classes")
    st.write("Répartition des signaux d'achat (1) et de vente (0) :")
    class_distribution = data['Signal'].value_counts()
    st.bar_chart(class_distribution)
    
    # Visualisation des colonnes importantes
    st.subheader("Visualisation des données")
    st.write("Évolution des prix de clôture au fil du temps :")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x=data.index, y='Close', ax=ax)
    ax.set_title("Évolution des prix de clôture")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Prix de clôture")
    st.pyplot(fig)

# Section : Préparation des données
elif menu == "Préparation des données":
    st.header("⚙️ Préparation des données")
    
    st.write(
        
  
        """
        Dans cette section, nous détaillons les étapes de préparation des données utilisées pour entraîner notre modèle de prédiction des signaux de trading Bitcoin.
        
        ### 1. Gestion des valeurs manquantes
        Les valeurs manquantes ont été identifiées et supprimées afin de garantir la qualité des données utilisées pour l'entraînement du modèle.
        
        ### 2. Suppression des doublons
        Les lignes dupliquées ont été supprimées pour éviter de fausser les résultats.
        
        ### 3. Suppression de colonnes inutiles
        Certaines colonnes redondantes ont été retirées, notamment la colonne `column_to_remove`, pour garder uniquement les données pertinentes pour notre modèle.
        
        ### 4. Création de nouvelles caractéristiques
        - **Moyenne mobile du volume** : La moyenne mobile sur 7 jours du volume échangé a été calculée.
        - **Identification des pics de volume** : Les périodes où le volume échangé dépasse deux fois la moyenne mobile ont été étiquetées comme des "pics de volume".
        - **Direction du prix** : Une nouvelle caractéristique a été créée pour capturer la direction du prix (hausse ou baisse).
        
        ### 5. Création des signaux de trading
        Les données ont été étiquetées en fonction des conditions de marché :
        - Signal d'achat (1) si un pic de volume se produit et que le prix augmente.
        - Signal de vente (0) si un pic de volume se produit et que le prix baisse.
        
        ### 6. Séparation des données
        Les données ont été divisées en ensembles d’entraînement (80%) et de test (20%) pour évaluer la performance du modèle.
        
        ### 7. Normalisation des caractéristiques
        Toutes les caractéristiques d'entrée ont été normalisées pour garantir des performances optimales du modèle.
        
        Ces étapes sont essentielles pour préparer les données de manière appropriée avant de les utiliser dans le modèle de prédiction.
        """
    )


# Section : Visualisation des donnés

elif menu == "Visualisation des données":
    st.header("🔍 Visualisation des données")
    
    # Aperçu des données
    st.write("Voici un aperçu des données utilisées pour prédire les signaux de trading Bitcoin :")
    st.dataframe(data.head())  # Afficher un échantillon des données

    # Visualisation de la distribution des signaux
    st.subheader("Distribution des classes de signal (Achat vs Vente)")
    fig, ax = plt.subplots()
    sns.countplot(x='Signal', data=data, palette={'0': 'red', '1': 'green'}, ax=ax)
    ax.set_title('Distribution des classes de Signal')
    ax.set_xlabel('Signal (0 = Vente, 1 = Achat)')
    ax.set_ylabel('Nombre d\'observations')
    st.pyplot(fig)

    # Visualisation de la matrice de corrélation
    st.subheader("Matrice de corrélation")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('Matrice de Corrélation des Variables')
    st.pyplot(fig)

    # Visualisation de la distribution des prix de clôture
    st.subheader("Distribution des prix de clôture")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(data_filtered['Close'], bins=30, kde=True, color='blue', ax=ax)
    mean_close = data_filtered['Close'].mean()
    median_close = data_filtered['Close'].median()
    ax.axvline(mean_close, color='green', linestyle='--', label=f'Moyenne: {mean_close:.2f}')
    ax.axvline(median_close, color='red', linestyle='--', label=f'Médiane: {median_close:.2f}')
    ax.set_title('Distribution des prix de clôture')
    ax.set_xlabel('Prix de clôture')
    ax.set_ylabel('Fréquence')
    ax.legend()
    st.pyplot(fig)

    # Visualisation de l'évolution des prix de clôture au fil du temps
    st.subheader("Évolution des prix de clôture au fil du temps")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Timestamp'], data['Close'], label='Prix de clôture', color='green')
    ax.set_title('Évolution des prix de clôture')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Prix de clôture')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisation de l'évolution du volume échangé en BTC
    st.subheader("Évolution du volume échangé en BTC")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data_filtered['Timestamp'], data_filtered['Volume_(BTC)'], label='Volume échangé (BTC)', color='purple')
    ax.set_title('Évolution du volume échangé en BTC au fil du temps')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Volume échangé (BTC)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Visualisation des signaux de trading en fonction des pics de volume
    st.subheader("Signaux d\'achat et de vente basés sur les pics de volume")
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(data_filtered['Timestamp'], data_filtered['Close'], label='Prix de clôture', color='green')
    ax.scatter(data_filtered[data_filtered['Signal'] == 1]['Timestamp'], data_filtered[data_filtered['Signal'] == 1]['Close'],
               color='blue', label='Signal d\'achat', marker='^')
    ax.scatter(data_filtered[data_filtered['Signal'] == 0]['Timestamp'], data_filtered[data_filtered['Signal'] == 0]['Close'],
               color='red', label='Signal de vente', marker='v')
    ax.set_title('Signaux d\'achat et de vente')
    ax.set_xlabel('Temps')
    ax.set_ylabel('Prix de clôture')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Section : Modélisation et Évaluation
elif menu == "Modélisation et Évaluation":
    st.header("📈 Modélisation et Évaluation")
    st.write("Entrez les caractéristiques pour prédire un signal de trading.")
    
    # Liste des modèles et résultats
    models = [
        ('Random Forest', rf_model),
        ('SVM', svm_model),
        ('Logistic Regression', lr_model),
        ('KNN', knn_model),
        ('MLPClassifier', nn_model)
    ]
    
    # Affichage des résultats
    st.subheader("Résultats des modèles")
    
    # Tableau des résultats
    results_df = pd.DataFrame(results, columns=metrics, index=[model_name for model_name, _ in models])
    st.dataframe(results_df)
    
    # Carte thermique des performances
    st.subheader("Comparaison des Performances des Modèles")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(results_df, annot=True, cmap='YlGnBu', fmt='.3f', ax=ax)
    st.pyplot(fig)
    
    # Diagramme en barres des performances
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax, color=[colors[model_name] for model_name in results_df.index])
    ax.set_title('Comparaison des Performances des Modèles')
    ax.set_ylabel('Score')
    ax.set_xlabel('Modèles')
    st.pyplot(fig)
  

    # Sélectionner un modèle pour afficher ses courbes ROC et la matrice de confusion
    model_choice = st.selectbox("Choisir un modèle pour afficher les résultats détaillés:", models)
    selected_model = model_choice[1]  # Le modèle sélectionné

    # Expliquer le choix du modèle Random Forest
    st.subheader("Modèle choisi : Random Forest")
    st.write(
        """
        Le modèle **Random Forest** a été choisi comme modèle final en raison de ses performances supérieures par rapport aux autres modèles testés.
        
        ### Pourquoi Random Forest ?
        - **Meilleur compromis entre précision et rappel** : Random Forest a un F1-score plus élevé que les autres modèles.
        - **Robustesse** : Random Forest gère bien les données déséquilibrées et est moins sensible au surajustement (overfitting).
        - **Interprétabilité** : Les arbres de décision sous-jacents permettent de comprendre les décisions du modèle.
        
        ### Performances de Random Forest :
        """
    )
    # Afficher les performances de Random Forest
    st.write("**Résultats de Random Forest :**")
    st.write(f"- **Accuracy** : {results['Random Forest']['Accuracy']:.4f}")
    st.write(f"- **Précision** : {results['Random Forest']['Precision']:.4f}")
    st.write(f"- **Rappel** : {results['Random Forest']['Recall']:.4f}")
    st.write(f"- **F1-score** : {results['Random Forest']['F1 Score']:.4f}")

    # Matrice de confusion
    st.subheader("Matrice de confusion - Random Forest")
    y_pred = selected_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'], ax=ax)
    st.pyplot(fig)
    
    # Courbe ROC
    st.subheader("Courbe ROC - Random Forest")
    fpr, tpr, _ = roc_curve(y_test, selected_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='blue', linestyle='--')
    ax.set_title(f'Courbe ROC - {model_choice[0]}')
    ax.set_xlabel('Taux de Faux Positifs (FPR)')
    ax.set_ylabel('Taux de Vrais Positifs (TPR)')
    ax.legend(loc='lower right')
    st.pyplot(fig)

      # Entrée utilisateur
st.subheader("Prédiction du signal de trading")
avg_7_days = st.number_input("Moyenne mobile sur 7 jours", value=0.0, step=0.01)
avg_30_days = st.number_input("Moyenne mobile sur 30 jours", value=0.0, step=0.01)

open_price = st.number_input("Prix d'ouverture (Open)", value=0.0, step=0.01)
high_price = st.number_input("Prix le plus haut (High)", value=0.0, step=0.01)
low_price = st.number_input("Prix le plus bas (Low)", value=0.0, step=0.01)
close_price = st.number_input("Prix de clôture (Close)", value=0.0, step=0.01)

# Bouton de prédiction
if st.button("Prédire"):
    try:
        # Vérifier si le modèle est disponible
        if model:
            # Créer une DataFrame avec les données saisies par l'utilisateur
            input_data = pd.DataFrame({
                '7_day_avg': [avg_7_days],
                '30_day_avg': [avg_30_days],
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Close': [close_price]
            })

          # Effectuer la prédiction
            prediction = model.predict(input_data)
            
            # Traduire la prédiction en signal
            signal = "Acheter ✅" if prediction[0] == 1 else "Vendre ❌"
            
            # Afficher le résultat
            st.success(f"### Signal prédit : {signal}")
        else:
            st.error("Modèle non chargé. Veuillez vérifier le chargement du modèle.")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
