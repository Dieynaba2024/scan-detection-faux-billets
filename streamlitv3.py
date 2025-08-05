# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version ultra-optimisée
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import io
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Configuration initiale optimisée
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💰",
    layout="centered",  # Plus rapide que "wide"
    initial_sidebar_state="collapsed"  # Chargement plus rapide
)

# 2. Cache optimisé pour le modèle
@st.cache_resource(ttl=3600, show_spinner=False)  # Désactive le spinner pour le cache
def load_model():
    try:
        model = joblib.load("random_forest_model.sav")
        scaler = joblib.load("scaler.sav")
        return model, scaler
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return None, None

# 3. Chargement anticipé des ressources
@st.cache_data
def load_images():
    GENUINE_IMG = "vraibillet.PNG"
    FAKE_IMG = "fauxbillet.png"
    
    if not all(os.path.exists(img) for img in [GENUINE_IMG, FAKE_IMG]):
        st.error("Images manquantes!")
        return None, None
    
    def img_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    return img_to_base64(GENUINE_IMG), img_to_base64(FAKE_IMG)

# Chargement initial
model, scaler = load_model()
genuine_img, fake_img = load_images()

# 4. CSS optimisé (chargé une seule fois)
st.markdown("""
<style>
    .card {
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        padding: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .stat-card {
        text-align: center;
        padding: 0.6rem;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# 5. Interface simplifiée
st.title("💰 Détection de Faux Billets")
st.caption("Solution optimisée pour performance maximale")

# 6. Traitement des fichiers avec cache
@st.cache_data(ttl=300, max_entries=3)  # Limite le cache à 3 fichiers
def process_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        return df, None
    except Exception as e:
        return None, str(e)

uploaded_file = st.file_uploader("Téléversez votre fichier CSV", type=["csv"])

if uploaded_file:
    df, error = process_file(uploaded_file)
    
    if error:
        st.error(f"Erreur: {error}")
    else:
        # 7. Affichage optimisé des données
        with st.expander("Aperçu des données", expanded=False):
            st.dataframe(df.head(5), use_container_width=True)

        if st.button("Analyser", type="primary"):
            if model is None:
                st.error("Modèle non disponible")
            else:
                with st.spinner("Analyse en cours..."):
                    try:
                        # 8. Prédiction optimisée
                        required_cols = ['diagonal', 'height_left', 'height_right', 
                                       'margin_low', 'margin_up', 'length']
                        features = df[required_cols].values
                        features_scaled = scaler.transform(features)
                        probas = model.predict_proba(features_scaled)
                        
                        # 9. Calcul des stats sans stocker toutes les prédictions
                        genuine_count = np.sum(probas[:, 1] > 0.5)
                        fake_count = len(probas) - genuine_count
                        
                        # 10. Affichage optimisé des résultats
                        st.success("Analyse terminée!")
                        
                        # Stats immédiates
                        cols = st.columns(3)
                        cols[0].metric("Billets analysés", len(probas))
                        cols[1].metric("Authentiques", genuine_count, delta_color="off")
                        cols[2].metric("Faux", fake_count, delta_color="off")
                        
                        # Graphique simplifié
                        fig = px.pie(
                            names=['Authentiques', 'Faux'],
                            values=[genuine_count, fake_count],
                            hole=0.5,
                            width=300, height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 11. Affichage partiel des résultats (limité à 20)
                        st.subheader("Exemples de résultats")
                        sample_indices = np.random.choice(len(probas), min(20, len(probas)), replace=False)
                        
                        for idx in sample_indices:
                            prob = probas[idx, 1]
                            is_genuine = prob > 0.5
                            
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.progress(prob if is_genuine else (1-prob), 
                                           f"Billet #{idx+1} - {'Authentique' if is_genuine else 'Faux'}")
                            with col2:
                                st.image(f"data:image/png;base64,{genuine_img if is_genuine else fake_img}", 
                                        width=60)
                        
                    except Exception as e:
                        st.error(f"Erreur d'analyse: {str(e)}")

# 12. Pied de page optimisé
#st.caption("© 2023 - Application optimisée pour Streamlit Cloud")
