# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version optimisée pour Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import base64
import io
import os
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.sav")
        scaler = joblib.load("scaler.sav")
        return model, scaler
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {str(e)}")
        return None, None

model, scaler = load_model()

# Chemins des images (fichiers à la racine)
GENUINE_BILL_IMAGE = "vraibillet.PNG"
FAKE_BILL_IMAGE = "fauxbillet.png"

# Vérification que les images existent
if not os.path.exists(GENUINE_BILL_IMAGE):
    st.error(f"Image manquante: {GENUINE_BILL_IMAGE}")
if not os.path.exists(FAKE_BILL_IMAGE):
    st.error(f"Image manquante: {FAKE_BILL_IMAGE}")

# Fonction pour convertir image en base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# CSS personnalisé
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --success: #4CAF50;
        --danger: #F44336;
        --light: #f8f9fa;
        --dark: #212529;
    }
   
    .header {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
   
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: none;
        position: relative;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 16px rgba(0,0,0,0.1);
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
    }
   
    .genuine-card {
        border-top: 4px solid var(--success);
    }
    
    .genuine-card::before {
        background: linear-gradient(90deg, rgba(76,175,80,0.2), rgba(76,175,80,0));
    }
   
    .fake-card {
        border: 2px solid var(--danger);
    }
    
    .fake-card::before {
        background: linear-gradient(90deg, rgba(244,67,54,0.2), rgba(244,67,54,0));
    }
   
    .stat-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: scale(1.03);
    }
   
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--secondary);
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 1rem;
        color: var(--dark);
        opacity: 0.8;
    }
   
    .show-more-btn {
        margin-top: 0.5rem;
        background: #f0f2f6 !important;
        color: var(--secondary) !important;
        border: 1px solid #ddd !important;
        transition: all 0.3s ease;
    }
    
    .show-more-btn:hover {
        background: var(--secondary) !important;
        color: white !important;
    }
    
    .prediction-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    
    .genuine-badge {
        background-color: rgba(76,175,80,0.2);
        color: var(--success);
    }
    
    .fake-badge {
        background-color: rgba(244,67,54,0.2);
        color: var(--danger);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h2 style="color:white; margin:0;">  📇 Application Scan Franc cfa </h2>
    <p style="color:white; opacity:0.9; margin:10;"> ✅ Solution de détection de faux billets</p>
</div>
""", unsafe_allow_html=True)

# Section Analyse
uploaded_file = st.file_uploader(
    "Faites glisser et déposez le fichier ici ou cliquez sur le bouton 'Browse files' pour Parcourir",
    type=["csv"],
    key="file_uploader"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        
        # Aperçu des données avec prédiction
        st.markdown("#### Aperçu des données")
        
        # Faire la prédiction pour l'aperçu
        if model is not None:
            required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
            if all(col in df.columns for col in required_cols):
                features = df[required_cols].head(5)
                features_scaled = scaler.transform(features)
                probas = model.predict_proba(features_scaled)
                predictions = ["Genuine" if p[1] > 0.5 else "Fake" for p in probas]
                
                # Créer un DataFrame pour l'affichage avec les prédictions
                preview_df = df.head(5).copy()
                preview_df['Prediction'] = predictions
                
                # Afficher le tableau avec les prédictions
                table_placeholder = st.empty()
                table_placeholder.dataframe(preview_df, height=210, use_container_width=True)
            else:
                table_placeholder = st.empty()
                table_placeholder.dataframe(df.head(5), height=210, use_container_width=True)
                st.warning("Les colonnes requises pour la prédiction ne sont pas toutes présentes dans le fichier.")
        else:
            table_placeholder = st.empty()
            table_placeholder.dataframe(df.head(5), height=210, use_container_width=True)
       
        if len(df) > 5:
            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
                if model is not None and all(col in df.columns for col in required_cols):
                    features = df[required_cols]
                    features_scaled = scaler.transform(features)
                    probas = model.predict_proba(features_scaled)
                    predictions = ["Genuine" if p[1] > 0.5 else "Fake" for p in probas]
                    full_df = df.copy()
                    full_df['Prediction'] = predictions
                    table_placeholder.dataframe(full_df, height=min(800, len(df)*35), use_container_width=True)
                else:
                    table_placeholder.dataframe(df, height=min(800, len(df)*35), use_container_width=True)
       
        if st.button("Lancer la détection", key="analyze_btn"):
            with st.spinner("Analyse en cours..."):
                if model is None:
                    st.error("Modèle non chargé - Impossible d'effectuer la prédiction")
                else:
                    try:
                        required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                        if not all(col in df.columns for col in required_cols):
                            raise ValueError("Colonnes requises manquantes dans le fichier CSV")
                           
                        features = df[required_cols]
                        features_scaled = scaler.transform(features)
                        probas = model.predict_proba(features_scaled)
                       
                        predictions = [{
                            'id': i,
                            'prediction': "Genuine" if p[1] > 0.5 else "Fake",
                            'probability': p[1]
                        } for i, p in enumerate(probas)]
                       
                        st.success("Analyse terminée avec succès !")
                       
                        # Affichage des résultats
                        st.markdown("#### Résultats de la détection")
                        genuine_img = image_to_base64(GENUINE_BILL_IMAGE)
                        fake_img = image_to_base64(FAKE_BILL_IMAGE)
                       
                        cols_per_row = 3
                        for i in range(0, len(predictions), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < len(predictions):
                                    pred = predictions[i + j]
                                    with cols[j]:
                                        if pred['prediction'] == 'Genuine':
                                            st.markdown(f"""
                                            <div class="card genuine-card">
                                                <div style="display:flex; align-items:center; justify-content:space-between;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--dark); margin:0 0 0.5rem 0;">Billet n°{pred['id'] + 1}</h5>
                                                        <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                                                            <span style="font-size:1.2rem; font-weight:600; color:var(--success);">Authentique</span>
                                                            <span class="prediction-badge genuine-badge">{(pred['probability']*100):.1f}%</span>
                                                        </div>
                                                        <div style="height:8px; background:#e9ecef; border-radius:4px; overflow:hidden;">
                                                            <div style="height:100%; width:{pred['probability']*100}%; background:var(--success); border-radius:4px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1.5rem;">
                                                        <img src="data:image/png;base64,{genuine_img}" width="90" style="border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="card fake-card">
                                                <div style="display:flex; align-items:center; justify-content:space-between;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--dark); margin:0 0 0.5rem 0;">Billet n°{pred['id'] + 1}</h5>
                                                        <div style="display:flex; align-items:center; margin-bottom:0.5rem;">
                                                            <span style="font-size:1.2rem; font-weight:600; color:var(--danger);">Faux</span>
                                                            <span class="prediction-badge fake-badge">{(1-pred['probability'])*100:.1f}%</span>
                                                        </div>
                                                        <div style="height:8px; background:#e9ecef; border-radius:4px; overflow:hidden;">
                                                            <div style="height:100%; width:{(1-pred['probability'])*100}%; background:var(--danger); border-radius:4px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1.5rem;">
                                                        <img src="data:image/png;base64,{fake_img}" width="90" style="border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1);">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                       
                        # Statistiques
                        genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                        fake_count = len(predictions) - genuine_count
                       
                        st.markdown("<h4 style='text-align: center; margin-top:2rem;'>Statistiques de détection</h4>", unsafe_allow_html=True)
                       
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{len(predictions)}</div>
                                <div class="stat-label">Billets analysés</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color:var(--success);">{genuine_count}</div>
                                <div class="stat-label">Authentiques</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col3:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color:var(--danger);">{fake_count}</div>
                                <div class="stat-label">Faux billets</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        # Graphique
                        st.markdown("<h4 style='text-align: center; margin-top:2rem;'>Répartition des résultats</h4>", unsafe_allow_html=True)
                        fig = px.pie(
                            names=['Authentiques', 'Faux'],
                            values=[genuine_count, fake_count],
                            color=['Authentiques', 'Faux'],
                            color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
                            hole=0.4
                        )
                        fig.update_layout(
                            showlegend=True, 
                            margin=dict(l=20, r=20, t=30, b=20),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        st.markdown("""
        <div class="card fake-card">
            <h5 style="color:var(--danger);">Format de fichier incorrect</h5>
            <p>Assurez-vous que votre fichier :</p>
            <ul>
                <li>Est un CSV valide</li>
                <li>Utilise le point-virgule (;) comme séparateur</li>
                <li>Contient les colonnes requises</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

