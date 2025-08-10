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
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        padding: 1rem;
        margin-bottom: 1rem;
    }
   
    .genuine-card {
        border-left: 4px solid var(--success);
    }
   
    .fake-card {
        border-left: 4px solid var(--danger);
    }
   
    .stat-card {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
   
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 1rem;
        color: #6c757d;
        font-weight: 500;
    }
    
    .stat-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
        color: var(--secondary);
    }
   
    .show-more-btn {
        margin-top: 0.5rem;
        background: #f0f2f6 !important;
        color: var(--secondary) !important;
        border: 1px solid #ddd !important;
    }
    
    .result-item {
        display: flex;
        align-items: center;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .result-icon {
        font-size: 1.8rem;
        margin-right: 1rem;
        min-width: 50px;
        text-align: center;
    }
    
    .result-content {
        flex: 1;
    }
    
    .result-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .result-progress {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        margin-top: 0.5rem;
        overflow: hidden;
    }
    
    .result-progress-bar {
        height: 100%;
    }
    
    .result-image {
        margin-left: 1rem;
        border-radius: 6px;
        width: 80px;
        height: 80px;
        object-fit: cover;
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
        
        # Préparation des données pour l'affichage
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
                        
                        # Ajout de la colonne de prédiction au DataFrame
                        df['Prédiction'] = ['Genuine' if p[1] > 0.5 else 'Fake' for p in probas]
                        df['Probabilité'] = [p[1] if p[1] > 0.5 else 1-p[1] for p in probas]
                       
                        st.success("Analyse terminée avec succès !")
                       
                        # Affichage de l'aperçu avec prédiction
                        preview_rows = 5
                        table_placeholder = st.empty()
                        table_placeholder.dataframe(df[['id'] + required_cols + ['Prédiction', 'Probabilité']].head(preview_rows), 
                                                  height=210, use_container_width=True)
       
                        if len(df) > preview_rows:
                            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
                                table_placeholder.dataframe(df[['id'] + required_cols + ['Prédiction', 'Probabilité']], 
                                                          height=min(800, len(df)*35), use_container_width=True)
                       
                        # Affichage des résultats sous forme de liste
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
                                            <div class="result-item" style="border-left: 4px solid var(--success);">
                                                <div class="result-icon" style="color: var(--success);">✓</div>
                                                <div class="result-content">
                                                    <div class="result-title">Billet n°{pred['id']} - Authentique</div>
                                                    <div>Probabilité: <strong>{pred['probability']*100:.1f}%</strong></div>
                                                    <div class="result-progress">
                                                        <div class="result-progress-bar" style="width: {pred['probability']*100}%; background: var(--success);"></div>
                                                    </div>
                                                </div>
                                                <img src="data:image/png;base64,{genuine_img}" class="result-image">
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="result-item" style="border-left: 4px solid var(--danger);">
                                                <div class="result-icon" style="color: var(--danger);">✗</div>
                                                <div class="result-content">
                                                    <div class="result-title">Billet n°{pred['id']} - Faux</div>
                                                    <div>Probabilité: <strong>{(1-pred['probability'])*100:.1f}%</strong></div>
                                                    <div class="result-progress">
                                                        <div class="result-progress-bar" style="width: {(1-pred['probability'])*100}%; background: var(--danger);"></div>
                                                    </div>
                                                </div>
                                                <img src="data:image/png;base64,{fake_img}" class="result-image">
                                            </div>
                                            """, unsafe_allow_html=True)
                       
                        # Statistiques améliorées
                        genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                        fake_count = len(predictions) - genuine_count
                        genuine_percent = (genuine_count / len(predictions)) * 100
                        fake_percent = (fake_count / len(predictions)) * 100
                       
                        st.markdown("<h4 style='text-align: center; margin-top: 2rem;'>Statistiques de détection</h4>", unsafe_allow_html=True)
                       
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">📊</div>
                                <div class="stat-value">{len(predictions)}</div>
                                <div class="stat-label">Total analysés</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">✅</div>
                                <div class="stat-value" style="color:var(--success);">{genuine_count}</div>
                                <div class="stat-label">Authentiques ({genuine_percent:.1f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        with col3:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">❌</div>
                                <div class="stat-value" style="color:var(--danger);">{fake_count}</div>
                                <div class="stat-label">Faux billets ({fake_percent:.1f}%)</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with col4:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-icon">🔍</div>
                                <div class="stat-value" style="color:var(--secondary);">{fake_percent:.1f}%</div>
                                <div class="stat-label">Taux de fraude</div>
                            </div>
                            """, unsafe_allow_html=True)
                       
                        # Graphique amélioré
                        st.markdown("<h4 style='text-align: center; margin-top: 2rem;'>Répartition des résultats</h4>", unsafe_allow_html=True)
                        
                        fig_col1, fig_col2 = st.columns([2, 1])
                        
                        with fig_col1:
                            fig = px.pie(
                                names=['Authentiques', 'Faux'],
                                values=[genuine_count, fake_count],
                                color=['Authentiques', 'Faux'],
                                color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
                                hole=0.5
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            fig.update_layout(
                                showlegend=False, 
                                margin=dict(l=20, r=20, t=30, b=20),
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with fig_col2:
                            avg_confidence_genuine = np.mean([p['probability'] for p in predictions if p['prediction'] == 'Genuine']) * 100
                            avg_confidence_fake = np.mean([1-p['probability'] for p in predictions if p['prediction'] == 'Fake']) * 100
                            
                            st.markdown("""
                            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); height: 100%;">
                                <h5 style="margin-top: 0; color: var(--secondary);">Confiance moyenne</h5>
                                <div style="margin-bottom: 1rem;">
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                        <span>Authentiques:</span>
                                        <span><strong>{:.1f}%</strong></span>
                                    </div>
                                    <div style="height: 8px; background: #e0e0e0; border-radius: 4px;">
                                        <div style="height: 100%; width: {}%; background: var(--success); border-radius: 4px;"></div>
                                    </div>
                                </div>
                                <div>
                                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                                        <span>Faux:</span>
                                        <span><strong>{:.1f}%</strong></span>
                                    </div>
                                    <div style="height: 8px; background: #e0e0e0; border-radius: 4px;">
                                        <div style="height: 100%; width: {}%; background: var(--danger); border-radius: 4px;"></div>
                                    </div>
                                </div>
                            </div>
                            """.format(
                                avg_confidence_genuine if genuine_count > 0 else 0,
                                avg_confidence_genuine if genuine_count > 0 else 0,
                                avg_confidence_fake if fake_count > 0 else 0,
                                avg_confidence_fake if fake_count > 0 else 0
                            ), unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        st.markdown("""
        <div class="card" style="border-left: 4px solid var(--danger);">
            <h5>Format de fichier incorrect</h5>
            <p>Assurez-vous que votre fichier :</p>
            <ul>
                <li>Est un CSV valide</li>
                <li>Utilise le point-virgule (;) comme séparateur</li>
                <li>Contient les colonnes requises</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
