# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version optimisée
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

# 1. Configuration avec cache optimisé
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Chargement du modèle avec cache persistant
@st.cache_resource(ttl=3600)  # Cache 1 heure
def load_assets():
    # Modèle
    model = joblib.load("random_forest_model.sav")
    scaler = joblib.load("scaler.sav")
    
    # Images
    def load_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    genuine_img = load_image("vraibillet.PNG")
    fake_img = load_image("fauxbillet.png")
    
    return model, scaler, genuine_img, fake_img

model, scaler, genuine_img, fake_img = load_assets()

# 3. CSS identique mais chargé une seule fois
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
        padding: 0.8rem;
    }
    
    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--secondary);
    }
</style>
""", unsafe_allow_html=True)

# Header identique
st.markdown("""
<div class="header">
    <h2 style="color:white; margin:0;">💰 Application Scan Franc cfa </h2>
    <p style="color:white; opacity:0.9; margin:0;">Solution de détection de faux billets</p>
</div>
""", unsafe_allow_html=True)

# 4. Traitement du fichier avec cache
@st.cache_data(ttl=300, max_entries=3)  # Cache 5 min, 3 fichiers max
def process_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        return df, None
    except Exception as e:
        return None, str(e)

uploaded_file = st.file_uploader(
    "Faites glisser et déposez le fichier ici ou cliquez sur le bouton pour Parcourir", 
    type=["csv"],
    key="file_uploader"
)

if uploaded_file is not None:
    df, error = process_data(uploaded_file)
    
    if error:
        st.error(f"Erreur de lecture: {error}")
    else:
        # 5. Affichage tableau identique mais optimisé
        st.markdown("#### Aperçu des données")
        preview_rows = 5
        table_placeholder = st.empty()
        table_placeholder.dataframe(df.head(preview_rows), height=210, use_container_width=True)
        
        if len(df) > preview_rows:
            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
                table_placeholder.dataframe(df, height=min(800, len(df)*35), use_container_width=True)
        
        if st.button("Lancer la détection", key="analyze_btn"):
            with st.spinner("Analyse en cours..."):
                if model is None:
                    st.error("Modèle non chargé")
                else:
                    try:
                        # 6. Prédiction vectorisée
                        required_cols = ['diagonal', 'height_left', 'height_right', 
                                       'margin_low', 'margin_up', 'length']
                        features = df[required_cols].values
                        features_scaled = scaler.transform(features)
                        probas = model.predict_proba(features_scaled)
                        
                        # 7. Calcul optimisé des stats
                        genuine_mask = probas[:, 1] > 0.5
                        genuine_count = np.sum(genuine_mask)
                        fake_count = len(probas) - genuine_count
                        
                        st.success("Analyse terminée !")
                        
                        # 8. Affichage résultats (identique mais optimisé)
                        st.markdown("#### Résultats de la détection")
                        
                        # Limite à 50 résultats pour la performance
                        max_display = min(50, len(probas))
                        display_indices = np.random.choice(len(probas), max_display, replace=False)
                        
                        cols_per_row = 3
                        for i in range(0, max_display, cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < max_display:
                                    idx = display_indices[i + j]
                                    pred = {
                                        'id': idx,
                                        'prediction': "Genuine" if genuine_mask[idx] else "Fake",
                                        'probability': probas[idx, 1]
                                    }
                                    with cols[j]:
                                        if pred['prediction'] == 'Genuine':
                                            st.markdown(f"""
                                            <div class="card genuine-card">
                                                <div style="display:flex; align-items:center;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--success); margin:0 0 0.3rem 0;">Billet n°{pred['id']+1} - Authentique</h5>
                                                        <p style="margin:0 0 0.2rem 0;">Probabilité: <strong>{pred['probability']*100:.1f}%</strong></p>
                                                        <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                            <div style="height:100%; width:{pred['probability']*100}%; background:var(--success); border-radius:3px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1rem;">
                                                        <img src="data:image/png;base64,{genuine_img}" width="80" style="border-radius:6px;">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"""
                                            <div class="card fake-card">
                                                <div style="display:flex; align-items:center;">
                                                    <div style="flex:1;">
                                                        <h5 style="color:var(--danger); margin:0 0 0.3rem 0;">Billet n°{pred['id']+1} - Faux</h5>
                                                        <p style="margin:0 0 0.2rem 0;">Probabilité: <strong>{(1-pred['probability'])*100:.1f}%</strong></p>
                                                        <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                            <div style="height:100%; width:{(1-pred['probability'])*100}%; background:var(--danger); border-radius:3px;"></div>
                                                        </div>
                                                    </div>
                                                    <div style="margin-left:1rem;">
                                                        <img src="data:image/png;base64,{fake_img}" width="80" style="border-radius:6px;">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                        
                        # 9. Statistiques (identique)
                        st.markdown("<h4 style='text-align: center;'>Statistiques de détection</h4>", unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="card stat-card">
                                <div class="stat-value">{len(probas)}</div>
                                <div class="stat-label">Billets analysés</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="card stat-card">
                                <div class="stat-value" style="color:var(--success);">{genuine_count}</div>
                                <div class="stat-label">Authentiques</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="card stat-card">
                                <div class="stat-value" style="color:var(--danger);">{fake_count}</div>
                                <div class="stat-label">Faux billets</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # 10. Graphique optimisé
                        st.markdown("<h4 style='text-align: center;'>Graphique des statistiques</h4>", unsafe_allow_html=True)
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
                            height=400  # Taille fixe pour meilleures perfs
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Erreur: {str(e)}")
