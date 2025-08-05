# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version optimisée (même design)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuration de la page (identique)
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Chargement optimisé des ressources
@st.cache_resource(ttl=3600)  # Cache 1 heure
def load_resources():
    # Modèle et scaler
    model = joblib.load("random_forest_model.sav")
    scaler = joblib.load("scaler.sav")
    
    # Images en base64 (cache permanent)
    def load_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    return {
        'model': model,
        'scaler': scaler,
        'images': {
            'genuine': load_image("vraibillet.PNG"),
            'fake': load_image("fauxbillet.png")
        }
    }

resources = load_resources()

# 2. CSS identique
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
    <h2 style="color:white; margin:0;">💰 Application Scan Franc cfa thille </h2>
    <p style="color:white; opacity:0.9; margin:0;">Solution de détection de faux billets</p>
</div>
""", unsafe_allow_html=True)

# 3. Traitement optimisé du fichier
@st.cache_data(ttl=300)  # Cache 5 minutes
def process_file(uploaded_file):
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
    df, error = process_file(uploaded_file)
    
    if error:
        st.error(f"Erreur de lecture du fichier: {error}")
    else:
        # Affichage tableau identique
        st.markdown("#### Aperçu des données")
        preview_rows = 5
        table_placeholder = st.empty()
        table_placeholder.dataframe(df.head(preview_rows), height=210, use_container_width=True)
        
        if len(df) > preview_rows:
            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
                table_placeholder.dataframe(df, height=min(800, len(df)*35), use_container_width=True)
        
        if st.button("Lancer la détection", key="analyze_btn"):
            with st.spinner("Analyse en cours..."):
                try:
                    # 4. Prédiction vectorisée optimisée
                    required_cols = ['diagonal', 'height_left', 'height_right', 
                                   'margin_low', 'margin_up', 'length']
                    features = df[required_cols].values
                    features_scaled = resources['scaler'].transform(features)
                    probas = resources['model'].predict_proba(features_scaled)
                    
                    # Calcul des stats (optimisé)
                    genuine_mask = probas[:, 1] > 0.5
                    genuine_count = np.sum(genuine_mask)
                    fake_count = len(probas) - genuine_count
                    
                    st.success("Analyse terminée avec succès !")
                    
                    # 5. Affichage optimisé (même design)
                    st.markdown("#### Résultats de la détection")
                    
                    # Génération du HTML en une fois
                    results_html = []
                    cols_per_row = 3
                    for i in range(0, min(50, len(probas)), cols_per_row):  # Limité à 50
                        row_html = []
                        for j in range(cols_per_row):
                            if i + j < min(50, len(probas)):
                                idx = i + j
                                pred = "Genuine" if genuine_mask[idx] else "Fake"
                                prob = probas[idx, 1] if pred == "Genuine" else 1 - probas[idx, 1]
                                img = resources['images']['genuine'] if pred == "Genuine" else resources['images']['fake']
                                
                                row_html.append(f"""
                                <div class="card {'genuine-card' if pred == 'Genuine' else 'fake-card'}">
                                    <div style="display:flex; align-items:center;">
                                        <div style="flex:1;">
                                            <h5 style="color:{'var(--success)' if pred == 'Genuine' else 'var(--danger)'}; margin:0 0 0.3rem 0;">
                                                Billet n°{idx+1} - {'Authentique' if pred == 'Genuine' else 'Faux'}
                                            </h5>
                                            <p style="margin:0 0 0.2rem 0;">Probabilité: <strong>{prob*100:.1f}%</strong></p>
                                            <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                <div style="height:100%; width:{prob*100}%; background:{'var(--success)' if pred == 'Genuine' else 'var(--danger)'}; border-radius:3px;"></div>
                                            </div>
                                        </div>
                                        <div style="margin-left:1rem;">
                                            <img src="data:image/png;base64,{img}" width="80" style="border-radius:6px;">
                                        </div>
                                    </div>
                                </div>
                                """)
                        
                        # Ajout des colonnes
                        results_html.append(f"""
                        <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                            {''.join(f'<div style="flex: 1;">{html}</div>' for html in row_html)}
                        </div>
                        """)
                    
                    st.markdown(''.join(results_html), unsafe_allow_html=True)
                    
                    # Statistiques (identique)
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
                    
                    # Graphique (optimisé)
                    st.markdown("<h4 style='text-align: center;'>Graphique des statistiques</h4>", unsafe_allow_html=True)
                    fig = px.pie(
                        names=['Authentiques', 'Faux'],
                        values=[genuine_count, fake_count],
                        color=['Authentiques', 'Faux'],
                        color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
                        hole=0.4,
                        height=400  # Taille fixe pour meilleures perfs
                    )
                    fig.update_layout(
                        showlegend=True,
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
