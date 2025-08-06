# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version Modern UI Dashboard
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

GENUINE_BILL_IMAGE = "vraibillet.PNG"
FAKE_BILL_IMAGE = "fauxbillet.png"

# Fonction pour convertir image en base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# --- CSS Design Neon Modern ---
st.markdown("""
<style>
body {
    background: #0f2027;
    background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
    color: #f0f0f0;
}
.header {
    background: linear-gradient(135deg, #12c2e9, #c471ed, #f64f59);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.5rem;
    font-weight: bold;
}
.stat-badge {
    display: inline-block;
    padding: 1rem 2rem;
    border-radius: 30px;
    background: linear-gradient(135deg, #00f2fe, #4facfe);
    color: white;
    font-size: 1.3rem;
    margin: 0.5rem;
    animation: pulse 2s infinite;
}
.stat-badge.fake {
    background: linear-gradient(135deg, #f85032, #e73827);
}
.stat-badge.total {
    background: linear-gradient(135deg, #11998e, #38ef7d);
}
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}
.dataframe th {
    background-color: #0d1117 !important;
    color: #58a6ff !important;
}
.dataframe td {
    background-color: #161b22 !important;
    color: #f0f6fc !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">💵 Détection de Faux Billets CFA - Dashboard</div>', unsafe_allow_html=True)

# --- Upload ---
uploaded_file = st.file_uploader("📄 Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')

        # --- Layout Start ---
        col1, col2 = st.columns([1.5, 2])

        with col1:
            st.markdown("### 📊 Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True, height=300)

        with col2:
            st.markdown("### ⚙️ Analyse des billets")

            if st.button("Lancer la détection"):
                if model is None:
                    st.error("Modèle non chargé - Impossible d'effectuer la prédiction")
                else:
                    required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                    if not all(col in df.columns for col in required_cols):
                        st.error("Colonnes requises manquantes dans le fichier CSV")
                    else:
                        with st.spinner("Analyse en cours..."):
                            features = df[required_cols]
                            features_scaled = scaler.transform(features)
                            probas = model.predict_proba(features_scaled)

                            predictions = [{
                                'id': i,
                                'prediction': "Authentique" if p[1] > 0.5 else "Faux",
                                'proba_authentic': p[1]
                            } for i, p in enumerate(probas)]

                            df_results = pd.DataFrame(predictions)
                            genuine_count = (df_results['prediction'] == "Authentique").sum()
                            fake_count = (df_results['prediction'] == "Faux").sum()
                            total = len(df_results)

                            # --- Stats Badges ---
                            st.markdown(f"""
                            <div style="text-align:center;">
                                <span class="stat-badge total">🔢 Total : {total}</span>
                                <span class="stat-badge">✅ Authentiques : {genuine_count}</span>
                                <span class="stat-badge fake">❌ Faux : {fake_count}</span>
                            </div>
                            """, unsafe_allow_html=True)

                            # --- Graphique (Barres Horizontales) ---
                            st.markdown("### 📈 Statistiques Visuelles")
                            fig = px.bar(
                                x=[genuine_count, fake_count],
                                y=['Authentiques', 'Faux'],
                                orientation='h',
                                color=['Authentiques', 'Faux'],
                                color_discrete_map={'Authentiques': '#00f2fe', 'Faux': '#f85032'},
                                text_auto=True,
                                height=300
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color="#f0f0f0"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # --- Résultats détaillés ---
                            st.markdown("### 📝 Résultats de la détection")
                            st.dataframe(df_results, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
else:
    st.info("Veuillez importer un fichier CSV pour lancer l'analyse.")

