# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Visual Cards Design
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import os
import joblib
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

genuine_img_b64 = image_to_base64(GENUINE_BILL_IMAGE)
fake_img_b64 = image_to_base64(FAKE_BILL_IMAGE)

# --- CSS Modern & Classy ---
st.markdown("""
<style>
body {
    background: #0f0f0f;
    color: #e0e0e0;
}
.header {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.5rem;
    font-weight: bold;
}
.badge {
    display: inline-block;
    padding: 0.8rem 1.5rem;
    border-radius: 20px;
    background: #1c1c1c;
    color: #cfcfcf;
    font-size: 1.2rem;
    margin: 0.5rem;
    border: 1px solid #333;
}
.badge.auth {
    border-left: 6px solid #4cb8c4;
}
.badge.fake {
    border-left: 6px solid #f7797d;
}
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
}
.card {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 0 10px rgba(0,0,0,0.5);
    transition: transform 0.3s ease;
}
.card:hover {
    transform: scale(1.02);
}
.progress-bar {
    height: 10px;
    background: #333;
    border-radius: 5px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.progress-fill {
    height: 100%;
    transition: width 0.5s ease;
    border-radius: 5px;
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

        # --- Layout: Data Overview ---
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
                                <span class="badge">🔢 Total : {total}</span>
                                <span class="badge auth">✅ Authentiques : {genuine_count}</span>
                                <span class="badge fake">❌ Faux : {fake_count}</span>
                            </div>
                            """, unsafe_allow_html=True)

                            # --- Graphique ---
                            st.markdown("### 📈 Répartition des Billets")
                            fig = px.bar(
                                x=[genuine_count, fake_count],
                                y=['Authentiques', 'Faux'],
                                orientation='h',
                                color=['Authentiques', 'Faux'],
                                color_discrete_map={'Authentiques': '#4cb8c4', 'Faux': '#f7797d'},
                                text_auto=True,
                                height=300
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color="#e0e0e0"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # --- Visual Cards Résultats ---
                            st.markdown("### 🖼️ Résultats Visuels des Billets")
                            st.markdown('<div class="card-grid">', unsafe_allow_html=True)

                            for idx, row in df_results.iterrows():
                                img_b64 = genuine_img_b64 if row['prediction'] == "Authentique" else fake_img_b64
                                color_fill = "#4cb8c4" if row['prediction'] == "Authentique" else "#f7797d"
                                percent = int(row['proba_authentic']*100) if row['prediction'] == "Authentique" else int((1-row['proba_authentic'])*100)
                                description = "Billet Authentique" if row['prediction'] == "Authentique" else "Billet Faux"

                                st.markdown(f"""
                                <div class="card">
                                    <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:8px; margin-bottom:0.5rem;">
                                    <h5 style="margin:0; color:{color_fill};">{description}</h5>
                                    <p style="margin:0.2rem 0;">Prédiction : {percent}%</p>
                                    <div class="progress-bar">
                                        <div class="progress-fill" style="width:{percent}%; background:{color_fill};"></div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
else:
    st.info("Veuillez importer un fichier CSV pour lancer l'analyse.")
