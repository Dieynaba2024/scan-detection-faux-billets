# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version Clean Pastel UI
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

# --- CSS Modern Pastel Clean ---
st.markdown("""
<style>
body {
    background: #f9fafc;
    color: #333;
}
.header {
    background: linear-gradient(135deg, #a8edea, #fed6e3);
    color: #333;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 1.5rem;
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
}
.genuine-card {
    border-left: 6px solid #6bcfbc;
}
.fake-card {
    border-left: 6px solid #f78c6b;
}
.stat-card {
    text-align: center;
    padding: 0.8rem;
    background: #ffffff;
    border-radius: 10px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
.stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: #4a4a4a;
}
.progress-bar {
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px;
    margin-top: 0.3rem;
}
.progress-fill {
    height: 100%;
    border-radius: 4px;
}
.show-more-btn {
    margin-top: 0.5rem;
    background: #f0f0f0 !important;
    color: #555 !important;
    border: 1px solid #ccc !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="header">💵 Détection de Faux Billets CFA - Interface Soft</div>', unsafe_allow_html=True)

# --- Upload ---
uploaded_file = st.file_uploader("📄 Importer un fichier CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')

        # --- Aperçu des données ---
        st.markdown("### 📊 Aperçu des données")
        preview_rows = 5
        table_placeholder = st.empty()
        table_placeholder.dataframe(df.head(preview_rows), height=210, use_container_width=True)

        if len(df) > preview_rows:
            if st.button("Afficher plus", key="show_more_btn", type="secondary"):
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

                        # Résultats des Prédictions
                        st.markdown("### 📝 Résultats de la détection")
                        genuine_img = image_to_base64(GENUINE_BILL_IMAGE)
                        fake_img = image_to_base64(FAKE_BILL_IMAGE)

                        cols_per_row = 3
                        for i in range(0, len(predictions), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < len(predictions):
                                    pred = predictions[i + j]
                                    prob_percent = pred['probability'] * 100 if pred['prediction'] == 'Genuine' else (1 - pred['probability']) * 100
                                    color_fill = "#6bcfbc" if pred['prediction'] == 'Genuine' else "#f78c6b"
                                    img_b64 = genuine_img if pred['prediction'] == 'Genuine' else fake_img
                                    card_class = "genuine-card" if pred['prediction'] == 'Genuine' else "fake-card"

                                    with cols[j]:
                                        st.markdown(f"""
                                        <div class="card {card_class}">
                                            <div style="display:flex; align-items:center;">
                                                <div style="flex:1;">
                                                    <h5 style="margin:0 0 0.3rem 0;">Billet n°{pred['id']} - {"Authentique" if pred['prediction']=="Genuine" else "Faux"}</h5>
                                                    <p style="margin:0 0 0.3rem 0;">Prédiction : <strong>{prob_percent:.1f}%</strong></p>
                                                    <div class="progress-bar">
                                                        <div class="progress-fill" style="width:{prob_percent}%; background:{color_fill};"></div>
                                                    </div>
                                                </div>
                                                <div style="margin-left:1rem;">
                                                    <img src="data:image/png;base64,{img_b64}" width="80" style="border-radius:8px;">
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                        # --- Statistiques ---
                        genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                        fake_count = len(predictions) - genuine_count

                        st.markdown("### 📊 Statistiques globales")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{len(predictions)}</div>
                                <div>Billets analysés</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color:#6bcfbc;">{genuine_count}</div>
                                <div>Authentiques</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="color:#f78c6b;">{fake_count}</div>
                                <div>Faux billets</div>
                            </div>
                            """, unsafe_allow_html=True)

                        # --- Graphique Soft ---
                        st.markdown("### 📊 Graphique de répartition")
                        fig = px.pie(
                            names=['Authentiques', 'Faux'],
                            values=[genuine_count, fake_count],
                            color=['Authentiques', 'Faux'],
                            color_discrete_map={'Authentiques': '#6bcfbc', 'Faux': '#f78c6b'},
                            hole=0.5
                        )
                        fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")

    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
else:
    st.info("Veuillez importer un fichier CSV pour lancer l'analyse.")
