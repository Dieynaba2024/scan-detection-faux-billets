# -*- coding: utf-8 -*-
"""
Application optimisée de détection de faux billets - Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import base64
import joblib
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Détection de Faux Billets",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des ressources avec cache
@st.cache_resource
def load_resources():
    model = joblib.load("random_forest_model.sav")
    scaler = joblib.load("scaler.sav")
    def img_to_base64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    return {
        'model': model,
        'scaler': scaler,
        'images': {
            'genuine': img_to_base64("vraibillet.PNG"),
            'fake': img_to_base64("fauxbillet.png")
        }
    }

resources = load_resources()

# CSS Design
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h2>💰 Application Scan Franc CFA</h2>
    <p>Solution de détection de faux billets</p>
</div>
""", unsafe_allow_html=True)

# Upload fichier
uploaded_file = st.file_uploader("Déposez votre fichier CSV ici", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        st.markdown("### Aperçu des données")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Lancer la détection"):
            with st.spinner("Analyse en cours..."):
                required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                features = df[required_cols]
                features_scaled = resources['scaler'].transform(features)
                probas = resources['model'].predict_proba(features_scaled)

                predictions = [{
                    'id': i+1,
                    'prediction': "Genuine" if p[1] > 0.5 else "Fake",
                    'probability': p[1]
                } for i, p in enumerate(probas)]

                st.success("Analyse terminée avec succès !")

                # Affichage des résultats
                st.markdown("### Résultats de la détection")
                for i in range(0, len(predictions), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(predictions):
                            pred = predictions[i + j]
                            card_color = 'genuine-card' if pred['prediction'] == 'Genuine' else 'fake-card'
                            prob_display = pred['probability']*100 if pred['prediction'] == 'Genuine' else (1-pred['probability'])*100
                            img_src = resources['images']['genuine'] if pred['prediction'] == 'Genuine' else resources['images']['fake']

                            with cols[j]:
                                st.markdown(f"""
                                <div class="card {card_color}">
                                    <div style="display:flex; align-items:center;">
                                        <div style="flex:1;">
                                            <h5>Billet n°{pred['id']} - {'Authentique' if pred['prediction']=='Genuine' else 'Faux'}</h5>
                                            <p>Probabilité: <strong>{prob_display:.2f}%</strong></p>
                                            <div style="height:6px; background:#e9ecef; border-radius:3px;">
                                                <div style="height:100%; width:{prob_display}%; background: {'var(--success)' if pred['prediction'] == 'Genuine' else 'var(--danger)'}; border-radius:3px;"></div>
                                            </div>
                                        </div>
                                        <div style="margin-left:1rem;">
                                            <img src="data:image/png;base64,{img_src}" width="70" style="border-radius:6px;">
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                # Statistiques Globales
                genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                fake_count = len(predictions) - genuine_count

                st.markdown("---")
                st.markdown("### Statistiques Globales")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Billets analysés", len(predictions))
                with col2:
                    st.metric("Authentiques", genuine_count, delta=f"{(genuine_count/len(predictions))*100:.1f}%")
                with col3:
                    st.metric("Faux", fake_count, delta=f"{(fake_count/len(predictions))*100:.1f}%")

                # Graphique camembert
                fig = px.pie(
                    names=['Authentiques', 'Faux'],
                    values=[genuine_count, fake_count],
                    color=['Authentiques', 'Faux'],
                    color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur : {str(e)}")
else:
    st.info("Veuillez importer un fichier CSV au format attendu (séparateur ;)")
