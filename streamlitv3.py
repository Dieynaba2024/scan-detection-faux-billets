# -*- coding: utf-8 -*-
"""
Application de détection de faux billets - Version premium
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Scan Franc CFA - Détection de Faux Billets",
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

# Chemins des images
GENUINE_BILL_IMAGE = "vraibillet.PNG"
FAKE_BILL_IMAGE = "fauxbillet.png"

# CSS personnalisé
st.markdown("""
<style>
    :root {
        --primary: #4a6fa5;
        --secondary: #166088;
        --accent: #4fc3f7;
        --success: #28a745;
        --danger: #dc3545;
        --warning: #ffc107;
        --light: #f8f9fa;
        --dark: #343a40;
    }
    
    .header-container {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        transform: rotate(30deg);
    }
    
    .card {
        background: white;
        border-radius: 12px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: none;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.1);
    }
    
    .card-genuine {
        border-left: 5px solid var(--success);
        background: linear-gradient(to right, rgba(40, 167, 69, 0.05), white);
    }
    
    .card-fake {
        border-left: 5px solid var(--danger);
        background: linear-gradient(to right, rgba(220, 53, 69, 0.05), white);
    }
    
    .stat-card {
        text-align: center;
        padding: 1.5rem 1rem;
        border-radius: 12px;
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }
    
    .stat-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--dark);
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .progress-container {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        margin: 0.8rem 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
    }
    
    .btn-analyze {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white !important;
        border: none !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-analyze:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .file-uploader {
        border: 2px dashed #ced4da !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        background: rgba(248, 249, 250, 0.5) !important;
    }
    
    .feature-importance-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <div style="position: relative; z-index: 1;">
        <h1 style="color:white; margin:0; font-size:2.2rem;">💵 Scan Franc CFA t</h1>
        <p style="color:white; opacity:0.9; margin:0; font-size:1.1rem;">Solution avancée de détection de faux billets</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Fonction pour convertir image en base64
def image_to_base64(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return ""

# Section Analyse
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader(
        "📤 Téléversez votre fichier CSV contenant les données des billets",
        type=["csv"],
        key="file_uploader",
        help="Le fichier doit contenir les colonnes: diagonal, height_left, height_right, margin_low, margin_up, length"
    )

with col2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Lancer l'analyse", key="analyze_btn", disabled=uploaded_file is None, 
                           help="Cliquez pour analyser les billets après avoir téléversé un fichier")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        
        # Aperçu des données
        with st.expander("📊 Aperçu des données", expanded=True):
            st.dataframe(df.style
                        .background_gradient(cmap='Blues', subset=df.select_dtypes(include='number').columns))
        
        if analyze_btn:
            with st.spinner("🔍 Analyse en cours... Veuillez patienter"):
                if model is None:
                    st.error("Modèle non chargé - Impossible d'effectuer la prédiction")
                else:
                    try:
                        # Exemple de prétraitement
                        required_cols = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                        if not all(col in df.columns for col in required_cols):
                            raise ValueError("Colonnes requises manquantes dans le fichier CSV")
                            
                        features = df[required_cols]
                        features_scaled = scaler.transform(features)
                        probas = model.predict_proba(features_scaled)
                        
                        predictions = [{
                            'id': i+1,
                            'prediction': "Genuine" if p[1] > 0.5 else "Fake",
                            'probability': p[1] if p[1] > 0.5 else 1-p[1],
                            'confidence': "high" if (p[1] > 0.8 or p[1] < 0.2) else "medium" if (p[1] > 0.7 or p[1] < 0.3) else "low"
                        } for i, p in enumerate(probas)]
                        
                        st.success("✅ Analyse terminée avec succès !")
                        
                        # Affichage des résultats
                        st.markdown("## 📝 Résultats de la détection")
                        genuine_img = image_to_base64(GENUINE_BILL_IMAGE)
                        fake_img = image_to_base64(FAKE_BILL_IMAGE)
                        
                        # Affichage par cartes
                        cols_per_row = 3
                        for i in range(0, len(predictions), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j in range(cols_per_row):
                                if i + j < len(predictions):
                                    pred = predictions[i + j]
                                    with cols[j]:
                                        if pred['prediction'] == 'Genuine':
                                            confidence_color = "#28a745" if pred['confidence'] == "high" else "#5cb85c" if pred['confidence'] == "medium" else "#7bbf7b"
                                            st.markdown(f"""
                                            <div class="card card-genuine">
                                                <div style="display:flex; align-items:center; gap:1rem;">
                                                    <div style="flex:1;">
                                                        <h3 style="margin:0 0 0.5rem 0; color:var(--success);">Billet #{pred['id']}</h3>
                                                        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                                                            <div style="background:{confidence_color}; width:12px; height:12px; border-radius:50%;"></div>
                                                            <span style="font-weight:600; color:{confidence_color};">Authentique ({pred['confidence']})</span>
                                                        </div>
                                                        <p style="margin:0 0 0.3rem 0; font-size:0.9rem;">Confiance: <strong>{pred['probability']*100:.1f}%</strong></p>
                                                        <div class="progress-container">
                                                            <div class="progress-bar" style="width:{pred['probability']*100}%; background:{confidence_color};"></div>
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <img src="data:image/png;base64,{genuine_img}" width="80" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            confidence_color = "#dc3545" if pred['confidence'] == "high" else "#d9534f" if pred['confidence'] == "medium" else "#df7e7b"
                                            st.markdown(f"""
                                            <div class="card card-fake">
                                                <div style="display:flex; align-items:center; gap:1rem;">
                                                    <div style="flex:1;">
                                                        <h3 style="margin:0 0 0.5rem 0; color:var(--danger);">Billet #{pred['id']}</h3>
                                                        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                                                            <div style="background:{confidence_color}; width:12px; height:12px; border-radius:50%;"></div>
                                                            <span style="font-weight:600; color:{confidence_color};">Faux ({pred['confidence']})</span>
                                                        </div>
                                                        <p style="margin:0 0 0.3rem 0; font-size:0.9rem;">Confiance: <strong>{pred['probability']*100:.1f}%</strong></p>
                                                        <div class="progress-container">
                                                            <div class="progress-bar" style="width:{pred['probability']*100}%; background:{confidence_color};"></div>
                                                        </div>
                                                    </div>
                                                    <div>
                                                        <img src="data:image/png;base64,{fake_img}" width="80" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                                                    </div>
                                                </div>
                                            </div>
                                            """, unsafe_allow_html=True)
                        
                        # Statistiques
                        genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
                        fake_count = len(predictions) - genuine_count
                        
                        st.markdown("## 📊 Statistiques globales")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Total analysé</div>
                                <div class="stat-value">{len(predictions)}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Authentiques</div>
                                <div class="stat-value" style="color:var(--success);">{genuine_count}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Faux billets</div>
                                <div class="stat-value" style="color:var(--danger);">{fake_count}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-label">Taux de faux</div>
                                <div class="stat-value" style="color:var(--warning);">{fake_count/len(predictions)*100:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Graphiques avancés
                        tab1, tab2 = st.tabs(["📈 Répartition", "📊 Analyse des caractéristiques"])
                        
                        with tab1:
                            fig = go.Figure()
                            
                            # Camembert avec effet 3D
                            fig.add_trace(go.Pie(
                                labels=['Authentiques', 'Faux'],
                                values=[genuine_count, fake_count],
                                hole=0.5,
                                marker_colors=['#28a745', '#dc3545'],
                                textinfo='percent+value',
                                textposition='inside',
                                insidetextorientation='radial',
                                hoverinfo='label+percent',
                                pull=[0.1, 0],
                                rotation=45
                            ))
                            
                            fig.update_layout(
                                title="Répartition des billets analysés",
                                showlegend=True,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.2,
                                    xanchor="center",
                                    x=0.5
                                ),
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            # Graphique d'importance des caractéristiques (exemple)
                            features_importance = {
                                'Caractéristique': required_cols,
                                'Importance': [0.25, 0.18, 0.15, 0.22, 0.12, 0.08]  # Valeurs d'exemple
                            }
                            
                            fig = px.bar(
                                features_importance,
                                x='Importance',
                                y='Caractéristique',
                                orientation='h',
                                color='Importance',
                                color_continuous_scale='Blues',
                                title="Importance des caractéristiques dans la détection"
                            )
                            
                            fig.update_layout(
                                yaxis=dict(autorange="reversed"),
                                coloraxis_showscale=False,
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur lors de la prédiction : {str(e)}")
    except Exception as e:
        st.error(f"Erreur de lecture du fichier: {str(e)}")
        st.markdown("""
        <div class="card" style="border-left: 4px solid var(--danger);">
            <h4>❌ Format de fichier incorrect</h4>
            <p>Assurez-vous que votre fichier :</p>
            <ul>
                <li>Est un CSV valide avec séparateur point-virgule (;)</li>
                <li>Contient les colonnes requises</li>
                <li>N'a pas de lignes vides ou de données manquantes</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Section d'information quand aucun fichier n'est chargé
if uploaded_file is None:
    st.markdown("""
    <div class="card" style="text-align:center; padding:2rem;">
        <h3 style="color:var(--primary);">Comment utiliser cette application</h3>
        <ol style="text-align:left; margin:1rem auto; max-width:600px;">
            <li>Téléversez un fichier CSV contenant les caractéristiques des billets</li>
            <li>Vérifiez que les données s'affichent correctement</li>
            <li>Cliquez sur le bouton "Lancer l'analyse"</li>
            <li>Consultez les résultats et statistiques</li>
        </ol>
        <p style="font-style:italic; margin-top:1rem;">Pour des résultats optimaux, assurez-vous que votre fichier suit le format requis.</p>
    </div>
    """, unsafe_allow_html=True)


