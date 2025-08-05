# Affichage des résultats - Vue Tableau + Cartes Toggle
view_mode = st.radio("Mode d'affichage des résultats :", ["Cartes", "Tableau"], horizontal=True)

if view_mode == "Tableau":
    results_df = pd.DataFrame([
        {
            "ID": p['id'],
            "Prédiction": p['prediction'],
            "Probabilité (Authentique %)": round(p['probability']*100, 1) if p['prediction'] == "Genuine" else round((1-p['probability'])*100, 1)
        }
        for p in predictions
    ])
    st.dataframe(results_df, use_container_width=True, height=min(500, len(predictions)*35 + 50))
else:
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
                            <div style="display:flex; align-items:center;">
                                <div style="flex:1;">
                                    <h5 style="color:var(--success); margin:0 0 0.3rem 0;">Billet n°{pred['id']} - Authentique</h5>
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
                                    <h5 style="color:var(--danger); margin:0 0 0.3rem 0;">Billet n°{pred['id']} - Faux</h5>
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

# Statistiques (Chiffres + Graphiques)
genuine_count = sum(1 for p in predictions if p['prediction'] == 'Genuine')
fake_count = len(predictions) - genuine_count

with st.expander("📊 Voir les statistiques détaillées"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="card stat-card">
            <div class="stat-value">{len(predictions)}</div>
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

    # Graphique en colonnes
    stats_df = pd.DataFrame({
        'Type': ['Authentiques', 'Faux'],
        'Nombre': [genuine_count, fake_count]
    })

    bar_chart = px.bar(
        stats_df,
        x='Type',
        y='Nombre',
        color='Type',
        color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
        text='Nombre'
    )
    bar_chart.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(bar_chart, use_container_width=True)

    # Graphique circulaire (Pie chart)
    fig = px.pie(
        names=['Authentiques', 'Faux'],
        values=[genuine_count, fake_count],
        color=['Authentiques', 'Faux'],
        color_discrete_map={'Authentiques': '#4CAF50', 'Faux': '#F44336'},
        hole=0.4
    )
    fig.update_layout(showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)
