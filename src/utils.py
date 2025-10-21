import streamlit as st
import plotly.graph_objects as go

def show_probabilities(home_prob, draw_prob, away_prob):
    st.subheader("Predicted Outcome Probabilities")
    st.metric("🏠 Home Win", f"{home_prob*100:.1f}%")
    st.metric("🤝 Draw", f"{draw_prob*100:.1f}%")
    st.metric("🚀 Away Win", f"{away_prob*100:.1f}%")

    fig = go.Figure(data=[go.Bar(x=["Home", "Draw", "Away"],
                                 y=[home_prob, draw_prob, away_prob],
                                 marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])])
    st.plotly_chart(fig, use_container_width=True)
