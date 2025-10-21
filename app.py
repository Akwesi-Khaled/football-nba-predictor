import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from src.data_loader import load_football_data, load_nba_data
from src.model import train_poisson_model, train_nba_model
from src.predictor import predict_football, predict_nba
from src.utils import show_probabilities

st.set_page_config(page_title="Sports Predictor", page_icon="⚽🏀", layout="centered")

st.title("🏆 Sports Outcome Predictor")
st.markdown("#### Predict results for top **Football** and **NBA** matchups.")

league = st.sidebar.selectbox("Select League", ["Football", "NBA"])

if league == "Football":
    df = load_football_data()
    teams = sorted(df['HomeTeam'].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams)
    with col2:
        away_team = st.selectbox("Away Team", teams)

    if home_team != away_team and st.button("Predict Outcome"):
        home_model, away_model = train_poisson_model(df)
        home_prob, draw_prob, away_prob = predict_football(home_model, away_model, home_team, away_team)
        show_probabilities(home_prob, draw_prob, away_prob)

elif league == "NBA":
    df = load_nba_data()
    model = train_nba_model(df)

    st.write("Enter recent team performance points (approx.):")
    home_points = st.slider("Home Avg Points", 80, 140, 110)
    away_points = st.slider("Away Avg Points", 80, 140, 105)

    if st.button("Predict NBA Result"):
        home_win_prob = predict_nba(model, home_points, away_points)
        show_probabilities(home_win_prob, 0, 1 - home_win_prob)
