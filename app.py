
import streamlit as st
import pandas as pd
import joblib
from src.data_loader import load_football, load_nba
import numpy as np

st.set_page_config(page_title="Sample Football & NBA Predictor", layout="centered")
st.title("Sample Football & NBA Predictor (Demo)")

sport = st.selectbox("Choose sport", ["Football", "NBA"])

if sport == "Football":
    st.header("Football (Sample)")
    df = load_football('data/raw/football_matches_sample.csv')
    teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    home = st.selectbox("Home team", teams)
    away = st.selectbox("Away team", [t for t in teams if t!=home])
    if st.button("Predict football outcome"):
        # naive features: use average historical goals for the selected teams
        home_avg = df[df['home_team']==home]['home_score'].mean() if not df[df['home_team']==home].empty else df['home_score'].mean()
        away_avg = df[df['away_team']==away]['away_score'].mean() if not df[df['away_team']==away].empty else df['away_score'].mean()
        X = pd.DataFrame({'home_score':[home_avg],'away_score':[away_avg]})
        model = joblib.load('models/football_simple_lr.joblib')
        probs = model.predict_proba(X)[0]
        st.write("Predicted probabilities:")
        st.write(f"Home win: {probs[1]:.3f}, Draw: {probs[0]:.3f}, Away win: {probs[2]:.3f}")
        # compute expected score naive using averages
        st.write(f"Naive expected goals — Home: {home_avg:.2f}, Away: {away_avg:.2f}")

else:
    st.header("NBA (Sample)")
    df = load_nba('data/raw/nba_games_sample.csv')
    teams = sorted(list(set(df['home_team']).union(set(df['away_team']))))
    home = st.selectbox("Home team", teams)
    away = st.selectbox("Away team", [t for t in teams if t!=home])
    if st.button("Predict NBA outcome"):
        home_avg = df[df['home_team']==home]['home_points'].mean() if not df[df['home_team']==home].empty else df['home_points'].mean()
        away_avg = df[df['away_team']==away]['away_points'].mean() if not df[df['away_team']==away].empty else df['away_points'].mean()
        X = pd.DataFrame({'home_points':[home_avg],'away_points':[away_avg]})
        model = joblib.load('models/nba_simple_reg.joblib')
        pred_diff = model.predict(X)[0]
        # convert to win probability using logistic-like transform
        prob_home = 1 / (1 + np.exp(-pred_diff/10))  # scale factor for demo
        st.write(f"Predicted point differential (home - away): {pred_diff:.2f}")
        st.write(f"Estimated home win probability: {prob_home:.3f}")
        st.write(f"Naive expected points — Home: {home_avg:.1f}, Away: {away_avg:.1f}")
