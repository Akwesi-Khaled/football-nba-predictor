import pandas as pd
import requests
from io import BytesIO
import streamlit as st

@st.cache_data(ttl=86400)  # refresh daily
def load_football_data():
    """Fetch real football data (EPL 2024/25)"""
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    content = requests.get(url).content
    df = pd.read_csv(BytesIO(content))
    df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].dropna()
    df.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'}, inplace=True)
    return df

@st.cache_data(ttl=86400)
def load_nba_data():
    """Load NBA data (placeholder CSV from Basketball Reference or local backup)."""
    try:
        df = pd.read_csv("data/raw/nba_2425.csv")
    except Exception:
        df = pd.DataFrame({
            'HomeTeam': ['LAL', 'BOS', 'GSW'],
            'AwayTeam': ['MIA', 'DAL', 'DEN'],
            'HomePoints': [112, 109, 104],
            'AwayPoints': [107, 115, 99],
            'Result': ['H', 'A', 'H']
        })
    return df
