import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import LogisticRegression

# -------------------------------
# FOOTBALL POISSON MODEL
# -------------------------------

def train_poisson_model(df):
    """Train Poisson regression model for expected goals"""

    # Drop missing
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]).copy()

    # Ensure proper data types
    df["HomeTeam"] = df["HomeTeam"].astype(str)
    df["AwayTeam"] = df["AwayTeam"].astype(str)
    df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
    df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")

    # Encode teams as dummy variables
    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))
    home_dummies = pd.get_dummies(df["HomeTeam"], prefix="home", drop_first=False)
    away_dummies = pd.get_dummies(df["AwayTeam"], prefix="away", drop_first=False)

    # Align dummy columns (important)
    for t in teams:
        if f"home_{t}" not in home_dummies:
            home_dummies[f"home_{t}"] = 0
        if f"away_{t}" not in away_dummies:
            away_dummies[f"away_{t}"] = 0

    X_home = pd.concat([home_dummies, away_dummies], axis=1)
    X_home = sm.add_constant(X_home)
    X_home = X_home.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_home = df["HomeGoals"].astype(float)

    X_away = pd.concat([home_dummies, away_dummies], axis=1)
    X_away = sm.add_constant(X_away)
    X_away = X_away.apply(pd.to_numeric, errors="coerce").fillna(0)

    y_away = df["AwayGoals"].astype(float)

    # Fit models safely
    try:
        home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
        away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()
    except Exception as e:
        print("⚠️ Model fitting failed. Check data types:")
        print(X_home.dtypes)
        raise e

    return home_model, away_model


# -------------------------------
# NBA LOGISTIC MODEL
# -------------------------------

def train_nba_model(df):
    """Train simple NBA win/loss logistic regression model"""
    df = df.dropna(subset=["HomePoints", "AwayPoints", "Result"]).copy()
    df["HomeWin"] = (df["Result"] == "H").astype(int)

    X = df[["HomePoints", "AwayPoints"]].astype(float)
    y = df["HomeWin"].astype(int)

    model = LogisticRegression()
    model.fit(X, y)

    return model
