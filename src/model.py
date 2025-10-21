import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_poisson_model(df):
    """Train Poisson regression for football expected goals"""
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]).copy()

    # Encode teams as dummy variables
    team_dummies = pd.get_dummies(pd.concat([df['HomeTeam'], df['AwayTeam']]), drop_first=False)
    home_dummies = team_dummies.loc[df.index]
    away_dummies = team_dummies.loc[df.index]

    X_home = pd.concat([home_dummies.add_prefix("home_"), away_dummies.add_prefix("away_")], axis=1)
    X_home = sm.add_constant(X_home)
    y_home = df['HomeGoals'].astype(float)

    X_away = pd.concat([home_dummies.add_prefix("home_"), away_dummies.add_prefix("away_")], axis=1)
    X_away = sm.add_constant(X_away)
    y_away = df['AwayGoals'].astype(float)

    home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()

    return home_model, away_model


def train_nba_model(df):
    """Train simple NBA win/loss logistic regression model"""
    # Clean data
    df = df.dropna(subset=['HomePoints', 'AwayPoints', 'Result']).copy()
    df['HomeWin'] = (df['Result'] == 'H').astype(int)

    # Features and target
    X = df[['HomePoints', 'AwayPoints']].astype(float)
    y = df['HomeWin']

    model = LogisticRegression()
    model.fit(X, y)

    return model
