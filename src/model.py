import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_poisson_model(df):
    """Train Poisson regression for football expected goals"""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_dummies = pd.get_dummies(df[['HomeTeam', 'AwayTeam']], drop_first=True)

    home_model = sm.GLM(df['HomeGoals'],
                        sm.add_constant(team_dummies),
                        family=sm.families.Poisson()).fit()

    away_model = sm.GLM(df['AwayGoals'],
                        sm.add_constant(team_dummies),
                        family=sm.families.Poisson()).fit()

    return home_model, away_model

def train_nba_model(df):
    """Simple NBA win classifier"""
    df['HomeWin'] = (df['Result'] == 'H').astype(int)
    X = df[['HomePoints', 'AwayPoints']]
    y = df['HomeWin']
    model = LogisticRegression()
    model.fit(X, y)
    return model
