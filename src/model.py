import pandas as pd
import statsmodels.api as sm

def train_poisson_model(df):
    """Train Poisson regression for football expected goals"""
    # Drop missing and ensure numeric
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]).copy()

    # Encode teams as dummy variables
    team_dummies = pd.get_dummies(pd.concat([df['HomeTeam'], df['AwayTeam']]), drop_first=False)
    home_dummies = team_dummies.loc[df.index]
    away_dummies = team_dummies.loc[df.index]

    # Home goals model
    X_home = pd.concat([home_dummies.add_prefix("home_"), away_dummies.add_prefix("away_")], axis=1)
    X_home = sm.add_constant(X_home)
    y_home = df['HomeGoals'].astype(float)

    # Away goals model
    X_away = pd.concat([home_dummies.add_prefix("home_"), away_dummies.add_prefix("away_")], axis=1)
    X_away = sm.add_constant(X_away)
    y_away = df['AwayGoals'].astype(float)

    # Fit models
    home_model = sm.GLM(y_home, X_home, family=sm.families.Poisson()).fit()
    away_model = sm.GLM(y_away, X_away, family=sm.families.Poisson()).fit()

    return home_model, away_model
