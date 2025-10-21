import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- FOOTBALL PREDICTION ---

def predict_football(home_model, away_model, home_team, away_team):
    """Predict expected goals and win/draw/loss probabilities"""

    # Extract team list from model
    teams = [c.replace("home_", "").replace("away_", "") for c in home_model.params.index if c.startswith("home_")]
    teams = sorted(list(set(teams)))

    # Create dummy input
    df_input = pd.DataFrame(columns=[f"home_{t}" for t in teams] + [f"away_{t}" for t in teams])
    df_input.loc[0] = 0
    if f"home_{home_team}" in df_input.columns:
        df_input.loc[0, f"home_{home_team}"] = 1
    if f"away_{away_team}" in df_input.columns:
        df_input.loc[0, f"away_{away_team}"] = 1

    # Add constant
    X = sm.add_constant(df_input, has_constant='add')

    # Predict goals
    home_exp_goals = home_model.predict(X)[0]
    away_exp_goals = away_model.predict(X)[0]

    # Estimate probabilities
    home_prob = 1 / (1 + np.exp(-(home_exp_goals - away_exp_goals)))
    draw_prob = max(0.15, 1 - abs(home_prob - 0.5) * 2)
    away_prob = 1 - home_prob - draw_prob

    return home_prob, draw_prob, away_prob


# --- NBA PREDICTION ---

def predict_nba(model, home_points, away_points):
    """Predict NBA win probability based on trained logistic regression"""
    X = np.array([[home_points, away_points]])
    prob_home_win = model.predict_proba(X)[0][1]
    prob_away_win = 1 - prob_home_win
    return prob_home_win, prob_away_win
