import numpy as np

def predict_football(home_model, away_model, home_team, away_team):
    """Estimate expected goals and probabilities"""
    try:
        home_goals = home_model.predict().mean()
        away_goals = away_model.predict().mean()
        home_prob = 1 / (1 + np.exp(-(home_goals - away_goals)))
        draw_prob = 0.2
        away_prob = 1 - home_prob - draw_prob
        return home_prob, draw_prob, away_prob
    except Exception:
        return 0.33, 0.33, 0.34

def predict_nba(model, home_points, away_points):
    """Predict NBA win probability"""
    prob = model.predict_proba([[home_points, away_points]])[0][1]
    return prob
