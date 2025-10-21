
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression, LinearRegression
from src.data_loader import load_football, load_nba

def train_football(csv_in, model_out):
    df = load_football(csv_in).copy()
    # Simple features: difference in recent average goals for home/away (naive)
    df['goal_diff'] = df['home_score'] - df['away_score']
    # label: 0 draw, 1 home win, 2 away win
    df['label'] = df['goal_diff'].apply(lambda x: 1 if x>0 else (0 if x==0 else 2))
    X = df[['home_score', 'away_score']].fillna(0)
    y = df['label']
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    clf.fit(X, y)
    dump(clf, model_out)
    print("Football model saved to", model_out)

def train_nba(csv_in, model_out):
    df = load_nba(csv_in).copy()
    df['point_diff'] = df['home_points'] - df['away_points']
    X = df[['home_points', 'away_points']].fillna(0)
    y = df['point_diff']
    reg = LinearRegression()
    reg.fit(X, y)
    dump(reg, model_out)
    print("NBA model saved to", model_out)

if __name__ == "__main__":
    train_football('data/raw/football_matches_sample.csv', 'models/football_simple_lr.joblib')
    train_nba('data/raw/nba_games_sample.csv', 'models/nba_simple_reg.joblib')
