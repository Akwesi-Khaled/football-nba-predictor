
import pandas as pd

def load_football(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def load_nba(path):
    df = pd.read_csv(path, parse_dates=['date'])
    return df
