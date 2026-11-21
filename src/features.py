# Constructs the training, validation, and test set that all our models will use
# To prevent leakage we will construct each set based on chronological dates
# Training set will get matches from June 2012 - Dec 2015
# Validation set will get matches from Jan 2016 - Jun 2016
# Test set will get matches from Jul 2016 - Feb 2017

import os
import re
import math 
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parent.parent

data_path = ROOT_DIR / "data/processed/cleaned_matches.parquet"
initial_mu = 0.50

def smoothed_log_odds(w: int, n: int, mu: float, k: float = 200.0) -> float:
    r = (w + k * mu) / (n + k)
    return math.log(r / (1.0 - r))

def compute_priors(df: pd.DataFrame) -> pd.DataFrame:
    player_stats = defaultdict(lambda: {
        "serve_won": 0,
        "serve_played": 0,
        "return_won": 0,
        "return_played": 0,
    })
    
    tour_stats = defaultdict(lambda: {
        "serve_won": 0,
        "serve_played": 0,
    })

    a_serve_priors = []
    a_return_priors = []
    b_serve_priors = []
    b_return_priors = []

    # First we should sort the rows based on date
    df = df.sort_values(by="date").reset_index(drop=True)
    for date, day_df in df.groupby("date"):
        for row in day_df.itertuples():
            tour = tour_stats[row.tour]
            stats_a = player_stats[row.player_a]
            stats_b = player_stats[row.player_b]
            mu = 0
            if tour["serve_played"] > 0:
                mu = tour["serve_won"] / tour["serve_played"]
            else:
                mu = initial_mu

            a_serve_priors.append(smoothed_log_odds(stats_a["serve_won"], stats_a["serve_played"], mu))
            a_return_priors.append(smoothed_log_odds(stats_a["return_won"], stats_a["return_played"], 1 - mu))
            b_serve_priors.append(smoothed_log_odds(stats_b["serve_won"], stats_b["serve_played"], mu))
            b_return_priors.append(smoothed_log_odds(stats_b["return_won"], stats_b["return_played"], 1 - mu)) 

        for row in day_df.itertuples():
            t = tour_stats[row.tour]
            stats_a = player_stats[row.player_a]
            stats_b = player_stats[row.player_b]

            for pt in row.points:
                server_won = (pt["winner"] == pt["server"])
                t["serve_played"] += 1
                if server_won:
                    t["serve_won"] += 1

                if pt["server"] == 0:  
                    stats_a["serve_played"] += 1
                    stats_b["return_played"] += 1
                    if server_won:
                        stats_a["serve_won"] += 1
                    else:
                        stats_b["return_won"] += 1
                else:                 
                    stats_b["serve_played"] += 1
                    stats_a["return_played"] += 1
                    if server_won:
                        stats_b["serve_won"] += 1
                    else:
                        stats_a["return_won"] += 1

    df["a_serve_prior"] = a_serve_priors
    df["a_return_prior"] = a_return_priors
    df["b_serve_prior"] = b_serve_priors
    df["b_return_prior"] = b_return_priors
    return df
            
def main():
    df = compute_priors(pd.read_parquet(data_path))
    training_set = df[df['date'].between("2012-06-01", "2015-12-31", "both")]
    validation_set = df[df['date'].between("2016-01-01", "2016-06-30", "both")]
    test_set = df[df['date'].between("2016-07-01", "2017-02-18", "both")]



if __name__ == "__main__":
    main()