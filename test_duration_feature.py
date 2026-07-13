"""
Test: does champion duration profile + duration-win correlation add signal?

Two champion-level signals, kept SEPARATE on purpose:
  1) duration_avg   -> does the game run long when this champ is picked
                        (regardless of who wins). Captures "this champ/comp
                        just makes games long."
  2) duration_slope -> does this champ's WIN RATE go up in long games vs
                        short games. Captures "this champ scales / benefits
                        from a long game."

Match-level "expected duration" = blend of:
  - both teams' own historical avg game duration (team pace)
  - the average duration_avg of the 10 champs picked (comp pace)

Then: duration_score(side) = mean(champ duration_slope for that side's picks)
      * (expected_duration - overall_avg_duration)
      -> rewards teams with late-scaling picks when the expected game is
         long, and early-game picks when the expected game is short.

feature = duration_score(blue) - duration_score(red)

Everything is trained on 2023-2025 only and evaluated on 2026 only
(no leakage), consistent with the existing grid-search methodology.

Run this from the same folder as proplay_matches.csv.
Requires: pandas, numpy, scikit-learn (pip install scikit-learn if missing)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

MIN_CHAMP_GAMES = 15          # min games for a champ's duration stats to count
DURATION_BLEND_TEAM_WEIGHT = 0.5   # weight on team avg duration vs champ avg duration
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Load + basic prep
# ---------------------------------------------------------------
df = pd.read_csv("proplay_matches.csv", parse_dates=["date"])
df = df.dropna(subset=["game_duration_min", "blue_picks", "red_picks", "blue_win"]).copy()
df = df.sort_values("date").reset_index(drop=True)

df["blue_picks_list"] = df["blue_picks"].str.split(",")
df["red_picks_list"] = df["red_picks"].str.split(",")

train_df = df[df["year"] <= 2025].copy()
test_df = df[df["year"] == 2026].copy()

print(f"Train games (2023-2025): {len(train_df)}")
print(f"Test games (2026):        {len(test_df)}")

if len(test_df) == 0:
    raise SystemExit("No 2026 games found in data — check the 'year' column / file.")

overall_avg_duration = train_df["game_duration_min"].mean()
print(f"Overall avg game duration (train): {overall_avg_duration:.2f} min")

# ---------------------------------------------------------------
# Build long-format (one row per team per game) for champ + team stats
# ---------------------------------------------------------------
def build_long(frame):
    blue = pd.DataFrame({
        "game_id": frame["game_id"],
        "date": frame["date"],
        "team": frame["blue_team"],
        "duration": frame["game_duration_min"],
        "win": frame["blue_win"],
        "picks": frame["blue_picks_list"],
    })
    red = pd.DataFrame({
        "game_id": frame["game_id"],
        "date": frame["date"],
        "team": frame["red_team"],
        "duration": frame["game_duration_min"],
        "win": 1 - frame["blue_win"],
        "picks": frame["red_picks_list"],
    })
    return pd.concat([blue, red], ignore_index=True)

train_long = build_long(train_df)

# ---------------------------------------------------------------
# Per-champion duration stats (TRAIN ONLY)
# ---------------------------------------------------------------
champ_rows = train_long.explode("picks").rename(columns={"picks": "champ"})
champ_rows["champ"] = champ_rows["champ"].str.strip()

champ_stats = {}
for champ, g in champ_rows.groupby("champ"):
    n = len(g)
    if n < MIN_CHAMP_GAMES:
        continue
    dur_avg = g["duration"].mean()
    median_dur = g["duration"].median()
    short = g[g["duration"] <= median_dur]
    long_ = g[g["duration"] > median_dur]
    # guard against empty long/short slices on tiny/edge distributions
    wr_short = short["win"].mean() if len(short) > 0 else np.nan
    wr_long = long_["win"].mean() if len(long_) > 0 else np.nan
    slope = (wr_long - wr_short) if not (np.isnan(wr_short) or np.isnan(wr_long)) else 0.0
    champ_stats[champ] = {
        "n_games": n,
        "duration_avg": dur_avg,
        "duration_slope": slope,
    }

champ_stats_df = pd.DataFrame(champ_stats).T.sort_values("duration_slope", ascending=False)
print(f"\nChampions meeting min-games threshold ({MIN_CHAMP_GAMES}+): {len(champ_stats_df)}")

print("\nTop 15 champs — win rate rises MOST in long games (late-game scaling):")
print(champ_stats_df.head(15)[["n_games", "duration_avg", "duration_slope"]]
      .rename(columns={"duration_slope": "long_game_win_boost"}))

print("\nBottom 15 champs — win rate DROPS most in long games (early-game/snowball):")
print(champ_stats_df.tail(15)[["n_games", "duration_avg", "duration_slope"]]
      .rename(columns={"duration_slope": "long_game_win_boost"}))

print("\nTop 15 champs by raw avg game duration when picked (comp just runs long):")
print(champ_stats_df.sort_values("duration_avg", ascending=False)
      .head(15)[["n_games", "duration_avg", "duration_slope"]])

print("\nBottom 15 champs by raw avg game duration when picked (comp ends fast):")
print(champ_stats_df.sort_values("duration_avg", ascending=True)
      .head(15)[["n_games", "duration_avg", "duration_slope"]])

# fallback for champs below threshold / unseen in test
fallback_duration_avg = train_long["duration"].mean()
fallback_slope = 0.0

def champ_lookup(champ, field, fallback):
    champ = champ.strip()
    if champ in champ_stats:
        return champ_stats[champ][field]
    return fallback

# ---------------------------------------------------------------
# Per-team avg duration (TRAIN ONLY) -- simple team pace signal
# ---------------------------------------------------------------
team_avg_duration = train_long.groupby("team")["duration"].mean().to_dict()
fallback_team_duration = overall_avg_duration

def team_dur_lookup(team):
    return team_avg_duration.get(team, fallback_team_duration)

# ---------------------------------------------------------------
# Simple baseline: team win rate (train only), + blue-side dummy
# ---------------------------------------------------------------
team_win_rate = train_long.groupby("team")["win"].mean().to_dict()
fallback_wr = train_long["win"].mean()

def wr_lookup(team):
    return team_win_rate.get(team, fallback_wr)

# ---------------------------------------------------------------
# Build features for TRAIN (for fitting logistic regression) and TEST
# ---------------------------------------------------------------
def build_features(frame):
    rows = []
    for _, r in frame.iterrows():
        blue_champs = [c.strip() for c in r["blue_picks_list"]]
        red_champs = [c.strip() for c in r["red_picks_list"]]

        blue_champ_dur_avg = np.mean([champ_lookup(c, "duration_avg", fallback_duration_avg) for c in blue_champs])
        red_champ_dur_avg = np.mean([champ_lookup(c, "duration_avg", fallback_duration_avg) for c in red_champs])

        blue_slope = np.mean([champ_lookup(c, "duration_slope", fallback_slope) for c in blue_champs])
        red_slope = np.mean([champ_lookup(c, "duration_slope", fallback_slope) for c in red_champs])

        blue_team_dur = team_dur_lookup(r["blue_team"])
        red_team_dur = team_dur_lookup(r["red_team"])

        # expected duration = blend of team pace (both sides) and comp pace (both sides)
        team_component = (blue_team_dur + red_team_dur) / 2
        comp_component = (blue_champ_dur_avg + red_champ_dur_avg) / 2
        expected_duration = (DURATION_BLEND_TEAM_WEIGHT * team_component
                              + (1 - DURATION_BLEND_TEAM_WEIGHT) * comp_component)

        dur_delta = expected_duration - overall_avg_duration
        blue_duration_score = blue_slope * dur_delta
        red_duration_score = red_slope * dur_delta
        duration_feature = blue_duration_score - red_duration_score

        rows.append({
            "game_id": r["game_id"],
            "blue_win": r["blue_win"],
            "team_wr_diff": wr_lookup(r["blue_team"]) - wr_lookup(r["red_team"]),
            "duration_feature": duration_feature,
            "expected_duration": expected_duration,
        })
    return pd.DataFrame(rows)

train_feat = build_features(train_df)
test_feat = build_features(test_df)

# ---------------------------------------------------------------
# Evaluate: does duration_feature correlate with outcome at all?
# ---------------------------------------------------------------
corr = np.corrcoef(test_feat["duration_feature"], test_feat["blue_win"])[0, 1]
print(f"\nRaw correlation: duration_feature vs blue_win (2026 holdout): {corr:.4f}")

# standalone AUC of the duration feature alone
solo_auc = roc_auc_score(test_feat["blue_win"], test_feat["duration_feature"])
print(f"Standalone AUC of duration_feature alone (2026 holdout): {solo_auc:.4f}  (0.50 = no signal)")

# ---------------------------------------------------------------
# Baseline model vs baseline+duration model
# ---------------------------------------------------------------
X_train_base = train_feat[["team_wr_diff"]].values
X_test_base = test_feat[["team_wr_diff"]].values

X_train_full = train_feat[["team_wr_diff", "duration_feature"]].values
X_test_full = test_feat[["team_wr_diff", "duration_feature"]].values

y_train = train_feat["blue_win"].values
y_test = test_feat["blue_win"].values

base_model = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_base, y_train)
full_model = LogisticRegression(random_state=RANDOM_STATE).fit(X_train_full, y_train)

base_probs = base_model.predict_proba(X_test_base)[:, 1]
full_probs = full_model.predict_proba(X_test_full)[:, 1]

base_auc = roc_auc_score(y_test, base_probs)
full_auc = roc_auc_score(y_test, full_probs)

base_acc = accuracy_score(y_test, base_probs > 0.5)
full_acc = accuracy_score(y_test, full_probs > 0.5)

print("\n" + "=" * 50)
print("RESULTS (2026 holdout)")
print("=" * 50)
print(f"Baseline (team_wr_diff only):        AUC {base_auc:.4f}  ACC {base_acc:.4f}")
print(f"Baseline + duration_feature:         AUC {full_auc:.4f}  ACC {full_acc:.4f}")
print(f"AUC uplift: {full_auc - base_auc:+.4f}")
print(f"ACC uplift: {full_acc - base_acc:+.4f}")

print("\nLogistic regression coefficients (full model):")
print(f"  team_wr_diff:      {full_model.coef_[0][0]:.4f}")
print(f"  duration_feature:  {full_model.coef_[0][1]:.4f}")
print("\n(Note: baseline here is deliberately simple -- just team win rate diff --")
print(" not the full V8 feature set. This is a go/no-go signal check, not a")
print(" final backtest. If uplift looks real, next step is wiring this into")
print(" build_dataset.py / train_and_save.py alongside the full V8 feature set.)")
