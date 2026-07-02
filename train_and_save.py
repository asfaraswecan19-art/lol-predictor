import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score

# =================================================================
# TRAIN AND SAVE MODELS V8
# Win:  10% PC / 90% role-champ / H2H cap 60%
#       Weighted form window=8 + recent win rate w=20 blend=0.2
#       min_pc=12 | n_estimators=125 | max_depth=2 | lr=0.1
#       + gold trajectory: avg_gd20 + late_scaling (window=15)
# FT5:  extended kill window (10/15/20/25min) + H2H cap 60%
#       n_estimators=125 | max_depth=1 | lr=0.03 | form=8
# =================================================================

TEAM_ALIASES = {
    'Team BDS': 'Team Shifters',
    'BDS':      'Team Shifters',
}
def normalize_team(name):
    return TEAM_ALIASES.get(str(name), str(name))

WIN_DATA          = 'proplay_matches.csv'
FT5_DATA          = 'kill_timelines.csv'
FORM_WINDOW       = 8
RECENT_WINDOW     = 20
RECENT_WEIGHT     = 0.2
GOLD_WINDOW       = 15
BLUE_SIDE_WINRATE = 0.5312
MIN_PC_GAMES      = 12
MIN_ROLE_GAMES    = 5
PC_WEIGHT         = 0.10
RC_WEIGHT         = 0.90
H2H_CAP           = 0.60
POSITIONS         = ['top', 'jng', 'mid', 'adc', 'sup']
TARGET_LEAGUES    = ['LCK','LPL','LEC','LCS','CBLOL','MSI','WLDs','LTA N','LTA S','LTA','FST']

def cap_h2h(rate):
    return max(1 - H2H_CAP, min(H2H_CAP, rate))

def weighted_form(hist, window):
    h = hist[-window:] if hist else []
    if not h: return 0.5
    weights = list(range(1, len(h) + 1))
    return sum(v * w for v, w in zip(h, weights)) / sum(weights)

def get_blended_avg(players, picks, pc_r, rc_r):
    rates = []
    for i, (player, champ) in enumerate(zip(players, picks)):
        pos    = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        pc_key = (player.strip(), champ.strip())
        rc_key = (pos, champ.strip())
        pc_val = pc_r.get(pc_key, 0.5)
        rc_val = rc_r.get(rc_key, 0.5)
        rates.append(PC_WEIGHT * pc_val + RC_WEIGHT * rc_val)
    return sum(rates) / len(rates) if rates else 0.5

# =================================================================
# LOAD GOLD TRAJECTORY FROM RAW ORACLE'S ELIXIR FILES
# =================================================================
import glob
from collections import defaultdict

print("Loading gold trajectory data from Oracle's Elixir...")
raw_files = sorted([f for f in glob.glob('*LoL_esports_match_data_from_OraclesElixir*.csv')
                    if '2022' not in f])
print(f"  Files: {raw_files}")

raw_dfs = []
for f in raw_files:
    try:
        tmp = pd.read_csv(f, low_memory=False)
        raw_dfs.append(tmp)
    except Exception as e:
        print(f"  {f}: ERROR {e}")

raw = pd.concat(raw_dfs, ignore_index=True)
raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
raw['year'] = raw['date'].dt.year
raw = raw[raw['league'].isin(TARGET_LEAGUES)].copy()
raw['teamname'] = raw['teamname'].apply(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
team_raw = raw[raw['position'] == 'team'].sort_values('date').reset_index(drop=True)

for col in ['golddiffat10', 'golddiffat20', 'gamelength', 'teamkills']:
    team_raw[col] = pd.to_numeric(team_raw[col], errors='coerce').fillna(0)

gd10_hist      = defaultdict(list)
gd20_hist      = defaultdict(list)
gamelength_hist = defaultdict(list)
kills_hist      = defaultdict(list)
gold_lookup = {}  # (date_str, teamname) -> {avg_gd20, late_scaling}
STYLE_WINDOW = 15

for _, row in team_raw.iterrows():
    team     = row['teamname']
    date_str = str(row['date'])[:10]
    w        = GOLD_WINDOW
    sw       = STYLE_WINDOW

    h10 = gd10_hist[team]
    h20 = gd20_hist[team]
    avg_gd20     = sum(h20[-w:]) / len(h20[-w:]) if h20 else 0.0
    avg_gd10     = sum(h10[-w:]) / len(h10[-w:]) if h10 else 0.0
    late_scaling = avg_gd20 - avg_gd10

    gold_lookup[(date_str, team)] = {
        'avg_gd20':     avg_gd20,
        'late_scaling': late_scaling,
    }

    gd10_hist[team].append(float(row['golddiffat10']))
    gd20_hist[team].append(float(row['golddiffat20']))
    if row['gamelength'] > 0:
        gamelength_hist[team].append(float(row['gamelength']))
    if row['teamkills'] > 0:
        kills_hist[team].append(float(row['teamkills']))

# Build final avg gamelength and avg kills per team (rolling last 15)
team_avg_gamelength = {}
team_avg_kills      = {}
for team in set(list(gamelength_hist.keys()) + list(kills_hist.keys())):
    gh = gamelength_hist[team][-STYLE_WINDOW:]
    kh = kills_hist[team][-STYLE_WINDOW:]
    team_avg_gamelength[team] = sum(gh)/len(gh) if gh else 0.0
    team_avg_kills[team]      = sum(kh)/len(kh) if kh else 0.0

print(f"  Gold lookup entries: {len(gold_lookup)}")
covered = sum(1 for k in gold_lookup if gold_lookup[k]['avg_gd20'] != 0.0)
print(f"  Non-zero avg_gd20:   {covered} / {len(gold_lookup)}")
print(f"  Teams with gamelength data: {len(team_avg_gamelength)}")
print(f"  Teams with kills data:      {len(team_avg_kills)}")

# =================================================================
# =================================================================
# WIN MODEL
# =================================================================
print("\nLoading win data...")
win_df = pd.read_csv(WIN_DATA)
win_df['blue_team']    = win_df['blue_team'].apply(normalize_team)
win_df['red_team']     = win_df['red_team'].apply(normalize_team)
win_df['blue_picks']   = win_df['blue_picks'].apply(lambda x: x.split(','))
win_df['red_picks']    = win_df['red_picks'].apply(lambda x: x.split(','))
win_df['blue_players'] = win_df['blue_players'].apply(lambda x: str(x).split(','))
win_df['red_players']  = win_df['red_players'].apply(lambda x: str(x).split(','))

if 'date' in win_df.columns:
    win_df['date'] = pd.to_datetime(win_df['date'], errors='coerce')
    win_df = win_df.sort_values('date').reset_index(drop=True)

role_champ_wins  = {}
role_champ_games = {}
for _, row in win_df.iterrows():
    result = row['blue_win']
    for i, champ in enumerate(row['blue_picks']):
        pos = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        key = (pos, champ.strip())
        role_champ_games[key] = role_champ_games.get(key, 0) + 1
        role_champ_wins[key]  = role_champ_wins.get(key,  0) + result
    for i, champ in enumerate(row['red_picks']):
        pos = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        key = (pos, champ.strip())
        role_champ_games[key] = role_champ_games.get(key, 0) + 1
        role_champ_wins[key]  = role_champ_wins.get(key,  0) + (1 - result)
role_champ_rate = {
    k: role_champ_wins[k] / role_champ_games[k]
    for k in role_champ_games if role_champ_games[k] >= MIN_ROLE_GAMES
}
print(f"  Role-specific champion combos (win): {len(role_champ_rate)}")

pc_wins  = {}
pc_games = {}
for _, row in win_df.iterrows():
    result = row['blue_win']
    for player, champ in zip(row['blue_players'], row['blue_picks']):
        key = (player.strip(), champ.strip())
        pc_games[key] = pc_games.get(key, 0) + 1
        pc_wins[key]  = pc_wins.get(key,  0) + result
    for player, champ in zip(row['red_players'], row['red_picks']):
        key = (player.strip(), champ.strip())
        pc_games[key] = pc_games.get(key, 0) + 1
        pc_wins[key]  = pc_wins.get(key,  0) + (1 - result)
pc_rate = {k: pc_wins[k] / pc_games[k] for k in pc_games if pc_games[k] >= MIN_PC_GAMES}
print(f"  Player-champion combos (win, 5+ games): {len(pc_rate)}")

win_team_wins  = {}
win_team_games = {}
for _, row in win_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    win_team_games[blue] = win_team_games.get(blue, 0) + 1
    win_team_games[red]  = win_team_games.get(red,  0) + 1
    win_team_wins[blue]  = win_team_wins.get(blue, 0) + row['blue_win']
    win_team_wins[red]   = win_team_wins.get(red,  0) + (1 - row['blue_win'])
win_team_rate = {t: win_team_wins[t] / win_team_games[t] for t in win_team_games}

win_champ_wins  = {}
win_champ_games = {}
for _, row in win_df.iterrows():
    result = row['blue_win']
    for c in row['blue_picks']:
        win_champ_games[c] = win_champ_games.get(c, 0) + 1
        win_champ_wins[c]  = win_champ_wins.get(c,  0) + result
    for c in row['red_picks']:
        win_champ_games[c] = win_champ_games.get(c, 0) + 1
        win_champ_wins[c]  = win_champ_wins.get(c,  0) + (1 - result)
win_champ_rate = {c: win_champ_wins[c] / win_champ_games[c] for c in win_champ_games}

win_h2h = {}
for _, row in win_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    matchup = tuple(sorted([blue, red]))
    if matchup not in win_h2h:
        win_h2h[matchup] = {}
    win_h2h[matchup][blue] = win_h2h[matchup].get(blue, 0) + row['blue_win']
    win_h2h[matchup][red]  = win_h2h[matchup].get(red,  0) + (1 - row['blue_win'])

win_team_recent = {}
win_df_sorted = win_df.copy().reset_index(drop=True)
for idx, row in win_df_sorted.iterrows():
    blue, red = row['blue_team'], row['red_team']
    if blue not in win_team_recent: win_team_recent[blue] = []
    if red  not in win_team_recent: win_team_recent[red]  = []

    b_hist = win_team_recent[blue]
    r_hist = win_team_recent[red]

    b_form      = weighted_form(b_hist, FORM_WINDOW)
    r_form      = weighted_form(r_hist, FORM_WINDOW)
    b_recent_wr = sum(b_hist[-RECENT_WINDOW:]) / len(b_hist[-RECENT_WINDOW:]) if b_hist else 0.5
    r_recent_wr = sum(r_hist[-RECENT_WINDOW:]) / len(r_hist[-RECENT_WINDOW:]) if r_hist else 0.5

    win_df_sorted.at[idx, 'blue_form']      = b_form
    win_df_sorted.at[idx, 'red_form']       = r_form
    win_df_sorted.at[idx, 'blue_recent_wr'] = b_recent_wr
    win_df_sorted.at[idx, 'red_recent_wr']  = r_recent_wr

    win_team_recent[blue].append(1 if row['blue_win'] == 1 else 0)
    win_team_recent[red].append(0  if row['blue_win'] == 1 else 1)

win_df = win_df_sorted
win_df['form_diff']           = win_df['blue_form']      - win_df['red_form']
win_df['recent_wr_diff']      = win_df['blue_recent_wr'] - win_df['red_recent_wr']
win_df['blue_side_advantage'] = BLUE_SIDE_WINRATE
win_df['date_str']            = win_df['date'].astype(str).str[:10]

# Gold trajectory features from Oracle's Elixir
def get_gold_feat(row, team_col, feat):
    key = (row['date_str'], row[team_col])
    return gold_lookup.get(key, {}).get(feat, 0.0)

win_df['blue_avg_gd20']    = win_df.apply(lambda r: get_gold_feat(r, 'blue_team', 'avg_gd20'),    axis=1)
win_df['red_avg_gd20']     = win_df.apply(lambda r: get_gold_feat(r, 'red_team',  'avg_gd20'),    axis=1)
win_df['gd20_diff']        = win_df['blue_avg_gd20']    - win_df['red_avg_gd20']
win_df['blue_late_scaling']= win_df.apply(lambda r: get_gold_feat(r, 'blue_team', 'late_scaling'),axis=1)
win_df['red_late_scaling'] = win_df.apply(lambda r: get_gold_feat(r, 'red_team',  'late_scaling'),axis=1)
win_df['late_scaling_diff']= win_df['blue_late_scaling'] - win_df['red_late_scaling']

covered = (win_df['blue_avg_gd20'] != 0.0).sum()
print(f"  Gold features matched: {covered}/{len(win_df)} games")
win_df['blue_team_winrate']   = win_df['blue_team'].map(win_team_rate)
win_df['red_team_winrate']    = win_df['red_team'].map(win_team_rate)
win_df['team_winrate_diff']   = win_df['blue_team_winrate'] - win_df['red_team_winrate']
win_df['blue_team_games']     = win_df['blue_team'].map(win_team_games)
win_df['red_team_games']      = win_df['red_team'].map(win_team_games)

win_df['h2h_winrate'] = win_df.apply(
    lambda row: cap_h2h(
        (win_h2h.get(tuple(sorted([row['blue_team'], row['red_team']])), {})
         .get(row['blue_team'], 0)) /
        max(sum(win_h2h.get(tuple(sorted([row['blue_team'],
            row['red_team']])), {}).values()), 1)), axis=1)

win_df['blue_avg_winrate'] = win_df['blue_picks'].apply(
    lambda picks: sum(win_champ_rate.get(c, 0.5) for c in picks) / len(picks))
win_df['red_avg_winrate']  = win_df['red_picks'].apply(
    lambda picks: sum(win_champ_rate.get(c, 0.5) for c in picks) / len(picks))
win_df['winrate_diff']     = win_df['blue_avg_winrate'] - win_df['red_avg_winrate']

win_df['blue_pc_avg'] = win_df.apply(
    lambda row: get_blended_avg(row['blue_players'], row['blue_picks'],
                                pc_rate, role_champ_rate), axis=1)
win_df['red_pc_avg']  = win_df.apply(
    lambda row: get_blended_avg(row['red_players'], row['red_picks'],
                                pc_rate, role_champ_rate), axis=1)
win_df['pc_avg_diff'] = win_df['blue_pc_avg'] - win_df['red_pc_avg']

win_mlb = MultiLabelBinarizer()
win_mlb.fit(win_df['blue_picks'] + win_df['red_picks'])
win_blue_enc = pd.DataFrame(win_mlb.transform(win_df['blue_picks']),
    columns=['blue_' + c for c in win_mlb.classes_]).reset_index(drop=True)
win_red_enc  = pd.DataFrame(win_mlb.transform(win_df['red_picks']),
    columns=['red_'  + c for c in win_mlb.classes_]).reset_index(drop=True)
win_extra = win_df[[
    'blue_team_winrate', 'red_team_winrate', 'team_winrate_diff',
    'blue_team_games',   'red_team_games',
    'blue_avg_winrate',  'red_avg_winrate',  'winrate_diff',
    'h2h_winrate',
    'blue_form',         'red_form',         'form_diff',
    'blue_recent_wr',    'red_recent_wr',    'recent_wr_diff',
    'blue_side_advantage',
    'blue_pc_avg',       'red_pc_avg',       'pc_avg_diff',
    'blue_avg_gd20',     'red_avg_gd20',     'gd20_diff',
    'blue_late_scaling', 'red_late_scaling', 'late_scaling_diff',
]].reset_index(drop=True)

win_X = pd.concat([win_blue_enc, win_red_enc, win_extra], axis=1)
win_y = win_df['blue_win'].reset_index(drop=True)

# Time-based split — no data leakage
# Train: 2023-2024 | Val: 2025 (for hyperparameter search) | Final train: 2023-2025
if 'year' not in win_df.columns:
    win_df['year'] = pd.to_datetime(win_df['date'], errors='coerce').dt.year
win_year = win_df['year'].reset_index(drop=True)

val_mask   = win_year == 2025
train_mask = win_year <= 2024

print(f"\n  Hyperparameter search (train=2023-24, val=2025)...")
print(f"  Train: {train_mask.sum()} games | Val: {val_mask.sum()} games")

from sklearn.metrics import roc_auc_score as _auc

param_grid = [
    {'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.1},
    {'n_estimators': 125, 'max_depth': 2, 'learning_rate': 0.1},
    {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.1},
    {'n_estimators': 125, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 125, 'max_depth': 2, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 2, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 150, 'max_depth': 2, 'learning_rate': 0.08},
]

best_auc    = 0
best_params = param_grid[1]  # default fallback

for params in param_grid:
    m = GradientBoostingClassifier(**params, random_state=42)
    m.fit(win_X[train_mask], win_y[train_mask])
    pr = m.predict_proba(win_X[val_mask])[:,1]
    auc = _auc(win_y[val_mask], pr)
    acc = (m.predict(win_X[val_mask]) == win_y[val_mask].values).mean()
    marker = ' ← best' if auc > best_auc else ''
    print(f"  est={params['n_estimators']} depth={params['max_depth']} lr={params['learning_rate']}  "
          f"→  AUC:{auc:.4f}  Acc:{acc*100:.2f}%{marker}")
    if auc > best_auc:
        best_auc    = auc
        best_params = params

print(f"\n  Best params: {best_params} (val AUC: {best_auc:.4f})")

# Train final model on ALL data (2023-2025) with best params
print("Training win model on full 2023-2025 data...")
win_base  = GradientBoostingClassifier(**best_params, random_state=42)
win_model = CalibratedClassifierCV(win_base, method='isotonic', cv=5)
win_model.fit(win_X, win_y)
print("✅ Win model trained")

team_lineups = {}
for _, row in win_df.iterrows():
    b_players = [p.strip() for p in row['blue_players']]
    r_players = [p.strip() for p in row['red_players']]
    team_lineups[row['blue_team']] = {
        'top': b_players[0] if len(b_players) > 0 else '',
        'jng': b_players[1] if len(b_players) > 1 else '',
        'mid': b_players[2] if len(b_players) > 2 else '',
        'adc': b_players[3] if len(b_players) > 3 else '',
        'sup': b_players[4] if len(b_players) > 4 else '',
    }
    team_lineups[row['red_team']] = {
        'top': r_players[0] if len(r_players) > 0 else '',
        'jng': r_players[1] if len(r_players) > 1 else '',
        'mid': r_players[2] if len(r_players) > 2 else '',
        'adc': r_players[3] if len(r_players) > 3 else '',
        'sup': r_players[4] if len(r_players) > 4 else '',
    }

all_teams  = sorted(set(win_df['blue_team'].tolist() + win_df['red_team'].tolist()))
all_champs = sorted(set(
    [c for picks in win_df['blue_picks'] for c in picks] +
    [c for picks in win_df['red_picks']  for c in picks]))

if 'Team Shifters' in win_team_games:
    print(f"  Alias check — Team Shifters: {win_team_games['Team Shifters']} games")

# =================================================================
# FT5 MODEL
# =================================================================
ft5_df = pd.read_csv(FT5_DATA)
ft5_df['blue_team'] = ft5_df['blue_team'].apply(normalize_team)
ft5_df['red_team']  = ft5_df['red_team'].apply(normalize_team)
ft5_df = ft5_df[~ft5_df['tournament'].isin(['LPL'])].copy().reset_index(drop=True)
ft5_df['blue_picks']   = ft5_df['blue_picks'].apply(lambda x: x.split(','))
ft5_df['red_picks']    = ft5_df['red_picks'].apply(lambda x: x.split(','))
ft5_df['blue_players'] = ft5_df['blue_players'].apply(lambda x: str(x).split(','))
ft5_df['red_players']  = ft5_df['red_players'].apply(lambda x: str(x).split(','))
ft5_df['first_to_five_binary'] = ft5_df['first_to_five'].apply(
    lambda x: 1 if x == 'blue' else 0)

# Option 2: exclude ambiguous games from TRAINING only
# Ambiguous = both teams reached 5 kills in same 5-min window
if 'is_ambiguous' in ft5_df.columns:
    ft5_df['is_ambiguous'] = ft5_df['is_ambiguous'].fillna(0).astype(int)
    n_ambig = ft5_df['is_ambiguous'].sum()
    print(f"  FT5 ambiguous games (excluded from training): {n_ambig}")
else:
    ft5_df['is_ambiguous'] = 0
    print(f"  is_ambiguous column not found — training on all games")

ft5_champ_wins  = {}
ft5_champ_games = {}
for _, row in ft5_df.iterrows():
    result = row['first_to_five_binary']
    for c in row['blue_picks']:
        ft5_champ_games[c] = ft5_champ_games.get(c, 0) + 1
        ft5_champ_wins[c]  = ft5_champ_wins.get(c,  0) + result
    for c in row['red_picks']:
        ft5_champ_games[c] = ft5_champ_games.get(c, 0) + 1
        ft5_champ_wins[c]  = ft5_champ_wins.get(c,  0) + (1 - result)
champ_aggression = {c: ft5_champ_wins[c] / ft5_champ_games[c] for c in ft5_champ_games}

ft5_team_wins  = {}
ft5_team_games = {}
for _, row in ft5_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    ft5_team_games[blue] = ft5_team_games.get(blue, 0) + 1
    ft5_team_games[red]  = ft5_team_games.get(red,  0) + 1
    ft5_team_wins[blue]  = ft5_team_wins.get(blue, 0) + row['first_to_five_binary']
    ft5_team_wins[red]   = ft5_team_wins.get(red,  0) + (1 - row['first_to_five_binary'])
team_early_rate = {t: ft5_team_wins[t] / ft5_team_games[t] for t in ft5_team_games}

ft5_df['blue_aggression'] = ft5_df['blue_picks'].apply(
    lambda picks: sum(champ_aggression.get(c, 0.5) for c in picks) / len(picks))
ft5_df['red_aggression']  = ft5_df['red_picks'].apply(
    lambda picks: sum(champ_aggression.get(c, 0.5) for c in picks) / len(picks))
ft5_df['aggression_diff'] = ft5_df['blue_aggression'] - ft5_df['red_aggression']
ft5_df['blue_early_rate'] = ft5_df['blue_team'].map(team_early_rate)
ft5_df['red_early_rate']  = ft5_df['red_team'].map(team_early_rate)
ft5_df['early_rate_diff'] = ft5_df['blue_early_rate'] - ft5_df['red_early_rate']

ft5_avg_time    = {}
ft5_time_counts = {}
for _, row in ft5_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    if row['blue_time'] > 0:
        ft5_avg_time[blue]    = ft5_avg_time.get(blue, 0) + row['blue_time']
        ft5_time_counts[blue] = ft5_time_counts.get(blue, 0) + 1
    if row['red_time'] > 0:
        ft5_avg_time[red]    = ft5_avg_time.get(red, 0) + row['red_time']
        ft5_time_counts[red] = ft5_time_counts.get(red, 0) + 1
team_kill_speed = {t: ft5_avg_time[t] / ft5_time_counts[t]
                   for t in ft5_avg_time if ft5_time_counts[t] > 0}

ft5_df['blue_kill_speed'] = ft5_df['blue_team'].map(team_kill_speed).fillna(10.0)
ft5_df['red_kill_speed']  = ft5_df['red_team'].map(team_kill_speed).fillna(10.0)
ft5_df['speed_diff']      = ft5_df['red_kill_speed'] - ft5_df['blue_kill_speed']

ft5_h2h = {}
for _, row in ft5_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    matchup = tuple(sorted([blue, red]))
    if matchup not in ft5_h2h:
        ft5_h2h[matchup] = {}
    ft5_h2h[matchup][blue] = ft5_h2h[matchup].get(blue, 0) + row['first_to_five_binary']
    ft5_h2h[matchup][red]  = ft5_h2h[matchup].get(red,  0) + (1 - row['first_to_five_binary'])

ft5_team_recent = {}
ft5_df_sorted = ft5_df.copy().reset_index(drop=True)
for idx, row in ft5_df_sorted.iterrows():
    blue, red = row['blue_team'], row['red_team']
    if blue not in ft5_team_recent: ft5_team_recent[blue] = []
    if red  not in ft5_team_recent: ft5_team_recent[red]  = []
    b_form = weighted_form(ft5_team_recent[blue], FORM_WINDOW)
    r_form = weighted_form(ft5_team_recent[red],  FORM_WINDOW)
    ft5_df_sorted.at[idx, 'blue_early_form'] = b_form
    ft5_df_sorted.at[idx, 'red_early_form']  = r_form
    ft5_team_recent[blue].append(1 if row['first_to_five_binary'] == 1 else 0)
    ft5_team_recent[red].append(0  if row['first_to_five_binary'] == 1 else 1)
ft5_df = ft5_df_sorted
ft5_df['early_form_diff'] = ft5_df['blue_early_form'] - ft5_df['red_early_form']

ft5_df['h2h_early_rate'] = ft5_df.apply(
    lambda row: cap_h2h(
        (ft5_h2h.get(tuple(sorted([row['blue_team'], row['red_team']])), {})
         .get(row['blue_team'], 0)) /
        max(sum(ft5_h2h.get(tuple(sorted([row['blue_team'],
            row['red_team']])), {}).values()), 1)), axis=1)

ft5_df['blue_side_ft5'] = 0.548  # kept for payload compatibility but not used in model

ft5_mlb = MultiLabelBinarizer()
ft5_mlb.fit(ft5_df['blue_picks'] + ft5_df['red_picks'])
ft5_blue_enc = pd.DataFrame(ft5_mlb.transform(ft5_df['blue_picks']),
    columns=['blue_' + c for c in ft5_mlb.classes_]).reset_index(drop=True)
ft5_red_enc  = pd.DataFrame(ft5_mlb.transform(ft5_df['red_picks']),
    columns=['red_'  + c for c in ft5_mlb.classes_]).reset_index(drop=True)
ft5_extra = ft5_df[[
    'blue_aggression',  'red_aggression',  'aggression_diff',
    'blue_early_rate',  'red_early_rate',  'early_rate_diff',
    'blue_kill_speed',  'red_kill_speed',  'speed_diff',
    'h2h_early_rate',
    'blue_early_form',  'red_early_form',  'early_form_diff',
]].reset_index(drop=True)

ft5_X = pd.concat([ft5_blue_enc, ft5_red_enc, ft5_extra], axis=1)
ft5_y = ft5_df['first_to_five_binary'].reset_index(drop=True)

# Option 2: train only on non-ambiguous games
ft5_train_mask = ft5_df['is_ambiguous'] == 0
print(f"  FT5 training on {ft5_train_mask.sum()} non-ambiguous games "
      f"(dropped {(~ft5_train_mask).sum()} ambiguous)")
ft5_X_train = ft5_X[ft5_train_mask]
ft5_y_train = ft5_y[ft5_train_mask]

print("Training FT5 model...")
ft5_base  = GradientBoostingClassifier(n_estimators=125, max_depth=1, learning_rate=0.03, random_state=42)
ft5_model = CalibratedClassifierCV(ft5_base, method='isotonic', cv=5)
ft5_model.fit(ft5_X_train, ft5_y_train)
print("✅ FT5 model trained")

# =================================================================
# SAVE
# =================================================================
payload = {
    'win_model':        win_model,
    'win_mlb':          win_mlb,
    'win_team_rate':    win_team_rate,
    'win_team_games':   win_team_games,
    'win_champ_rate':   win_champ_rate,
    'win_h2h':          win_h2h,
    'win_team_recent':  win_team_recent,
    'pc_rate':          pc_rate,
    'pc_games':         pc_games,
    'role_champ_rate':  role_champ_rate,
    'ft5_model':        ft5_model,
    'ft5_mlb':          ft5_mlb,
    'champ_aggression': champ_aggression,
    'team_early_rate':  team_early_rate,
    'team_kill_speed':  team_kill_speed,
    'team_avg_gamelength': team_avg_gamelength,
    'team_avg_kills':      team_avg_kills,
    'ft5_h2h':          ft5_h2h,
    'ft5_team_recent':  ft5_team_recent,
    'ft5_team_games':   ft5_team_games,
    'team_lineups':     team_lineups,
    'all_teams':        all_teams,
    'all_champs':       all_champs,
    'team_aliases':     TEAM_ALIASES,
    'pc_weight':        PC_WEIGHT,
    'rc_weight':        RC_WEIGHT,
    'h2h_cap':          H2H_CAP,
    'min_pc_games':     MIN_PC_GAMES,
    'form_window':      FORM_WINDOW,
    'recent_window':    RECENT_WINDOW,
    'recent_weight':    RECENT_WEIGHT,
    'gold_window':      GOLD_WINDOW,
    'gold_lookup':      gold_lookup,
}

with open('model_payload.pkl', 'wb') as f:
    pickle.dump(payload, f)

import os
size_mb = os.path.getsize('model_payload.pkl') / (1024 * 1024)
print(f"\n✅ Saved model_payload.pkl ({size_mb:.1f} MB)")
print(f"   Teams:           {len(all_teams)}")
print(f"   Champions:       {len(all_champs)}")
print(f"   PC combos (win): {len(pc_rate)}")
print(f"   RC combos (win): {len(role_champ_rate)}")
print(f"   Form window:     {FORM_WINDOW} (weighted)")
print(f"   Recent window:   {RECENT_WINDOW} | blend: {RECENT_WEIGHT}")
print(f"   Blend: {int(PC_WEIGHT*100)}% PC / {int(RC_WEIGHT*100)}% RC | H2H cap: {int(H2H_CAP*100)}% | Min PC: {MIN_PC_GAMES}")
print(f"   Win GBM:  n_estimators={best_params['n_estimators']}, max_depth={best_params['max_depth']}, lr={best_params['learning_rate']} (grid searched)")
print(f"   FT5 GBM:  n_estimators=125, max_depth=1, lr=0.03")
print(f"   Gold window: {GOLD_WINDOW} games (avg_gd20 + late_scaling)")

# =================================================================
# TIER 2 MODEL — same pipeline, different data files
# =================================================================
T2_DATA  = 'proplay_matches_t2.csv'
T2_FT5   = 'kill_timelines_t2.csv'

if not os.path.exists(T2_DATA):
    print(f"\n⚠️  {T2_DATA} not found — skipping tier 2 model")
    print(f"   Run build_dataset.py first to generate tier 2 data")
else:
    print(f"\n{'='*55}")
    print(f"  TRAINING TIER 2 MODEL")
    print(f"{'='*55}")

    df_t2 = pd.read_csv(T2_DATA)
    df_t2['blue_team']    = df_t2['blue_team'].apply(normalize_team)
    df_t2['red_team']     = df_t2['red_team'].apply(normalize_team)
    df_t2['blue_picks']   = df_t2['blue_picks'].apply(lambda x: x.split(','))
    df_t2['red_picks']    = df_t2['red_picks'].apply(lambda x: x.split(','))
    df_t2['blue_players'] = df_t2['blue_players'].apply(lambda x: str(x).split(','))
    df_t2['red_players']  = df_t2['red_players'].apply(lambda x: str(x).split(','))
    if 'date' in df_t2.columns:
        df_t2['date'] = pd.to_datetime(df_t2['date'], errors='coerce')
        df_t2 = df_t2.sort_values('date').reset_index(drop=True)
    if 'year' not in df_t2.columns:
        df_t2['year'] = df_t2['date'].dt.year
    df_t2['date_str'] = df_t2['date'].dt.strftime('%Y-%m-%d')
    print(f"  Loaded {len(df_t2)} tier 2 games")

    # Build role-champ rates for T2
    rc_wins_t2={}; rc_games_t2={}
    for _, row in df_t2.iterrows():
        result = row['blue_win']
        for i, champ in enumerate(row['blue_picks']):
            key = (POSITIONS[i] if i < len(POSITIONS) else 'unknown', champ.strip())
            rc_games_t2[key] = rc_games_t2.get(key, 0) + 1
            rc_wins_t2[key]  = rc_wins_t2.get(key, 0) + result
        for i, champ in enumerate(row['red_picks']):
            key = (POSITIONS[i] if i < len(POSITIONS) else 'unknown', champ.strip())
            rc_games_t2[key] = rc_games_t2.get(key, 0) + 1
            rc_wins_t2[key]  = rc_wins_t2.get(key, 0) + (1-result)
    role_champ_rate_t2 = {k: rc_wins_t2[k]/rc_games_t2[k]
                          for k in rc_games_t2 if rc_games_t2[k] >= MIN_ROLE_GAMES}

    # Build win features for T2
    tw2={}; tg2={}; cw2={}; cg2={}; h2h2={}; tr2={}; pw2={}; pg2={}
    t2_win_rows = []
    for _, row in df_t2.iterrows():
        blue, red = row['blue_team'], row['red_team']
        bp = [c.strip() for c in row['blue_picks']]
        rp = [c.strip() for c in row['red_picks']]
        bpl = row['blue_players']; rpl = row['red_players']
        ds  = row['date_str']

        bg=tg2.get(blue,0); rg=tg2.get(red,0)
        bwr=tw2.get(blue,0)/bg if bg>0 else 0.5
        rwr=tw2.get(red,0)/rg  if rg>0 else 0.5
        bcw=sum(cw2.get(c,0)/cg2[c] if cg2.get(c,0)>0 else 0.5 for c in bp)/len(bp)
        rcw=sum(cw2.get(c,0)/cg2[c] if cg2.get(c,0)>0 else 0.5 for c in rp)/len(rp)
        mk=tuple(sorted([blue,red])); hr=h2h2.get(mk,{}); ht=sum(hr.values())
        h2h_r=cap_h2h(hr.get(blue,0)/ht) if ht>0 else 0.5
        bh=tr2.get(blue,[]); rh=tr2.get(red,[])
        b_form=weighted_form(bh,FORM_WINDOW); r_form=weighted_form(rh,FORM_WINDOW)
        b_rwr=sum(bh[-RECENT_WINDOW:])/len(bh[-RECENT_WINDOW:]) if bh else 0.5
        r_rwr=sum(rh[-RECENT_WINDOW:])/len(rh[-RECENT_WINDOW:]) if rh else 0.5
        bpc=[PC_WEIGHT*(pw2.get((pl.strip(),c.strip()),0)/pg2.get((pl.strip(),c.strip()),1)
             if pg2.get((pl.strip(),c.strip()),0)>=MIN_PC_GAMES else 0.5)
             +RC_WEIGHT*role_champ_rate_t2.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c.strip()),0.5)
             for i,(pl,c) in enumerate(zip(bpl,bp))]
        rpc=[PC_WEIGHT*(pw2.get((pl.strip(),c.strip()),0)/pg2.get((pl.strip(),c.strip()),1)
             if pg2.get((pl.strip(),c.strip()),0)>=MIN_PC_GAMES else 0.5)
             +RC_WEIGHT*role_champ_rate_t2.get((POSITIONS[i] if i<len(POSITIONS) else 'unknown',c.strip()),0.5)
             for i,(pl,c) in enumerate(zip(rpl,rp))]
        bpca=sum(bpc)/len(bpc); rpca=sum(rpc)/len(rpc)
        bfl=gold_lookup.get((ds,blue),{}); rfl=gold_lookup.get((ds,red),{})

        t2_win_rows.append({
            'blue_team_winrate':bwr,'red_team_winrate':rwr,'team_winrate_diff':bwr-rwr,
            'blue_team_games':bg,'red_team_games':rg,
            'blue_avg_winrate':bcw,'red_avg_winrate':rcw,'winrate_diff':bcw-rcw,
            'h2h_winrate':h2h_r,
            'blue_form':b_form,'red_form':r_form,'form_diff':b_form-r_form,
            'blue_recent_wr':b_rwr,'red_recent_wr':r_rwr,'recent_wr_diff':b_rwr-r_rwr,
            'blue_side_advantage':BLUE_SIDE_WINRATE,
            'blue_pc_avg':bpca,'red_pc_avg':rpca,'pc_avg_diff':bpca-rpca,
            'blue_avg_gd20':bfl.get('avg_gd20',0.0),'red_avg_gd20':rfl.get('avg_gd20',0.0),
            'gd20_diff':bfl.get('avg_gd20',0.0)-rfl.get('avg_gd20',0.0),
            'blue_late_scaling':bfl.get('late_scaling',0.0),'red_late_scaling':rfl.get('late_scaling',0.0),
            'late_scaling_diff':bfl.get('late_scaling',0.0)-rfl.get('late_scaling',0.0),
            'year':row['year'],
        })

        result=row['blue_win']
        tg2[blue]=tg2.get(blue,0)+1; tg2[red]=tg2.get(red,0)+1
        tw2[blue]=tw2.get(blue,0)+result; tw2[red]=tw2.get(red,0)+(1-result)
        for c in bp: cg2[c]=cg2.get(c,0)+1; cw2[c]=cw2.get(c,0)+result
        for c in rp: cg2[c]=cg2.get(c,0)+1; cw2[c]=cw2.get(c,0)+(1-result)
        if mk not in h2h2: h2h2[mk]={}
        h2h2[mk][blue]=h2h2[mk].get(blue,0)+result; h2h2[mk][red]=h2h2[mk].get(red,0)+(1-result)
        if blue not in tr2: tr2[blue]=[]
        if red  not in tr2: tr2[red]=[]
        tr2[blue].append(1 if result==1 else 0); tr2[red].append(0 if result==1 else 1)
        for pl,c in zip(bpl,bp):
            key=(pl.strip(),c.strip()); pg2[key]=pg2.get(key,0)+1; pw2[key]=pw2.get(key,0)+result
        for pl,c in zip(rpl,rp):
            key=(pl.strip(),c.strip()); pg2[key]=pg2.get(key,0)+1; pw2[key]=pw2.get(key,0)+(1-result)

    t2_feat = pd.DataFrame(t2_win_rows).reset_index(drop=True)
    t2_y    = df_t2['blue_win'].reset_index(drop=True)

    t2_mlb = MultiLabelBinarizer()
    t2_mlb.fit((df_t2['blue_picks']+df_t2['red_picks']).tolist())
    t2_blue_enc = pd.DataFrame(t2_mlb.transform(df_t2['blue_picks']),
        columns=['blue_'+c for c in t2_mlb.classes_]).reset_index(drop=True)
    t2_red_enc  = pd.DataFrame(t2_mlb.transform(df_t2['red_picks']),
        columns=['red_' +c for c in t2_mlb.classes_]).reset_index(drop=True)
    t2_feat_cols = [c for c in t2_feat.columns if c != 'year']
    t2_X = pd.concat([t2_blue_enc, t2_red_enc, t2_feat[t2_feat_cols]], axis=1)

    # Grid search for T2
    t2_val  = t2_feat['year'] == 2025
    t2_tr   = t2_feat['year'] <= 2024
    t2_train= t2_feat['year'] <= 2025

    t2_best_auc=0; t2_best_params={'n_estimators':125,'max_depth':2,'learning_rate':0.1}
    if t2_tr.sum()>50 and t2_val.sum()>20:
        print(f"  Grid search for T2 win model...")
        for params in [
            {'n_estimators':125,'max_depth':2,'learning_rate':0.1},
            {'n_estimators':200,'max_depth':2,'learning_rate':0.05},
            {'n_estimators':150,'max_depth':2,'learning_rate':0.08},
        ]:
            m=GradientBoostingClassifier(**params,random_state=42)
            m.fit(t2_X[t2_tr],t2_y[t2_tr])
            pr=m.predict_proba(t2_X[t2_val])[:,1]
            from sklearn.metrics import roc_auc_score as _auc2
            a=_auc2(t2_y[t2_val],pr)
            marker=' ← best' if a>t2_best_auc else ''
            print(f"    est={params['n_estimators']} depth={params['max_depth']} lr={params['learning_rate']} → AUC:{a:.4f}{marker}")
            if a>t2_best_auc: t2_best_auc=a; t2_best_params=params

    print(f"  Training T2 win model (best params: {t2_best_params})...")
    t2_win_base  = GradientBoostingClassifier(**t2_best_params, random_state=42)
    t2_win_model = CalibratedClassifierCV(t2_win_base, method='isotonic', cv=5)
    t2_win_model.fit(t2_X[t2_train], t2_y[t2_train])
    print(f"  ✅ T2 win model trained on {t2_train.sum()} games")

    # T2 team/champ lookups
    t2_all_teams = sorted(set(df_t2['blue_team'].tolist()+df_t2['red_team'].tolist()))
    t2_all_champs = sorted(set(
        [c for picks in df_t2['blue_picks'].tolist()+df_t2['red_picks'].tolist() for c in picks]))
    t2_team_recent = {t: tr2[t] for t in tr2}
    t2_team_games  = {t: tg2[t] for t in tg2}
    t2_team_wins_d = {t: tw2[t] for t in tw2}
    t2_h2h         = h2h2
    t2_pc_rate     = {k: pw2[k]/pg2[k] for k in pg2 if pg2[k]>=MIN_PC_GAMES}

    # Build T2 payload (win model only — FT5 optional)
    t2_payload = {
        'win_model':        t2_win_model,
        'win_mlb':          t2_mlb,
        'win_team_rate':    {t: tw2[t]/tg2[t] if tg2[t]>0 else 0.5 for t in tg2},
        'win_team_games':   tg2,
        'win_champ_rate':   {c: cw2[c]/cg2[c] for c in cg2 if cg2[c]>0},
        'win_h2h':          h2h2,
        'win_team_recent':  tr2,
        'pc_rate':          t2_pc_rate,
        'pc_games':         pg2,
        'role_champ_rate':  role_champ_rate_t2,
        'pc_weight':        PC_WEIGHT,
        'rc_weight':        RC_WEIGHT,
        'h2h_cap':          H2H_CAP,
        'min_pc_games':     MIN_PC_GAMES,
        'form_window':      FORM_WINDOW,
        'recent_window':    RECENT_WINDOW,
        'recent_weight':    RECENT_WEIGHT,
        'gold_window':      GOLD_WINDOW,
        'gold_lookup':      gold_lookup,
        'team_avg_gamelength': team_avg_gamelength,
        'team_avg_kills':      team_avg_kills,
        'all_teams':        t2_all_teams,
        'all_champs':       t2_all_champs,
        'team_aliases':     TEAM_ALIASES,
        'tier':             'T2',
        'leagues':          ['LCKC','LFL','EM','PRM'],
    }

    with open('model_payload_t2.pkl', 'wb') as f:
        pickle.dump(t2_payload, f)
    size_t2 = os.path.getsize('model_payload_t2.pkl') / (1024*1024)
    print(f"\n✅ Saved model_payload_t2.pkl ({size_t2:.1f} MB)")
    print(f"   Teams: {len(t2_all_teams)} | Champions: {len(t2_all_champs)}")
    print(f"   Win GBM: {t2_best_params} (grid searched)")
