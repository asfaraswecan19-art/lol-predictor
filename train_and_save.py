import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

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

for col in ['golddiffat10', 'golddiffat20']:
    team_raw[col] = pd.to_numeric(team_raw[col], errors='coerce').fillna(0)

gd10_hist = defaultdict(list)
gd20_hist = defaultdict(list)
gold_lookup = {}  # (date_str, teamname) -> {avg_gd20, late_scaling}

for _, row in team_raw.iterrows():
    team     = row['teamname']
    date_str = str(row['date'])[:10]
    w        = GOLD_WINDOW

    h10 = gd10_hist[team]
    h20 = gd20_hist[team]
    avg_gd20    = sum(h20[-w:]) / len(h20[-w:]) if h20 else 0.0
    avg_gd10    = sum(h10[-w:]) / len(h10[-w:]) if h10 else 0.0
    late_scaling = avg_gd20 - avg_gd10

    gold_lookup[(date_str, team)] = {
        'avg_gd20':     avg_gd20,
        'late_scaling': late_scaling,
    }

    gd10_hist[team].append(float(row['golddiffat10']))
    gd20_hist[team].append(float(row['golddiffat20']))

print(f"  Gold lookup entries: {len(gold_lookup)}")
covered = sum(1 for k in gold_lookup if gold_lookup[k]['avg_gd20'] != 0.0)
print(f"  Non-zero avg_gd20:   {covered} / {len(gold_lookup)}")

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

# =================================================================
# LEAK-FREE FEATURE BUILD (V8.1)
# -----------------------------------------------------------------
# Every per-game feature is computed from ONLY prior games (state is
# updated AFTER the feature row is built). This is the key fix vs the
# previous version, which computed team/champ/H2H rates over the whole
# dataset first, leaking future results into training features.
#
# The final accumulated lookups (win_team_rate, win_champ_rate, win_h2h,
# pc_rate, role_champ_rate) are saved to the payload for INFERENCE — that
# is correct, because at prediction time we legitimately know all history.
# Only the TRAINING features must be leak-free, and they now are.
# =================================================================
print("  Building leak-free win features (chronological)...")
win_df = win_df.sort_values('date').reset_index(drop=True)
win_df['date_str'] = win_df['date'].astype(str).str[:10]

def get_gold_feat(date_str, team, feat):
    return gold_lookup.get((date_str, team), {}).get(feat, 0.0)

# Running state (updated after each game)
role_champ_wins  = {}; role_champ_games = {}
pc_wins          = {}; pc_games         = {}
win_team_wins    = {}; win_team_games   = {}
win_champ_wins   = {}; win_champ_games  = {}
win_h2h          = {}
win_team_recent  = {}

def cur_rate(wins, games, key, default=0.5, min_games=1):
    g = games.get(key, 0)
    return wins.get(key, 0) / g if g >= min_games else default

win_rows = []
for _, row in win_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    bp = [c.strip() for c in row['blue_picks']]
    rp = [c.strip() for c in row['red_picks']]
    bpl = [p.strip() for p in row['blue_players']]
    rpl = [p.strip() for p in row['red_players']]
    ds  = row['date_str']
    result = row['blue_win']

    # --- Team winrate (prior games only) ---
    b_tg = win_team_games.get(blue, 0); r_tg = win_team_games.get(red, 0)
    b_twr = win_team_wins.get(blue, 0) / b_tg if b_tg > 0 else 0.5
    r_twr = win_team_wins.get(red,  0) / r_tg if r_tg > 0 else 0.5

    # --- Champion winrate (prior games only) ---
    b_cwr = sum(cur_rate(win_champ_wins, win_champ_games, c) for c in bp) / len(bp)
    r_cwr = sum(cur_rate(win_champ_wins, win_champ_games, c) for c in rp) / len(rp)

    # --- H2H (prior meetings only) ---
    mk = tuple(sorted([blue, red]))
    hr = win_h2h.get(mk, {})
    ht = sum(hr.values())
    h2h_rate = cap_h2h(hr.get(blue, 0) / ht) if ht > 0 else 0.5

    # --- Form & recent WR (already leak-free in original) ---
    b_hist = win_team_recent.get(blue, [])
    r_hist = win_team_recent.get(red,  [])
    b_form = weighted_form(b_hist, FORM_WINDOW)
    r_form = weighted_form(r_hist, FORM_WINDOW)
    b_rwr  = sum(b_hist[-RECENT_WINDOW:]) / len(b_hist[-RECENT_WINDOW:]) if b_hist else 0.5
    r_rwr  = sum(r_hist[-RECENT_WINDOW:]) / len(r_hist[-RECENT_WINDOW:]) if r_hist else 0.5

    # --- PC/RC blended (prior games only) ---
    def blended(players, picks):
        rates = []
        for i, (pl, c) in enumerate(zip(players, picks)):
            pos = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
            pc_v = cur_rate(pc_wins, pc_games, (pl, c), 0.5, MIN_PC_GAMES) if pl else 0.5
            rc_v = cur_rate(role_champ_wins, role_champ_games, (pos, c), 0.5, MIN_ROLE_GAMES)
            rates.append(PC_WEIGHT * pc_v + RC_WEIGHT * rc_v)
        return sum(rates) / len(rates) if rates else 0.5
    b_pca = blended(bpl, bp)
    r_pca = blended(rpl, rp)

    # --- Gold trajectory (already leak-free; gold_lookup is built rolling) ---
    b_gd20 = get_gold_feat(ds, blue, 'avg_gd20'); r_gd20 = get_gold_feat(ds, red, 'avg_gd20')
    b_late = get_gold_feat(ds, blue, 'late_scaling'); r_late = get_gold_feat(ds, red, 'late_scaling')

    win_rows.append({
        'blue_team_winrate':b_twr,'red_team_winrate':r_twr,'team_winrate_diff':b_twr-r_twr,
        'blue_team_games':b_tg,'red_team_games':r_tg,
        'blue_avg_winrate':b_cwr,'red_avg_winrate':r_cwr,'winrate_diff':b_cwr-r_cwr,
        'h2h_winrate':h2h_rate,
        'blue_form':b_form,'red_form':r_form,'form_diff':b_form-r_form,
        'blue_recent_wr':b_rwr,'red_recent_wr':r_rwr,'recent_wr_diff':b_rwr-r_rwr,
        'blue_side_advantage':BLUE_SIDE_WINRATE,
        'blue_pc_avg':b_pca,'red_pc_avg':r_pca,'pc_avg_diff':b_pca-r_pca,
        'blue_avg_gd20':b_gd20,'red_avg_gd20':r_gd20,'gd20_diff':b_gd20-r_gd20,
        'blue_late_scaling':b_late,'red_late_scaling':r_late,'late_scaling_diff':b_late-r_late,
    })

    # --- UPDATE STATE (after row built) ---
    win_team_games[blue] = b_tg + 1; win_team_games[red] = r_tg + 1
    win_team_wins[blue]  = win_team_wins.get(blue,0) + result
    win_team_wins[red]   = win_team_wins.get(red,0)  + (1 - result)
    for c in bp:
        win_champ_games[c] = win_champ_games.get(c,0)+1; win_champ_wins[c] = win_champ_wins.get(c,0)+result
    for c in rp:
        win_champ_games[c] = win_champ_games.get(c,0)+1; win_champ_wins[c] = win_champ_wins.get(c,0)+(1-result)
    if mk not in win_h2h: win_h2h[mk] = {}
    win_h2h[mk][blue] = win_h2h[mk].get(blue,0)+result
    win_h2h[mk][red]  = win_h2h[mk].get(red,0)+(1-result)
    win_team_recent.setdefault(blue,[]).append(1 if result==1 else 0)
    win_team_recent.setdefault(red,[]).append(0 if result==1 else 1)
    for i, c in enumerate(bp):
        pos = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        role_champ_games[(pos,c)] = role_champ_games.get((pos,c),0)+1
        role_champ_wins[(pos,c)]  = role_champ_wins.get((pos,c),0)+result
    for i, c in enumerate(rp):
        pos = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        role_champ_games[(pos,c)] = role_champ_games.get((pos,c),0)+1
        role_champ_wins[(pos,c)]  = role_champ_wins.get((pos,c),0)+(1-result)
    for pl, c in zip(bpl, bp):
        pc_games[(pl,c)] = pc_games.get((pl,c),0)+1; pc_wins[(pl,c)] = pc_wins.get((pl,c),0)+result
    for pl, c in zip(rpl, rp):
        pc_games[(pl,c)] = pc_games.get((pl,c),0)+1; pc_wins[(pl,c)] = pc_wins.get((pl,c),0)+(1-result)

# Final lookups for INFERENCE (full-data totals — correct at prediction time)
win_team_rate   = {t: win_team_wins[t]/win_team_games[t] for t in win_team_games}
win_champ_rate  = {c: win_champ_wins[c]/win_champ_games[c] for c in win_champ_games}
role_champ_rate = {k: role_champ_wins[k]/role_champ_games[k]
                   for k in role_champ_games if role_champ_games[k] >= MIN_ROLE_GAMES}
pc_rate         = {k: pc_wins[k]/pc_games[k] for k in pc_games if pc_games[k] >= MIN_PC_GAMES}
print(f"  Role-champ combos: {len(role_champ_rate)} | PC combos: {len(pc_rate)}")

covered = sum(1 for r in win_rows if r['blue_avg_gd20'] != 0.0)
print(f"  Gold features matched: {covered}/{len(win_rows)} games")

win_features = pd.DataFrame(win_rows).reset_index(drop=True)

win_mlb = MultiLabelBinarizer()
win_mlb.fit((win_df['blue_picks'] + win_df['red_picks']).tolist())
win_blue_enc = pd.DataFrame(win_mlb.transform(win_df['blue_picks']),
    columns=['blue_' + c for c in win_mlb.classes_]).reset_index(drop=True)
win_red_enc  = pd.DataFrame(win_mlb.transform(win_df['red_picks']),
    columns=['red_'  + c for c in win_mlb.classes_]).reset_index(drop=True)
win_extra = win_features[[
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

# Train on ALL data for the deployed model (no random hold-out — accuracy is
# measured separately via the proper time-based backtest). Calibration uses
# internal CV so it doesn't need a separate hold-out.
print("Training win model (leak-free features, full data)...")
win_base  = GradientBoostingClassifier(n_estimators=125, max_depth=2, learning_rate=0.1, random_state=42)
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

# Merge dates from win data so we can sort chronologically (kill_timelines has no date)
_date_map = dict(zip(win_df['game_id'], win_df['date'])) if 'game_id' in win_df.columns else {}
if _date_map and 'game_id' in ft5_df.columns:
    ft5_df['date'] = ft5_df['game_id'].map(_date_map)
    ft5_df = ft5_df.sort_values('date').reset_index(drop=True)
    print(f"  FT5 sorted chronologically via game_id→date merge")
else:
    print(f"  WARNING: could not merge dates into FT5 — order may not be chronological")

# -----------------------------------------------------------------
# LEAK-FREE FT5 FEATURE BUILD (V8.1)
# Same principle as win features: each game's features use only prior
# games; running state updated afterward. Final lookups saved for inference.
# -----------------------------------------------------------------
print("  Building leak-free FT5 features (chronological)...")
ft5_champ_wins  = {}; ft5_champ_games = {}
ft5_team_wins   = {}; ft5_team_games  = {}
ft5_avg_time    = {}; ft5_time_counts = {}
ft5_h2h         = {}
ft5_team_recent = {}

ft5_rows = []
for _, row in ft5_df.iterrows():
    blue, red = row['blue_team'], row['red_team']
    bp = [c.strip() for c in row['blue_picks']]
    rp = [c.strip() for c in row['red_picks']]
    result = row['first_to_five_binary']

    # Aggression (champ FT5 rate, prior games only)
    b_agg = sum(cur_rate(ft5_champ_wins, ft5_champ_games, c) for c in bp) / len(bp)
    r_agg = sum(cur_rate(ft5_champ_wins, ft5_champ_games, c) for c in rp) / len(rp)
    # Team early rate (prior games only)
    b_tg = ft5_team_games.get(blue,0); r_tg = ft5_team_games.get(red,0)
    b_early = ft5_team_wins.get(blue,0)/b_tg if b_tg>0 else 0.5
    r_early = ft5_team_wins.get(red,0)/r_tg  if r_tg>0 else 0.5
    # Kill speed (prior games only; default = league mean, set below after pass)
    b_spd = ft5_avg_time.get(blue,0)/ft5_time_counts[blue] if ft5_time_counts.get(blue,0)>0 else None
    r_spd = ft5_avg_time.get(red,0)/ft5_time_counts[red]   if ft5_time_counts.get(red,0)>0  else None
    # H2H (prior meetings only)
    mk = tuple(sorted([blue,red])); hr = ft5_h2h.get(mk,{}); ht = sum(hr.values())
    h2h_rate = cap_h2h(hr.get(blue,0)/ht) if ht>0 else 0.5
    # Form (prior games only)
    b_form = weighted_form(ft5_team_recent.get(blue,[]), FORM_WINDOW)
    r_form = weighted_form(ft5_team_recent.get(red,[]),  FORM_WINDOW)

    ft5_rows.append({
        'blue_aggression':b_agg,'red_aggression':r_agg,'aggression_diff':b_agg-r_agg,
        'blue_early_rate':b_early,'red_early_rate':r_early,'early_rate_diff':b_early-r_early,
        'blue_kill_speed':b_spd,'red_kill_speed':r_spd,  # None filled after pass
        'h2h_early_rate':h2h_rate,
        'blue_early_form':b_form,'red_early_form':r_form,'early_form_diff':b_form-r_form,
    })

    # --- UPDATE STATE ---
    for c in bp:
        ft5_champ_games[c] = ft5_champ_games.get(c,0)+1; ft5_champ_wins[c] = ft5_champ_wins.get(c,0)+result
    for c in rp:
        ft5_champ_games[c] = ft5_champ_games.get(c,0)+1; ft5_champ_wins[c] = ft5_champ_wins.get(c,0)+(1-result)
    ft5_team_games[blue] = b_tg+1; ft5_team_games[red] = r_tg+1
    ft5_team_wins[blue]  = ft5_team_wins.get(blue,0)+result
    ft5_team_wins[red]   = ft5_team_wins.get(red,0)+(1-result)
    if row['blue_time'] > 0:
        ft5_avg_time[blue] = ft5_avg_time.get(blue,0)+row['blue_time']; ft5_time_counts[blue] = ft5_time_counts.get(blue,0)+1
    if row['red_time'] > 0:
        ft5_avg_time[red]  = ft5_avg_time.get(red,0)+row['red_time'];   ft5_time_counts[red]  = ft5_time_counts.get(red,0)+1
    if mk not in ft5_h2h: ft5_h2h[mk] = {}
    ft5_h2h[mk][blue] = ft5_h2h[mk].get(blue,0)+result
    ft5_h2h[mk][red]  = ft5_h2h[mk].get(red,0)+(1-result)
    ft5_team_recent.setdefault(blue,[]).append(1 if result==1 else 0)
    ft5_team_recent.setdefault(red,[]).append(0 if result==1 else 1)

# Final lookups for inference
champ_aggression = {c: ft5_champ_wins[c]/ft5_champ_games[c] for c in ft5_champ_games}
team_early_rate  = {t: ft5_team_wins[t]/ft5_team_games[t] for t in ft5_team_games}
team_kill_speed  = {t: ft5_avg_time[t]/ft5_time_counts[t] for t in ft5_avg_time if ft5_time_counts[t] > 0}

# Kill-speed default = league mean (~22 min), NOT 10.0. Fill the None placeholders.
KILL_SPEED_DEFAULT = (sum(ft5_avg_time.values()) / sum(ft5_time_counts.values())
                      if ft5_time_counts else 22.0)
print(f"  Kill-speed default for unknown teams: {KILL_SPEED_DEFAULT:.2f} min")

ft5_features = pd.DataFrame(ft5_rows).reset_index(drop=True)
ft5_features['blue_kill_speed'] = ft5_features['blue_kill_speed'].fillna(KILL_SPEED_DEFAULT)
ft5_features['red_kill_speed']  = ft5_features['red_kill_speed'].fillna(KILL_SPEED_DEFAULT)
ft5_features['speed_diff']      = ft5_features['red_kill_speed'] - ft5_features['blue_kill_speed']

ft5_mlb = MultiLabelBinarizer()
ft5_mlb.fit((ft5_df['blue_picks'] + ft5_df['red_picks']).tolist())
ft5_blue_enc = pd.DataFrame(ft5_mlb.transform(ft5_df['blue_picks']),
    columns=['blue_' + c for c in ft5_mlb.classes_]).reset_index(drop=True)
ft5_red_enc  = pd.DataFrame(ft5_mlb.transform(ft5_df['red_picks']),
    columns=['red_'  + c for c in ft5_mlb.classes_]).reset_index(drop=True)
ft5_extra = ft5_features[[
    'blue_aggression',  'red_aggression',  'aggression_diff',
    'blue_early_rate',  'red_early_rate',  'early_rate_diff',
    'blue_kill_speed',  'red_kill_speed',  'speed_diff',
    'h2h_early_rate',
    'blue_early_form',  'red_early_form',  'early_form_diff',
]].reset_index(drop=True)

ft5_X = pd.concat([ft5_blue_enc, ft5_red_enc, ft5_extra], axis=1)
ft5_y = ft5_df['first_to_five_binary'].reset_index(drop=True)

# Train only on non-ambiguous games
ft5_train_mask = ft5_df['is_ambiguous'] == 0
print(f"  FT5 training on {ft5_train_mask.sum()} non-ambiguous games "
      f"(dropped {(~ft5_train_mask).sum()} ambiguous)")
ft5_X_train = ft5_X[ft5_train_mask]
ft5_y_train = ft5_y[ft5_train_mask]

print("Training FT5 model (leak-free features)...")
ft5_base  = GradientBoostingClassifier(n_estimators=125, max_depth=1, learning_rate=0.03, random_state=42)
ft5_model = CalibratedClassifierCV(ft5_base, method='isotonic', cv=5)
ft5_model.fit(ft5_X_train, ft5_y_train)
print("✅ FT5 model trained")

# =================================================================
# TOTAL KILLS & DURATION MODELS (merged from time_kills pipeline)
# -----------------------------------------------------------------
# Predicts total kills (point + 10/50/90 quantiles + O/U classifiers at
# common bet lines) and game duration (interval only — regression barely
# beats baseline, so the UI shows the interval, not a point estimate).
#
# Leak-free: all team/champ/patch stats built chronologically. Models train
# on ALL data for deployment (no test-set window tuning, unlike the earlier
# analysis script). Champion archetypes computed from full data.
# =================================================================
print("\n" + "="*55)
print("  Building TOTAL KILLS & DURATION models...")
print("="*55)
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠️  lightgbm not installed — skipping kills/duration models.")
    print("     Run: pip install lightgbm")

if HAS_LGB:
    KLS_RECENT_WINDOW = 20
    KILLS_LINES = [22.5, 24.5, 26.5, 28.5, 30.5]
    SHRINK_K = 30

    # Reuse win_df (already loaded, has duration + kills + sorted by date)
    kdf = win_df.copy()
    # Filter unrealistic games (remakes / early surrenders) — noise no model fits
    kdf = kdf.dropna(subset=['game_duration_min', 'total_kills'])
    kdf = kdf[kdf['game_duration_min'] >= 15].reset_index(drop=True)
    print(f"  Kills/duration training games: {len(kdf)}")

    gdm_k = kdf['game_duration_min'].mean()
    gkm_k = kdf['total_kills'].mean()

    # Champion archetypes (full data, Bayesian-shrunk toward global mean)
    cd_sum = defaultdict(float); cd_cnt = defaultdict(int)
    ck_sum = defaultdict(float); ck_cnt = defaultdict(int)
    for _, row in kdf.iterrows():
        for c in [x.strip() for x in row['blue_picks']] + [x.strip() for x in row['red_picks']]:
            cd_sum[c] += row['game_duration_min']; cd_cnt[c] += 1
            ck_sum[c] += row['total_kills'];      ck_cnt[c] += 1
    champ_dur_score   = {c: (cd_sum[c]+SHRINK_K*gdm_k)/(cd_cnt[c]+SHRINK_K) for c in cd_cnt}
    champ_kills_score = {c: (ck_sum[c]+SHRINK_K*gkm_k)/(ck_cnt[c]+SHRINK_K) for c in ck_cnt}

    def _karch(picks):
        durs = [champ_dur_score.get(c, gdm_k) for c in picks]
        klls = [champ_kills_score.get(c, gkm_k) for c in picks]
        return (sum(durs)/len(durs) if durs else gdm_k,
                sum(klls)/len(klls) if klls else gkm_k)

    # Patch release dates (for patch_age feature)
    kpatch_release = {}
    if not raw.empty:
        raw['patch_str'] = raw['patch'].astype(str)
        kpatch_release = raw.groupby('patch_str')['date'].min().to_dict()
        kpatch_release = {k: v for k, v in kpatch_release.items() if k != 'nan' and pd.notna(v)}

    def _kpatch_age(p, d):
        r = kpatch_release.get(str(p))
        return 0 if r is None or pd.isna(d) else max(0, (d - r).days)

    # Leak-free feature build (chronological)
    kt_dur = defaultdict(list); kt_kls = defaultdict(list); kt_ckpm = defaultdict(list)
    kh_dur = defaultdict(list); kh_kls = defaultdict(list)
    kp_dur_sum = defaultdict(float); kp_dur_cnt = defaultdict(int)
    kp_kls_sum = defaultdict(float); kp_kls_cnt = defaultdict(int)

    def _rm(h, w, d): return sum(h[-w:])/len(h[-w:]) if h else d
    def _rs(h, w, d): return float(__import__('numpy').std(h[-w:])) if len(h) >= 3 else d

    import numpy as _np
    krows = []
    for _, row in kdf.iterrows():
        blue, red = row['blue_team'], row['red_team']
        bp = [c.strip() for c in row['blue_picks']]
        rp = [c.strip() for c in row['red_picks']]
        league, patch, date = row['league'], row['patch'], row['date']

        b_dur=_rm(kt_dur[blue],KLS_RECENT_WINDOW,gdm_k); r_dur=_rm(kt_dur[red],KLS_RECENT_WINDOW,gdm_k)
        b_kls=_rm(kt_kls[blue],KLS_RECENT_WINDOW,gkm_k); r_kls=_rm(kt_kls[red],KLS_RECENT_WINDOW,gkm_k)
        b_ck=_rm(kt_ckpm[blue],KLS_RECENT_WINDOW,gkm_k/gdm_k); r_ck=_rm(kt_ckpm[red],KLS_RECENT_WINDOW,gkm_k/gdm_k)
        b_ds=_rs(kt_dur[blue],KLS_RECENT_WINDOW,5.0); r_ds=_rs(kt_dur[red],KLS_RECENT_WINDOW,5.0)
        b_ks=_rs(kt_kls[blue],KLS_RECENT_WINDOW,8.0); r_ks=_rs(kt_kls[red],KLS_RECENT_WINDOW,8.0)
        mk = tuple(sorted([blue, red]))
        h_dur=_rm(kh_dur[mk],10,(b_dur+r_dur)/2); h_kls=_rm(kh_kls[mk],10,(b_kls+r_kls)/2)
        b_ad,b_ak=_karch(bp); r_ad,r_ak=_karch(rp)
        if kp_dur_cnt[patch] >= 10:
            p_dur=kp_dur_sum[patch]/kp_dur_cnt[patch]; p_kls=kp_kls_sum[patch]/kp_kls_cnt[patch]
        else:
            p_dur,p_kls = gdm_k,gkm_k
        p_age = _kpatch_age(patch, date)

        krows.append({
            'b_dur_mean':b_dur,'r_dur_mean':r_dur,'avg_dur_mean':(b_dur+r_dur)/2,
            'b_kills_mean':b_kls,'r_kills_mean':r_kls,'avg_kills_mean':(b_kls+r_kls)/2,
            'b_ckpm_mean':b_ck,'r_ckpm_mean':r_ck,'avg_ckpm_mean':(b_ck+r_ck)/2,
            'b_dur_std':b_ds,'r_dur_std':r_ds,'b_kills_std':b_ks,'r_kills_std':r_ks,
            'min_games_seen':min(len(kt_dur[blue]),len(kt_dur[red])),
            'h2h_dur_mean':h_dur,'h2h_kills_mean':h_kls,'h2h_n':len(kh_dur[mk]),
            'b_arch_dur':b_ad,'r_arch_dur':r_ad,'avg_arch_dur':(b_ad+r_ad)/2,
            'b_arch_kills':b_ak,'r_arch_kills':r_ak,'avg_arch_kills':(b_ak+r_ak)/2,
            'patch_dur':p_dur,'patch_kills':p_kls,
            'patch_age_days':p_age,'is_new_patch':1 if p_age<7 else 0,
            'league_idx':hash(league)%1000,
        })

        dur=row['game_duration_min']; k=row['total_kills']
        kt_dur[blue].append(dur); kt_dur[red].append(dur)
        kt_kls[blue].append(k); kt_kls[red].append(k)
        kt_ckpm[blue].append(k/dur); kt_ckpm[red].append(k/dur)
        kh_dur[mk].append(dur); kh_kls[mk].append(k)
        kp_dur_sum[patch]+=dur; kp_dur_cnt[patch]+=1
        kp_kls_sum[patch]+=k; kp_kls_cnt[patch]+=1

    kfeat = pd.DataFrame(krows)
    kls_feat_cols = list(kfeat.columns)
    kX = kfeat[kls_feat_cols]
    ky_dur = kdf['game_duration_min'].reset_index(drop=True)
    ky_kls = kdf['total_kills'].reset_index(drop=True)

    def _lgbreg(objective='regression_l1', alpha=None):
        params = dict(n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=15,
                      min_child_samples=30, reg_alpha=0.1, reg_lambda=0.1,
                      random_state=42, verbose=-1)
        if alpha is not None:
            params['objective']='quantile'; params['alpha']=alpha
        else:
            params['objective']=objective
        return lgb.LGBMRegressor(**params)

    print("  Training duration models (point + 10/50/90 quantiles)...")
    dur_point=_lgbreg('regression_l1'); dur_point.fit(kX, ky_dur)
    dur_q10=_lgbreg(alpha=0.10); dur_q10.fit(kX, ky_dur)
    dur_q50=_lgbreg(alpha=0.50); dur_q50.fit(kX, ky_dur)
    dur_q90=_lgbreg(alpha=0.90); dur_q90.fit(kX, ky_dur)

    print("  Training kills models (point + 10/50/90 quantiles)...")
    kls_point=_lgbreg('regression_l1'); kls_point.fit(kX, ky_kls)
    kls_q10=_lgbreg(alpha=0.10); kls_q10.fit(kX, ky_kls)
    kls_q50=_lgbreg(alpha=0.50); kls_q50.fit(kX, ky_kls)
    kls_q90=_lgbreg(alpha=0.90); kls_q90.fit(kX, ky_kls)

    print("  Training kills O/U classifiers at bet lines...")
    kls_ou_models = {}
    for line in KILLS_LINES:
        yb = (ky_kls > line).astype(int)
        if yb.mean() < 0.05 or yb.mean() > 0.95:
            continue
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=8,
                               min_child_samples=30, reg_alpha=0.2, reg_lambda=0.2,
                               random_state=42, verbose=-1)
        m.fit(kX, yb)
        kls_ou_models[line] = m
    print(f"  ✅ Kills/duration models trained ({len(kls_ou_models)} O/U lines)")

    # Final lookups for inference (full-data histories)
    kls_team_dur_hist   = {t: list(v) for t, v in kt_dur.items()}
    kls_team_kills_hist = {t: list(v) for t, v in kt_kls.items()}
    kls_team_ckpm_hist  = {t: list(v) for t, v in kt_ckpm.items()}
    kls_h2h_dur   = {k: list(v) for k, v in kh_dur.items()}
    kls_h2h_kills = {k: list(v) for k, v in kh_kls.items()}
    kls_patch_dur   = {k: kp_dur_sum[k]/kp_dur_cnt[k] for k in kp_dur_cnt if kp_dur_cnt[k] >= 10}
    kls_patch_kills = {k: kp_kls_sum[k]/kp_kls_cnt[k] for k in kp_kls_cnt if kp_kls_cnt[k] >= 10}

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
    'kill_speed_default': KILL_SPEED_DEFAULT,
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

# Add kills/duration models if they were trained
if HAS_LGB:
    payload.update({
        'kls_point': kls_point, 'kls_q10': kls_q10, 'kls_q50': kls_q50, 'kls_q90': kls_q90,
        'kls_ou_models': kls_ou_models, 'kls_feat_cols': kls_feat_cols,
        'champ_dur_score': champ_dur_score, 'champ_kills_score': champ_kills_score,
        'kls_global_dur': gdm_k, 'kls_global_kills': gkm_k,
        'kls_team_dur_hist': kls_team_dur_hist, 'kls_team_kills_hist': kls_team_kills_hist,
        'kls_team_ckpm_hist': kls_team_ckpm_hist,
        'kls_h2h_dur': kls_h2h_dur, 'kls_h2h_kills': kls_h2h_kills,
        'kls_patch_dur': kls_patch_dur, 'kls_patch_kills': kls_patch_kills,
        'kls_recent_window': KLS_RECENT_WINDOW, 'kls_lines': KILLS_LINES,
        'dur_point': dur_point, 'dur_q10': dur_q10, 'dur_q50': dur_q50, 'dur_q90': dur_q90,
    })

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
print(f"   Win GBM:  n_estimators=125, max_depth=2, lr=0.1")
print(f"   FT5 GBM:  n_estimators=125, max_depth=1, lr=0.03")
print(f"   Gold window: {GOLD_WINDOW} games (avg_gd20 + late_scaling)")
