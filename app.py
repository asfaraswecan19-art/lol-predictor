import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# =================================================================
# PAGE CONFIG
# =================================================================
st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 LoL Pro Match Predictor")
st.caption("Win + First to Five | ~60% true accuracy | Calibrated probabilities")

# =================================================================
# CONSTANTS
# =================================================================
WIN_DATA = 'proplay_matches.csv'
FT5_DATA = 'kill_timelines.csv'
FORM_WINDOW = 5
BLUE_SIDE_WINRATE = 0.5312

# =================================================================
# MODEL CONFIDENCE CALCULATOR
# =================================================================
def model_confidence(b_games, r_games, h2h_total,
                     form_diff, winrate_diff, champ_diff):
    score = 0
    reasons = []
    warnings_list = []

    min_games = min(b_games, r_games)
    if min_games >= 50:
        score += 3
        reasons.append(f"Strong team history ({min_games}+ games each)")
    elif min_games >= 20:
        score += 2
        reasons.append(f"Decent team history ({min_games}+ games each)")
    elif min_games >= 10:
        score += 1
        reasons.append(f"Limited team history ({min_games} games)")
    else:
        warnings_list.append(f"Very little team data ({min_games} games)")

    if h2h_total >= 10:
        score += 3
        reasons.append(f"Strong H2H record ({h2h_total} games)")
    elif h2h_total >= 5:
        score += 2
        reasons.append(f"Some H2H history ({h2h_total} games)")
    elif h2h_total >= 2:
        score += 1
        reasons.append(f"Limited H2H ({h2h_total} games)")
    else:
        warnings_list.append("No H2H history — using defaults")

    signals = [
        1 if winrate_diff > 0.05 else (-1 if winrate_diff < -0.05 else 0),
        1 if form_diff > 0.1 else (-1 if form_diff < -0.1 else 0),
        1 if champ_diff > 0.02 else (-1 if champ_diff < -0.02 else 0),
    ]
    non_zero = [s for s in signals if s != 0]
    if len(non_zero) >= 2:
        if all(s == non_zero[0] for s in non_zero):
            score += 3
            reasons.append("All signals agree")
        else:
            score += 1
            warnings_list.append("Mixed signals — features conflict")
    else:
        score += 1
        warnings_list.append("Weak signal strength")

    if score >= 7:
        level = "🟢 HIGH"
        desc  = "Strong data, clear favourite"
    elif score >= 4:
        level = "🟡 MEDIUM"
        desc  = "Reasonable data, some uncertainty"
    else:
        level = "🔴 LOW"
        desc  = "Limited data or conflicting signals"

    return level, desc, reasons, warnings_list

# =================================================================
# LOAD AND TRAIN
# =================================================================
@st.cache_resource
def load_and_train():
    # WIN MODEL
    win_df = pd.read_csv(WIN_DATA)
    win_df['blue_picks'] = win_df['blue_picks'].apply(lambda x: x.split(','))
    win_df['red_picks']  = win_df['red_picks'].apply(lambda x: x.split(','))

    win_team_wins  = {}
    win_team_games = {}
    for _, row in win_df.iterrows():
        blue, red = row['blue_team'], row['red_team']
        win_team_games[blue] = win_team_games.get(blue, 0) + 1
        win_team_games[red]  = win_team_games.get(red,  0) + 1
        win_team_wins[blue]  = win_team_wins.get(blue, 0) + row['blue_win']
        win_team_wins[red]   = win_team_wins.get(red,  0) + (1 - row['blue_win'])
    win_team_rate = {t: win_team_wins[t] / win_team_games[t] for t in win_team_games}

    win_df['blue_team_winrate'] = win_df['blue_team'].map(win_team_rate)
    win_df['red_team_winrate']  = win_df['red_team'].map(win_team_rate)
    win_df['team_winrate_diff'] = win_df['blue_team_winrate'] - win_df['red_team_winrate']
    win_df['blue_team_games']   = win_df['blue_team'].map(win_team_games)
    win_df['red_team_games']    = win_df['red_team'].map(win_team_games)

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

    win_df['blue_avg_winrate'] = win_df['blue_picks'].apply(
        lambda picks: sum(win_champ_rate.get(c, 0.5) for c in picks) / len(picks))
    win_df['red_avg_winrate'] = win_df['red_picks'].apply(
        lambda picks: sum(win_champ_rate.get(c, 0.5) for c in picks) / len(picks))
    win_df['winrate_diff'] = win_df['blue_avg_winrate'] - win_df['red_avg_winrate']

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
        b_form = sum(win_team_recent[blue][-FORM_WINDOW:]) / len(
            win_team_recent[blue][-FORM_WINDOW:]) if win_team_recent[blue] else 0.5
        r_form = sum(win_team_recent[red][-FORM_WINDOW:]) / len(
            win_team_recent[red][-FORM_WINDOW:]) if win_team_recent[red] else 0.5
        win_df_sorted.at[idx, 'blue_form'] = b_form
        win_df_sorted.at[idx, 'red_form']  = r_form
        win_team_recent[blue].append(1 if row['blue_win'] == 1 else 0)
        win_team_recent[red].append(0  if row['blue_win'] == 1 else 1)
    win_df = win_df_sorted
    win_df['form_diff']           = win_df['blue_form'] - win_df['red_form']
    win_df['blue_side_advantage'] = BLUE_SIDE_WINRATE

    win_df['h2h_winrate'] = win_df.apply(
        lambda row: (win_h2h.get(tuple(sorted([row['blue_team'], row['red_team']])), {})
                     .get(row['blue_team'], 0)) /
                    max(sum(win_h2h.get(tuple(sorted([row['blue_team'],
                        row['red_team']])), {}).values()), 1), axis=1)

    win_mlb = MultiLabelBinarizer()
    win_mlb.fit(win_df['blue_picks'] + win_df['red_picks'])
    win_blue_enc = pd.DataFrame(win_mlb.transform(win_df['blue_picks']),
        columns=['blue_' + c for c in win_mlb.classes_]).reset_index(drop=True)
    win_red_enc = pd.DataFrame(win_mlb.transform(win_df['red_picks']),
        columns=['red_' + c for c in win_mlb.classes_]).reset_index(drop=True)
    win_extra = win_df[[
        'blue_team_winrate', 'red_team_winrate', 'team_winrate_diff',
        'blue_team_games', 'red_team_games',
        'blue_avg_winrate', 'red_avg_winrate', 'winrate_diff',
        'h2h_winrate', 'blue_form', 'red_form', 'form_diff',
        'blue_side_advantage',
    ]].reset_index(drop=True)

    win_X = pd.concat([win_blue_enc, win_red_enc, win_extra], axis=1)
    win_y = win_df['blue_win'].reset_index(drop=True)
    win_X_train, _, win_y_train, _ = train_test_split(
        win_X, win_y, test_size=0.2, random_state=42)
    win_base  = GradientBoostingClassifier(n_estimators=200, random_state=42)
    win_model = CalibratedClassifierCV(win_base, method='isotonic', cv=5)
    win_model.fit(win_X_train, win_y_train)

    # FT5 MODEL
    ft5_df = pd.read_csv(FT5_DATA)
    ft5_df = ft5_df[~ft5_df['tournament'].isin(['LPL'])].copy().reset_index(drop=True)
    ft5_df['blue_picks'] = ft5_df['blue_picks'].apply(lambda x: x.split(','))
    ft5_df['red_picks']  = ft5_df['red_picks'].apply(lambda x: x.split(','))
    ft5_df['first_to_five_binary'] = ft5_df['first_to_five'].apply(
        lambda x: 1 if x == 'blue' else 0)

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

    ft5_df['blue_aggression'] = ft5_df['blue_picks'].apply(
        lambda picks: sum(champ_aggression.get(c, 0.5) for c in picks) / len(picks))
    ft5_df['red_aggression'] = ft5_df['red_picks'].apply(
        lambda picks: sum(champ_aggression.get(c, 0.5) for c in picks) / len(picks))
    ft5_df['aggression_diff'] = ft5_df['blue_aggression'] - ft5_df['red_aggression']

    ft5_team_wins  = {}
    ft5_team_games = {}
    for _, row in ft5_df.iterrows():
        blue, red = row['blue_team'], row['red_team']
        ft5_team_games[blue] = ft5_team_games.get(blue, 0) + 1
        ft5_team_games[red]  = ft5_team_games.get(red,  0) + 1
        ft5_team_wins[blue]  = ft5_team_wins.get(blue, 0) + row['first_to_five_binary']
        ft5_team_wins[red]   = ft5_team_wins.get(red,  0) + (1 - row['first_to_five_binary'])
    team_early_rate = {t: ft5_team_wins[t] / ft5_team_games[t] for t in ft5_team_games}

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
        b_form = sum(ft5_team_recent[blue][-FORM_WINDOW:]) / len(
            ft5_team_recent[blue][-FORM_WINDOW:]) if ft5_team_recent[blue] else 0.5
        r_form = sum(ft5_team_recent[red][-FORM_WINDOW:]) / len(
            ft5_team_recent[red][-FORM_WINDOW:]) if ft5_team_recent[red] else 0.5
        ft5_df_sorted.at[idx, 'blue_early_form'] = b_form
        ft5_df_sorted.at[idx, 'red_early_form']  = r_form
        ft5_team_recent[blue].append(1 if row['first_to_five_binary'] == 1 else 0)
        ft5_team_recent[red].append(0  if row['first_to_five_binary'] == 1 else 1)
    ft5_df = ft5_df_sorted
    ft5_df['early_form_diff'] = ft5_df['blue_early_form'] - ft5_df['red_early_form']

    ft5_df['h2h_early_rate'] = ft5_df.apply(
        lambda row: (ft5_h2h.get(tuple(sorted([row['blue_team'], row['red_team']])), {})
                     .get(row['blue_team'], 0)) /
                    max(sum(ft5_h2h.get(tuple(sorted([row['blue_team'],
                        row['red_team']])), {}).values()), 1), axis=1)

    ft5_mlb = MultiLabelBinarizer()
    ft5_mlb.fit(ft5_df['blue_picks'] + ft5_df['red_picks'])
    ft5_blue_enc = pd.DataFrame(ft5_mlb.transform(ft5_df['blue_picks']),
        columns=['blue_' + c for c in ft5_mlb.classes_]).reset_index(drop=True)
    ft5_red_enc = pd.DataFrame(ft5_mlb.transform(ft5_df['red_picks']),
        columns=['red_' + c for c in ft5_mlb.classes_]).reset_index(drop=True)
    ft5_extra = ft5_df[[
        'blue_aggression', 'red_aggression', 'aggression_diff',
        'blue_early_rate', 'red_early_rate', 'early_rate_diff',
        'blue_kill_speed', 'red_kill_speed', 'speed_diff',
        'h2h_early_rate', 'blue_early_form', 'red_early_form', 'early_form_diff',
    ]].reset_index(drop=True)

    ft5_X = pd.concat([ft5_blue_enc, ft5_red_enc, ft5_extra], axis=1)
    ft5_y = ft5_df['first_to_five_binary'].reset_index(drop=True)
    ft5_X_train, _, ft5_y_train, _ = train_test_split(
        ft5_X, ft5_y, test_size=0.2, random_state=42)
    ft5_base  = GradientBoostingClassifier(n_estimators=200, random_state=42)
    ft5_model = CalibratedClassifierCV(ft5_base, method='isotonic', cv=5)
    ft5_model.fit(ft5_X_train, ft5_y_train)

    all_teams  = sorted(set(win_df['blue_team'].tolist() + win_df['red_team'].tolist()))
    all_champs = sorted(set(
        [c for picks in win_df['blue_picks'] for c in picks] +
        [c for picks in win_df['red_picks']  for c in picks]))

    return (win_model, win_mlb, win_team_rate, win_team_games, win_champ_rate,
            win_h2h, win_team_recent,
            ft5_model, ft5_mlb, champ_aggression, team_early_rate,
            team_kill_speed, ft5_h2h, ft5_team_recent,
            ft5_team_games,
            all_teams, all_champs)

# =================================================================
# LOAD MODELS
# =================================================================
with st.spinner("Training models... (~30 seconds on first load)"):
    (win_model, win_mlb, win_team_rate, win_team_games, win_champ_rate,
     win_h2h, win_team_recent,
     ft5_model, ft5_mlb, champ_aggression, team_early_rate,
     team_kill_speed, ft5_h2h, ft5_team_recent,
     ft5_team_games,
     all_teams, all_champs) = load_and_train()

st.success("Models ready!")

# =================================================================
# HELPERS
# =================================================================
def get_h2h_rate(h2h_dict, blue, red):
    matchup = tuple(sorted([blue, red]))
    if matchup not in h2h_dict: return 0.5
    total = sum(h2h_dict[matchup].values())
    return h2h_dict[matchup].get(blue, 0) / total if total > 0 else 0.5

def get_h2h_record(h2h_dict, blue, red):
    matchup = tuple(sorted([blue, red]))
    record  = h2h_dict.get(matchup, {})
    return int(record.get(blue, 0)), int(record.get(red, 0))

def get_h2h_total(h2h_dict, blue, red):
    matchup = tuple(sorted([blue, red]))
    return sum(h2h_dict.get(matchup, {}).values())

def get_form(recent_dict, team):
    recent = recent_dict.get(team, [0.5])
    return sum(recent[-FORM_WINDOW:]) / len(recent[-FORM_WINDOW:])

def odds_label(odds):
    if odds < 1.60:
        return "⚠️ Low odds — lower payout"
    elif odds >= 2.30:
        return "🔥 Great odds — best ROI range"
    else:
        return "✅ Good odds"

def calc_edge(conf, odds):
    implied = 1 / odds
    edge = conf - implied
    if edge < 0.08:
        units, label = 0, "⛔ SKIP"
    elif edge < 0.12:
        units, label = 1, "✅ DECENT"
    elif edge < 0.18:
        units, label = 2, "✅ STRONG"
    else:
        units, label = 3, "🔥 VERY STRONG"
    return edge, units, label, implied

# =================================================================
# INPUT
# =================================================================
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 Blue Side")
    blue_team = st.selectbox("Team", options=all_teams, key="blue_team", index=0)
    blue_top  = st.selectbox("Top",     options=all_champs, key="bt", index=0)
    blue_jg   = st.selectbox("Jungle",  options=all_champs, key="bj", index=0)
    blue_mid  = st.selectbox("Mid",     options=all_champs, key="bm", index=0)
    blue_adc  = st.selectbox("ADC",     options=all_champs, key="ba", index=0)
    blue_sup  = st.selectbox("Support", options=all_champs, key="bs", index=0)

with col2:
    st.markdown("### 🔴 Red Side")
    red_team = st.selectbox("Team", options=all_teams, key="red_team", index=1)
    red_top  = st.selectbox("Top",     options=all_champs, key="rt", index=0)
    red_jg   = st.selectbox("Jungle",  options=all_champs, key="rj", index=0)
    red_mid  = st.selectbox("Mid",     options=all_champs, key="rm", index=0)
    red_adc  = st.selectbox("ADC",     options=all_champs, key="ra", index=0)
    red_sup  = st.selectbox("Support", options=all_champs, key="rs", index=0)

st.markdown("### 📊 Odds")
col3, col4 = st.columns(2)
with col3:
    st.markdown("**Match Winner**")
    win_blue_odds = st.number_input("Blue odds", min_value=1.01, max_value=10.0,
                                     value=1.85, step=0.05, key="wbo")
    win_red_odds  = st.number_input("Red odds",  min_value=1.01, max_value=10.0,
                                     value=1.95, step=0.05, key="wro")
with col4:
    st.markdown("**First to Five**")
    ft5_blue_odds = st.number_input("Blue odds", min_value=1.01, max_value=10.0,
                                     value=1.85, step=0.05, key="fbo")
    ft5_red_odds  = st.number_input("Red odds",  min_value=1.01, max_value=10.0,
                                     value=1.95, step=0.05, key="fro")

predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

# =================================================================
# PREDICTION
# =================================================================
if predict_btn:
    blue = [blue_top, blue_jg, blue_mid, blue_adc, blue_sup]
    red  = [red_top,  red_jg,  red_mid,  red_adc,  red_sup]

    if blue_team == red_team:
        st.error("Blue and red team can't be the same!")
    elif len(set(blue + red)) < 10:
        st.error("Each champion must be unique across both teams!")
    else:
        if blue_team not in win_team_rate:
            st.warning(f"{blue_team} not in dataset — using defaults")
        if red_team not in win_team_rate:
            st.warning(f"{red_team} not in dataset — using defaults")

        # Win model
        b_win_enc = pd.DataFrame(win_mlb.transform([blue]),
            columns=['blue_' + c for c in win_mlb.classes_])
        r_win_enc = pd.DataFrame(win_mlb.transform([red]),
            columns=['red_'  + c for c in win_mlb.classes_])

        b_wr       = win_team_rate.get(blue_team, 0.5)
        r_wr       = win_team_rate.get(red_team,  0.5)
        b_games    = win_team_games.get(blue_team, 0)
        r_games    = win_team_games.get(red_team,  0)
        b_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in blue) / len(blue)
        r_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in red)  / len(red)
        win_h2h_r  = get_h2h_rate(win_h2h, blue_team, red_team)
        b_form     = get_form(win_team_recent, blue_team)
        r_form     = get_form(win_team_recent, red_team)
        h2h_total  = get_h2h_total(win_h2h, blue_team, red_team)

        win_extra_row = pd.DataFrame([[
            b_wr, r_wr, b_wr - r_wr,
            b_games, r_games,
            b_champ_wr, r_champ_wr, b_champ_wr - r_champ_wr,
            win_h2h_r, b_form, r_form, b_form - r_form, BLUE_SIDE_WINRATE,
        ]], columns=[
            'blue_team_winrate', 'red_team_winrate', 'team_winrate_diff',
            'blue_team_games', 'red_team_games',
            'blue_avg_winrate', 'red_avg_winrate', 'winrate_diff',
            'h2h_winrate', 'blue_form', 'red_form', 'form_diff',
            'blue_side_advantage',
        ])

        win_row  = pd.concat([b_win_enc, r_win_enc, win_extra_row], axis=1)
        win_prob = win_model.predict_proba(win_row)[0]
        blue_win_conf = win_prob[1]
        red_win_conf  = win_prob[0]

        # FT5 model
        b_ft5_enc = pd.DataFrame(ft5_mlb.transform([blue]),
            columns=['blue_' + c for c in ft5_mlb.classes_])
        r_ft5_enc = pd.DataFrame(ft5_mlb.transform([red]),
            columns=['red_'  + c for c in ft5_mlb.classes_])

        b_agg        = sum(champ_aggression.get(c, 0.5) for c in blue) / len(blue)
        r_agg        = sum(champ_aggression.get(c, 0.5) for c in red)  / len(red)
        b_early      = team_early_rate.get(blue_team, 0.5)
        r_early      = team_early_rate.get(red_team,  0.5)
        b_speed      = team_kill_speed.get(blue_team, 10.0)
        r_speed      = team_kill_speed.get(red_team,  10.0)
        ft5_h2h_r    = get_h2h_rate(ft5_h2h, blue_team, red_team)
        ft5_h2h_tot  = get_h2h_total(ft5_h2h, blue_team, red_team)
        b_early_form = get_form(ft5_team_recent, blue_team)
        r_early_form = get_form(ft5_team_recent, red_team)

        ft5_extra_row = pd.DataFrame([[
            b_agg, r_agg, b_agg - r_agg,
            b_early, r_early, b_early - r_early,
            b_speed, r_speed, r_speed - b_speed,
            ft5_h2h_r, b_early_form, r_early_form, b_early_form - r_early_form
        ]], columns=[
            'blue_aggression', 'red_aggression', 'aggression_diff',
            'blue_early_rate', 'red_early_rate', 'early_rate_diff',
            'blue_kill_speed', 'red_kill_speed', 'speed_diff',
            'h2h_early_rate', 'blue_early_form', 'red_early_form', 'early_form_diff'
        ])

        ft5_row  = pd.concat([b_ft5_enc, r_ft5_enc, ft5_extra_row], axis=1)
        ft5_prob = ft5_model.predict_proba(ft5_row)[0]
        blue_ft5_conf = ft5_prob[1]
        red_ft5_conf  = ft5_prob[0]

        # Confidence levels
        win_conf_level, win_conf_desc, win_reasons, win_warnings = model_confidence(
            b_games, r_games, h2h_total,
            b_form - r_form, b_wr - r_wr, b_champ_wr - r_champ_wr)

        ft5_conf_level, ft5_conf_desc, ft5_reasons, ft5_warnings = model_confidence(
            ft5_team_games.get(blue_team, 0),
            ft5_team_games.get(red_team,  0),
            ft5_h2h_tot,
            b_early_form - r_early_form,
            b_early - r_early,
            b_agg - r_agg)

        # Edges
        win_blue_edge, win_blue_units, win_blue_label, win_blue_impl = calc_edge(blue_win_conf, win_blue_odds)
        win_red_edge,  win_red_units,  win_red_label,  win_red_impl  = calc_edge(red_win_conf,  win_red_odds)
        ft5_blue_edge, ft5_blue_units, ft5_blue_label, ft5_blue_impl = calc_edge(blue_ft5_conf, ft5_blue_odds)
        ft5_red_edge,  ft5_red_units,  ft5_red_label,  ft5_red_impl  = calc_edge(red_ft5_conf,  ft5_red_odds)

        b_win_h2h, r_win_h2h = get_h2h_record(win_h2h, blue_team, red_team)
        b_ft5_h2h, r_ft5_h2h = get_h2h_record(ft5_h2h, blue_team, red_team)
        faster_team = blue_team if b_speed < r_speed else red_team
        est_time    = (b_speed + r_speed) / 2
        win_winner  = blue_team if blue_win_conf > red_win_conf else red_team
        ft5_winner  = blue_team if blue_ft5_conf > red_ft5_conf else red_team
        win_caution = 0.60 <= max(blue_win_conf, red_win_conf) < 0.65
        ft5_caution = 0.60 <= max(blue_ft5_conf, red_ft5_conf) < 0.65

        st.divider()

        # Team stats
        st.markdown("### 📋 Team Stats")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**🔵 {blue_team}**")
            st.write(f"Win rate: {b_wr*100:.1f}%")
            st.write(f"Form (L5): {b_form*100:.0f}%")
            st.write(f"Early game rate: {b_early*100:.1f}%")
            st.write(f"Avg kill time: {b_speed:.1f} min")
        with sc2:
            st.markdown(f"**🔴 {red_team}**")
            st.write(f"Win rate: {r_wr*100:.1f}%")
            st.write(f"Form (L5): {r_form*100:.0f}%")
            st.write(f"Early game rate: {r_early*100:.1f}%")
            st.write(f"Avg kill time: {r_speed:.1f} min")

        hc1, hc2 = st.columns(2)
        with hc1:
            st.write(f"Win H2H: {blue_team} {b_win_h2h} — {r_win_h2h} {red_team}")
        with hc2:
            st.write(f"Early H2H: {blue_team} {b_ft5_h2h} — {r_ft5_h2h} {red_team}")

        st.divider()

        # Match winner
        st.markdown("### 🏆 Match Winner")
        winner_color = "🔵" if blue_win_conf > red_win_conf else "🔴"
        st.markdown(f"#### {winner_color} Model pick: **{win_winner}**")

        wc1, wc2 = st.columns(2)
        with wc1:
            st.metric(f"🔵 {blue_team} win prob", f"{blue_win_conf*100:.1f}%",
                      delta=f"Edge: {win_blue_edge*100:.1f}%")
            st.write(f"Odds: {win_blue_odds} | Implied: {win_blue_impl*100:.1f}%")
            st.write(odds_label(win_blue_odds))
            if blue_win_conf > red_win_conf:
                st.info(f"💰 {win_blue_units}u — {win_blue_label}" if win_blue_units > 0
                        else "💰 ⛔ SKIP")
        with wc2:
            st.metric(f"🔴 {red_team} win prob", f"{red_win_conf*100:.1f}%",
                      delta=f"Edge: {win_red_edge*100:.1f}%")
            st.write(f"Odds: {win_red_odds} | Implied: {win_red_impl*100:.1f}%")
            st.write(odds_label(win_red_odds))
            if red_win_conf > blue_win_conf:
                st.info(f"💰 {win_red_units}u — {win_red_label}" if win_red_units > 0
                        else "💰 ⛔ SKIP")

        st.markdown(f"**📊 Model confidence: {win_conf_level}** — {win_conf_desc}")
        for r in win_reasons:
            st.write(f"✔ {r}")
        for w in win_warnings:
            st.write(f"⚠️ {w}")
        if win_caution:
            st.warning("Win probability in 60-65% range — backtest shows only 54.3% actual accuracy here, be cautious")

        st.divider()

        # First to five
        st.markdown("### ⚔️ First to Five Kills")
        ft5_color = "🔵" if blue_ft5_conf > red_ft5_conf else "🔴"
        st.markdown(f"#### {ft5_color} Model pick: **{ft5_winner}**")
        st.caption(f"⏱️ Est. 5 kills at ~minute {est_time:.1f} ({faster_team} historically faster)")

        fc1, fc2 = st.columns(2)
        with fc1:
            st.metric(f"🔵 {blue_team} win prob", f"{blue_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_blue_edge*100:.1f}%")
            st.write(f"Odds: {ft5_blue_odds} | Implied: {ft5_blue_impl*100:.1f}%")
            st.write(odds_label(ft5_blue_odds))
            if blue_ft5_conf > red_ft5_conf:
                st.info(f"💰 {ft5_blue_units}u — {ft5_blue_label}" if ft5_blue_units > 0
                        else "💰 ⛔ SKIP")
        with fc2:
            st.metric(f"🔴 {red_team} win prob", f"{red_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_red_edge*100:.1f}%")
            st.write(f"Odds: {ft5_red_odds} | Implied: {ft5_red_impl*100:.1f}%")
            st.write(odds_label(ft5_red_odds))
            if red_ft5_conf > blue_ft5_conf:
                st.info(f"💰 {ft5_red_units}u — {ft5_red_label}" if ft5_red_units > 0
                        else "💰 ⛔ SKIP")

        st.markdown(f"**📊 Model confidence: {ft5_conf_level}** — {ft5_conf_desc}")
        for r in ft5_reasons:
            st.write(f"✔ {r}")
        for w in ft5_warnings:
            st.write(f"⚠️ {w}")
        if ft5_caution:
            st.warning("FT5 probability in 60-65% range — treat with extra caution")

        st.divider()
        st.caption("~60% true accuracy | Trust 65%+ win prob | Best ROI at 2.30+ odds")
