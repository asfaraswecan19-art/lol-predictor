import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import pickle
import requests

st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 LoL Pro Match Predictor")
st.caption("Win + First to Five | ~64.68% true accuracy | Calibrated probabilities")

FORM_WINDOW = 5
BLUE_SIDE_WINRATE = 0.5312

TEAM_ALIASES = {
    'Team BDS': 'Team Shifters',
    'BDS':      'Team Shifters',
}

def normalize_team(name):
    return TEAM_ALIASES.get(str(name), str(name))

def rate_champ(win_rate, pc_rate):
    combined = (win_rate + pc_rate) / 2
    if combined >= 0.58:   return "🟢 Strong"
    elif combined >= 0.50: return "🟡 Average"
    else:                  return "🔴 Weak"

def rate_agg(agg_score):
    if agg_score >= 0.58:   return "🟢 High aggression"
    elif agg_score >= 0.48: return "🟡 Average aggression"
    else:                   return "🔴 Low aggression"

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
            warnings_list.append("Mixed signals")
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

@st.cache_resource
def load_models():
    with open('model_payload.pkl', 'rb') as f:
        p = pickle.load(f)
    return p

with st.spinner("Loading models..."):
    p = load_models()

win_model        = p['win_model']
win_mlb          = p['win_mlb']
win_team_rate    = p['win_team_rate']
win_team_games   = p['win_team_games']
win_champ_rate   = p['win_champ_rate']
win_h2h          = p['win_h2h']
win_team_recent  = p['win_team_recent']
pc_rate          = p['pc_rate']
pc_games         = p['pc_games']
role_champ_rate  = p['role_champ_rate']
ft5_model        = p['ft5_model']
ft5_mlb          = p['ft5_mlb']
champ_aggression = p['champ_aggression']
team_early_rate  = p['team_early_rate']
team_kill_speed  = p['team_kill_speed']
ft5_h2h          = p['ft5_h2h']
ft5_team_recent  = p['ft5_team_recent']
ft5_team_games   = p['ft5_team_games']
team_lineups     = p['team_lineups']
all_teams        = p['all_teams']
all_champs       = p['all_champs']
PC_WEIGHT        = p.get('pc_weight', 0.10)
RC_WEIGHT        = p.get('rc_weight', 0.90)
H2H_CAP          = p.get('h2h_cap',  0.60)

POSITIONS = ['top', 'jng', 'mid', 'adc', 'sup']

st.success("Models ready!")

defaults = {
    'blue_team': None, 'red_team': None,
    'blue_top': None, 'blue_jg': None, 'blue_mid': None,
    'blue_adc': None, 'blue_sup': None,
    'red_top':  None, 'red_jg':  None, 'red_mid':  None,
    'red_adc':  None, 'red_sup': None,
    'blue_p_top': '', 'blue_p_jg': '', 'blue_p_mid': '',
    'blue_p_adc': '', 'blue_p_sup': '',
    'red_p_top':  '', 'red_p_jg':  '', 'red_p_mid':  '',
    'red_p_adc':  '', 'red_p_sup':  '',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def cap_h2h(rate):
    return max(1 - H2H_CAP, min(H2H_CAP, rate))

def get_h2h_rate(h2h_dict, blue, red):
    matchup = tuple(sorted([blue, red]))
    if matchup not in h2h_dict: return 0.5
    total = sum(h2h_dict[matchup].values())
    return cap_h2h(h2h_dict[matchup].get(blue, 0) / total) if total > 0 else 0.5

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
    if odds < 1.60:    return "⚠️ Low odds"
    elif odds >= 2.30: return "🔥 Great odds"
    else:              return "✅ Good odds"

def calc_edge(conf, odds):
    implied = 1 / odds
    edge    = conf - implied
    if edge < 0.08:   units, label = 0, "⛔ SKIP"
    elif edge < 0.12: units, label = 1, "✅ DECENT"
    elif edge < 0.18: units, label = 2, "✅ STRONG"
    else:             units, label = 3, "🔥 VERY STRONG"
    return edge, units, label, implied

def show_signal(label, b_val, r_val, low_t, high_t,
                blue_name, red_name, fmt=".1f"):
    diff     = b_val - r_val
    abs_diff = abs(diff)
    if abs_diff >= high_t:   strength = "🟢 Strong"
    elif abs_diff >= low_t:  strength = "🟡 Moderate"
    else:                    strength = "⚪ Weak"
    direction = f"favours 🔵 {blue_name}" if diff > 0 else \
                (f"favours 🔴 {red_name}" if diff < 0 else "even")
    st.write(f"**{label}:** 🔵 {b_val*100:{fmt}}% vs "
             f"🔴 {r_val*100:{fmt}}% — {strength} {direction}")

def autofill_players(lineup, side):
    if isinstance(lineup, dict):
        mapping = {
            f'{side}_p_top': lineup.get('top', ''),
            f'{side}_p_jg':  lineup.get('jng', ''),
            f'{side}_p_mid': lineup.get('mid', ''),
            f'{side}_p_adc': lineup.get('adc', ''),
            f'{side}_p_sup': lineup.get('sup', ''),
        }
    else:
        vals = list(lineup) + [''] * (5 - len(lineup))
        mapping = {
            f'{side}_p_top': vals[0],
            f'{side}_p_jg':  vals[1],
            f'{side}_p_mid': vals[2],
            f'{side}_p_adc': vals[3],
            f'{side}_p_sup': vals[4],
        }
    for k, v in mapping.items():
        st.session_state[k] = v

def get_blended_avg(players, picks, side='blue'):
    rates = []
    for i, (player, champ) in enumerate(zip(players, picks)):
        pos    = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        pc_key = (player.strip(), champ.strip())
        rc_key = (pos, champ.strip())
        pc_val = pc_rate.get(pc_key, 0.5) if player.strip() else 0.5
        rc_val = role_champ_rate.get(rc_key, 0.5)
        rates.append(PC_WEIGHT * pc_val + RC_WEIGHT * rc_val)
    return sum(rates) / len(rates) if rates else 0.5

def get_draft_only_prediction(blue, red, blue_players, red_players,
                               b_champ_wr, r_champ_wr,
                               b_pc_avg, r_pc_avg):
    b_win_enc = pd.DataFrame(win_mlb.transform([blue]),
        columns=['blue_' + c for c in win_mlb.classes_])
    r_win_enc = pd.DataFrame(win_mlb.transform([red]),
        columns=['red_'  + c for c in win_mlb.classes_])

    neutral_row = pd.DataFrame([[
        0.5, 0.5, 0.0, 50, 50,
        b_champ_wr, r_champ_wr, b_champ_wr - r_champ_wr,
        0.5, 0.5, 0.5, 0.0,
        BLUE_SIDE_WINRATE,
        b_pc_avg, r_pc_avg, b_pc_avg - r_pc_avg,
    ]], columns=[
        'blue_team_winrate', 'red_team_winrate', 'team_winrate_diff',
        'blue_team_games', 'red_team_games',
        'blue_avg_winrate', 'red_avg_winrate', 'winrate_diff',
        'h2h_winrate', 'blue_form', 'red_form', 'form_diff',
        'blue_side_advantage', 'blue_pc_avg', 'red_pc_avg', 'pc_avg_diff',
    ])

    win_row  = pd.concat([b_win_enc, r_win_enc, neutral_row], axis=1)
    win_prob = win_model.predict_proba(win_row)[0]
    blue_draft_win = win_prob[1]
    red_draft_win  = win_prob[0]

    b_ft5_enc = pd.DataFrame(ft5_mlb.transform([blue]),
        columns=['blue_' + c for c in ft5_mlb.classes_])
    r_ft5_enc = pd.DataFrame(ft5_mlb.transform([red]),
        columns=['red_'  + c for c in ft5_mlb.classes_])

    b_agg = sum(champ_aggression.get(c, 0.5) for c in blue) / len(blue)
    r_agg = sum(champ_aggression.get(c, 0.5) for c in red)  / len(red)

    neutral_ft5 = pd.DataFrame([[
        b_agg, r_agg, b_agg - r_agg,
        0.5, 0.5, 0.0,
        10.0, 10.0, 0.0,
        0.5, 0.5, 0.5, 0.0,
    ]], columns=[
        'blue_aggression', 'red_aggression', 'aggression_diff',
        'blue_early_rate', 'red_early_rate', 'early_rate_diff',
        'blue_kill_speed', 'red_kill_speed', 'speed_diff',
        'h2h_early_rate', 'blue_early_form', 'red_early_form', 'early_form_diff'
    ])

    ft5_row  = pd.concat([b_ft5_enc, r_ft5_enc, neutral_ft5], axis=1)
    ft5_prob = ft5_model.predict_proba(ft5_row)[0]
    return win_prob[1], win_prob[0], ft5_prob[1], ft5_prob[0]

def get_claude_reasoning(
        blue_team, red_team, blue_picks, red_picks,
        blue_players, red_players,
        blue_win_conf, red_win_conf,
        blue_ft5_conf, red_ft5_conf,
        b_wr, r_wr, b_form, r_form,
        b_champ_wr, r_champ_wr,
        b_pc_avg, r_pc_avg,
        b_win_h2h, r_win_h2h,
        b_agg, r_agg, b_early, r_early, b_speed, r_speed,
        win_blue_odds, win_red_odds, ft5_blue_odds, ft5_red_odds):

    pos_labels = ['Top', 'Jng', 'Mid', 'ADC', 'Sup']
    blue_roster = ', '.join([
        f"{blue_players[i] if i < len(blue_players) and blue_players[i].strip() else 'Unknown'}"
        f" ({pos_labels[i]}: {blue_picks[i]}, "
        f"champ wr {win_champ_rate.get(blue_picks[i], 0.5)*100:.0f}%, "
        f"agg {champ_aggression.get(blue_picks[i], 0.5)*100:.0f}%)"
        for i in range(len(blue_picks))
    ])
    red_roster = ', '.join([
        f"{red_players[i] if i < len(red_players) and red_players[i].strip() else 'Unknown'}"
        f" ({pos_labels[i]}: {red_picks[i]}, "
        f"champ wr {win_champ_rate.get(red_picks[i], 0.5)*100:.0f}%, "
        f"agg {champ_aggression.get(red_picks[i], 0.5)*100:.0f}%)"
        for i in range(len(red_picks))
    ])

    prompt = f"""You are an expert League of Legends analyst. Analyze this pro match and explain the predictions in 3-4 sentences each.

MATCH: {blue_team} (Blue) vs {red_team} (Red)

BLUE - {blue_team}: {blue_roster}
Win rate: {b_wr*100:.1f}% | Form: {b_form*100:.0f}% | H2H: {b_win_h2h}-{r_win_h2h}
Champ quality: {b_champ_wr*100:.1f}% | Player-champ: {b_pc_avg*100:.1f}%
Early rate: {b_early*100:.1f}% | Kill speed: {b_speed:.1f}m | Aggression: {b_agg*100:.1f}%

RED - {red_team}: {red_roster}
Win rate: {r_wr*100:.1f}% | Form: {r_form*100:.0f}% | H2H: {r_win_h2h}-{b_win_h2h}
Champ quality: {r_champ_wr*100:.1f}% | Player-champ: {r_pc_avg*100:.1f}%
Early rate: {r_early*100:.1f}% | Kill speed: {r_speed:.1f}m | Aggression: {r_agg*100:.1f}%

MODEL: Winner {blue_team} {blue_win_conf*100:.1f}% vs {red_team} {red_win_conf*100:.1f}%
FT5: {blue_team} {blue_ft5_conf*100:.1f}% vs {red_team} {red_ft5_conf*100:.1f}%

1. MATCH WINNER REASONING (3-4 sentences): Why does the model favour {blue_team if blue_win_conf > red_win_conf else red_team}?
2. FIRST TO 5 KILLS REASONING (3-4 sentences): Which team has the more aggressive early composition and why?

Be specific about champion picks, player strengths, and game style. Keep it concise."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": st.secrets["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        if 'content' in data and len(data['content']) > 0:
            return data['content'][0]['text']
        return f"Analysis unavailable — {data}"
    except Exception as e:
        return f"Analysis unavailable: {str(e)}"

# =================================================================
# UI
# =================================================================
st.divider()

btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
with btn_col1:
    if st.button("🗑️ Clear", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()
with btn_col2:
    if st.button("🔄 Swap", use_container_width=True):
        st.session_state['blue_team'], st.session_state['red_team'] = \
            st.session_state['red_team'], st.session_state['blue_team']
        for pos in ['top', 'jg', 'mid', 'adc', 'sup']:
            st.session_state[f'blue_{pos}'], st.session_state[f'red_{pos}'] = \
                st.session_state[f'red_{pos}'], st.session_state[f'blue_{pos}']
            st.session_state[f'blue_p_{pos}'], st.session_state[f'red_p_{pos}'] = \
                st.session_state[f'red_p_{pos}'], st.session_state[f'blue_p_{pos}']
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 Blue")
    prev_blue = st.session_state.get('_prev_blue_team', None)
    blue_team = st.selectbox(
        "Team (optional)", options=[None] + all_teams,
        format_func=lambda x: "— no team —" if x is None else x,
        key='blue_team')
    if blue_team != prev_blue:
        st.session_state['_prev_blue_team'] = blue_team
        blue_norm = normalize_team(blue_team) if blue_team else None
        if blue_norm and blue_norm in team_lineups:
            autofill_players(team_lineups[blue_norm], 'blue')

    blue_top = st.selectbox("Top", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='blue_top')
    blue_p_top = st.text_input("Top player", key='blue_p_top')
    blue_jg  = st.selectbox("Jungle", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='blue_jg')
    blue_p_jg = st.text_input("Jungle player", key='blue_p_jg')
    blue_mid = st.selectbox("Mid", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='blue_mid')
    blue_p_mid = st.text_input("Mid player", key='blue_p_mid')
    blue_adc = st.selectbox("ADC", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='blue_adc')
    blue_p_adc = st.text_input("ADC player", key='blue_p_adc')
    blue_sup = st.selectbox("Support", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='blue_sup')
    blue_p_sup = st.text_input("Support player", key='blue_p_sup')

with col2:
    st.markdown("### 🔴 Red")
    prev_red = st.session_state.get('_prev_red_team', None)
    red_team = st.selectbox(
        "Team (optional)", options=[None] + all_teams,
        format_func=lambda x: "— no team —" if x is None else x,
        key='red_team')
    if red_team != prev_red:
        st.session_state['_prev_red_team'] = red_team
        red_norm = normalize_team(red_team) if red_team else None
        if red_norm and red_norm in team_lineups:
            autofill_players(team_lineups[red_norm], 'red')

    red_top  = st.selectbox("Top", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='red_top')
    red_p_top = st.text_input("Top player", key='red_p_top')
    red_jg   = st.selectbox("Jungle", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='red_jg')
    red_p_jg = st.text_input("Jungle player", key='red_p_jg')
    red_mid  = st.selectbox("Mid", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='red_mid')
    red_p_mid = st.text_input("Mid player", key='red_p_mid')
    red_adc  = st.selectbox("ADC", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='red_adc')
    red_p_adc = st.text_input("ADC player", key='red_p_adc')
    red_sup  = st.selectbox("Support", options=[None] + all_champs,
        format_func=lambda x: "— pick —" if x is None else x, key='red_sup')
    red_p_sup = st.text_input("Support player", key='red_p_sup')

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
    blue = [c for c in [blue_top, blue_jg, blue_mid, blue_adc, blue_sup] if c is not None]
    red  = [c for c in [red_top,  red_jg,  red_mid,  red_adc,  red_sup]  if c is not None]
    blue_players = [blue_p_top, blue_p_jg, blue_p_mid, blue_p_adc, blue_p_sup]
    red_players  = [red_p_top,  red_p_jg,  red_p_mid,  red_p_adc,  red_p_sup]

    blue_team_norm = normalize_team(blue_team) if blue_team else None
    red_team_norm  = normalize_team(red_team)  if red_team  else None
    blue_team_name = blue_team if blue_team else "Blue Team"
    red_team_name  = red_team  if red_team  else "Red Team"

    picked = [c for c in blue + red if c]
    if len(picked) != len(set(picked)) and len(picked) > 0:
        st.error("Each champion must be unique across both teams!")
    elif blue_team and red_team and blue_team_norm == red_team_norm:
        st.error("Blue and red team can't be the same!")
    else:
        missing = []
        if not blue_team: missing.append("blue team")
        if not red_team:  missing.append("red team")
        if len(blue) < 5: missing.append(f"blue picks ({len(blue)}/5)")
        if len(red)  < 5: missing.append(f"red picks ({len(red)}/5)")
        if missing:
            st.info(f"ℹ️ Missing: {', '.join(missing)} — using dataset averages")

        if len(blue) == 5:
            b_win_enc = pd.DataFrame(win_mlb.transform([blue]),
                columns=['blue_' + c for c in win_mlb.classes_])
        else:
            b_win_enc = pd.DataFrame([[0]*len(win_mlb.classes_)],
                columns=['blue_' + c for c in win_mlb.classes_])

        if len(red) == 5:
            r_win_enc = pd.DataFrame(win_mlb.transform([red]),
                columns=['red_' + c for c in win_mlb.classes_])
        else:
            r_win_enc = pd.DataFrame([[0]*len(win_mlb.classes_)],
                columns=['red_' + c for c in win_mlb.classes_])

        b_wr    = win_team_rate.get(blue_team_norm, 0.5)  if blue_team_norm else 0.5
        r_wr    = win_team_rate.get(red_team_norm,  0.5)  if red_team_norm  else 0.5
        b_games = win_team_games.get(blue_team_norm, 0)   if blue_team_norm else 0
        r_games = win_team_games.get(red_team_norm,  0)   if red_team_norm  else 0

        b_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in blue) / len(blue) if blue else 0.5
        r_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in red)  / len(red)  if red  else 0.5

        win_h2h_r  = get_h2h_rate(win_h2h, blue_team_norm, red_team_norm) \
                     if blue_team_norm and red_team_norm else 0.5
        b_form     = get_form(win_team_recent, blue_team_norm) if blue_team_norm else 0.5
        r_form     = get_form(win_team_recent, red_team_norm)  if red_team_norm  else 0.5
        h2h_total  = get_h2h_total(win_h2h, blue_team_norm, red_team_norm) \
                     if blue_team_norm and red_team_norm else 0
        b_win_h2h, r_win_h2h = get_h2h_record(win_h2h, blue_team_norm, red_team_norm) \
                                if blue_team_norm and red_team_norm else (0, 0)

        b_pc_avg = get_blended_avg(blue_players, blue) if blue else 0.5
        r_pc_avg = get_blended_avg(red_players,  red)  if red  else 0.5

        win_extra_row = pd.DataFrame([[
            b_wr, r_wr, b_wr - r_wr,
            b_games, r_games,
            b_champ_wr, r_champ_wr, b_champ_wr - r_champ_wr,
            win_h2h_r, b_form, r_form, b_form - r_form,
            BLUE_SIDE_WINRATE, b_pc_avg, r_pc_avg, b_pc_avg - r_pc_avg,
        ]], columns=[
            'blue_team_winrate', 'red_team_winrate', 'team_winrate_diff',
            'blue_team_games', 'red_team_games',
            'blue_avg_winrate', 'red_avg_winrate', 'winrate_diff',
            'h2h_winrate', 'blue_form', 'red_form', 'form_diff',
            'blue_side_advantage', 'blue_pc_avg', 'red_pc_avg', 'pc_avg_diff',
        ])

        win_row  = pd.concat([b_win_enc, r_win_enc, win_extra_row], axis=1)
        win_prob = win_model.predict_proba(win_row)[0]
        blue_win_conf = win_prob[1]
        red_win_conf  = win_prob[0]

        if len(blue) == 5:
            b_ft5_enc = pd.DataFrame(ft5_mlb.transform([blue]),
                columns=['blue_' + c for c in ft5_mlb.classes_])
        else:
            b_ft5_enc = pd.DataFrame([[0]*len(ft5_mlb.classes_)],
                columns=['blue_' + c for c in ft5_mlb.classes_])

        if len(red) == 5:
            r_ft5_enc = pd.DataFrame(ft5_mlb.transform([red]),
                columns=['red_' + c for c in ft5_mlb.classes_])
        else:
            r_ft5_enc = pd.DataFrame([[0]*len(ft5_mlb.classes_)],
                columns=['red_' + c for c in ft5_mlb.classes_])

        b_agg        = sum(champ_aggression.get(c, 0.5) for c in blue) / len(blue) if blue else 0.5
        r_agg        = sum(champ_aggression.get(c, 0.5) for c in red)  / len(red)  if red  else 0.5
        b_early      = team_early_rate.get(blue_team_norm, 0.5)  if blue_team_norm else 0.5
        r_early      = team_early_rate.get(red_team_norm,  0.5)  if red_team_norm  else 0.5
        b_speed      = team_kill_speed.get(blue_team_norm, 10.0) if blue_team_norm else 10.0
        r_speed      = team_kill_speed.get(red_team_norm,  10.0) if red_team_norm  else 10.0
        ft5_h2h_r    = get_h2h_rate(ft5_h2h, blue_team_norm, red_team_norm) \
                       if blue_team_norm and red_team_norm else 0.5
        ft5_h2h_tot  = get_h2h_total(ft5_h2h, blue_team_norm, red_team_norm) \
                       if blue_team_norm and red_team_norm else 0
        b_ft5_h2h, r_ft5_h2h = get_h2h_record(ft5_h2h, blue_team_norm, red_team_norm) \
                                if blue_team_norm and red_team_norm else (0, 0)
        b_early_form = get_form(ft5_team_recent, blue_team_norm) if blue_team_norm else 0.5
        r_early_form = get_form(ft5_team_recent, red_team_norm)  if red_team_norm  else 0.5

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

        if len(blue) == 5 and len(red) == 5:
            blue_draft_win, red_draft_win, blue_draft_ft5, red_draft_ft5 = \
                get_draft_only_prediction(blue, red, blue_players, red_players,
                                          b_champ_wr, r_champ_wr, b_pc_avg, r_pc_avg)
        else:
            blue_draft_win = red_draft_win = blue_draft_ft5 = red_draft_ft5 = None

        win_blue_edge, win_blue_units, win_blue_label, win_blue_impl = calc_edge(blue_win_conf, win_blue_odds)
        win_red_edge,  win_red_units,  win_red_label,  win_red_impl  = calc_edge(red_win_conf,  win_red_odds)
        ft5_blue_edge, ft5_blue_units, ft5_blue_label, ft5_blue_impl = calc_edge(blue_ft5_conf, ft5_blue_odds)
        ft5_red_edge,  ft5_red_units,  ft5_red_label,  ft5_red_impl  = calc_edge(red_ft5_conf,  ft5_red_odds)

        win_winner  = blue_team_name if blue_win_conf > red_win_conf else red_team_name
        ft5_winner  = blue_team_name if blue_ft5_conf > red_ft5_conf else red_team_name
        faster_team = blue_team_name if b_speed < r_speed else red_team_name
        est_time    = (b_speed + r_speed) / 2
        pos_labels  = ['Top', 'Jng', 'Mid', 'ADC', 'Sup']

        win_conf_level, win_conf_desc, win_reasons, win_warnings = model_confidence(
            b_games, r_games, h2h_total,
            b_form - r_form, b_wr - r_wr, b_champ_wr - r_champ_wr)

        ft5_conf_level, ft5_conf_desc, ft5_reasons, ft5_warnings = model_confidence(
            ft5_team_games.get(blue_team_norm, 0) if blue_team_norm else 0,
            ft5_team_games.get(red_team_norm,  0) if red_team_norm  else 0,
            ft5_h2h_tot,
            b_early_form - r_early_form,
            b_early - r_early,
            b_agg - r_agg)

        win_caution = 0.60 <= max(blue_win_conf, red_win_conf) < 0.65
        ft5_caution = 0.60 <= max(blue_ft5_conf, red_ft5_conf) < 0.65

        st.divider()

        # =============================================================
        # TEAM STATS
        # =============================================================
        with st.expander("📋 Team Stats", expanded=False):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"**🔵 {blue_team_name}**")
                st.write(f"Win rate: {b_wr*100:.1f}%")
                st.write(f"Form (L5): {b_form*100:.0f}%")
                st.write(f"Early rate: {b_early*100:.1f}%")
                st.write(f"Avg kill time: {b_speed:.1f}m")
            with sc2:
                st.markdown(f"**🔴 {red_team_name}**")
                st.write(f"Win rate: {r_wr*100:.1f}%")
                st.write(f"Form (L5): {r_form*100:.0f}%")
                st.write(f"Early rate: {r_early*100:.1f}%")
                st.write(f"Avg kill time: {r_speed:.1f}m")
            hc1, hc2 = st.columns(2)
            with hc1:
                st.write(f"Win H2H: {blue_team_name} {b_win_h2h}–{r_win_h2h} {red_team_name}")
            with hc2:
                st.write(f"Early H2H: {blue_team_name} {b_ft5_h2h}–{r_ft5_h2h} {red_team_name}")

        # =============================================================
        # MATCH WINNER
        # =============================================================
        st.markdown("### 🏆 Match Winner")
        winner_color = "🔵" if blue_win_conf > red_win_conf else "🔴"
        st.markdown(f"#### {winner_color} Model pick: **{win_winner}**")
        wc1, wc2 = st.columns(2)
        with wc1:
            st.metric(f"🔵 {blue_team_name}", f"{blue_win_conf*100:.1f}%",
                      delta=f"Edge: {win_blue_edge*100:.1f}%")
            st.write(f"Odds: {win_blue_odds} | Implied: {win_blue_impl*100:.1f}%")
            st.write(odds_label(win_blue_odds))
            if blue_win_conf > red_win_conf:
                st.info(f"💰 {win_blue_units}u — {win_blue_label}" if win_blue_units > 0
                        else "💰 ⛔ SKIP")
        with wc2:
            st.metric(f"🔴 {red_team_name}", f"{red_win_conf*100:.1f}%",
                      delta=f"Edge: {win_red_edge*100:.1f}%")
            st.write(f"Odds: {win_red_odds} | Implied: {win_red_impl*100:.1f}%")
            st.write(odds_label(win_red_odds))
            if red_win_conf > blue_win_conf:
                st.info(f"💰 {win_red_units}u — {win_red_label}" if win_red_units > 0
                        else "💰 ⛔ SKIP")

        if blue_draft_win is not None:
            dw = "🔵" if blue_draft_win > red_draft_win else "🔴"
            dn = blue_team_name if blue_draft_win > red_draft_win else red_team_name
            st.caption(f"⚖️ Draft-only: 🔵 {blue_draft_win*100:.1f}% vs 🔴 {red_draft_win*100:.1f}% — {dw} {dn} has better draft")

        st.markdown(f"**📊 Confidence: {win_conf_level}** — {win_conf_desc}")
        for r in win_reasons:
            st.write(f"✔ {r}")
        for w in win_warnings:
            if "Mixed signals" in w:
                wr_dir    = f"🔵 {blue_team_name}" if b_wr > r_wr else f"🔴 {red_team_name}"
                form_dir  = f"🔵 {blue_team_name}" if b_form > r_form else f"🔴 {red_team_name}"
                champ_dir = f"🔵 {blue_team_name}" if b_champ_wr > r_champ_wr else f"🔴 {red_team_name}"
                st.write(f"⚠️ Mixed signals — win rate favours {wr_dir} ({abs(b_wr-r_wr)*100:.1f}%), "
                         f"form favours {form_dir} ({abs(b_form-r_form)*100:.0f}%), "
                         f"champ quality favours {champ_dir} ({abs(b_champ_wr-r_champ_wr)*100:.1f}%)")
            elif "Weak signal" in w:
                st.write(f"⚠️ Weak signals — win rate diff: {abs(b_wr-r_wr)*100:.1f}%, "
                         f"form diff: {abs(b_form-r_form)*100:.0f}%, "
                         f"champ diff: {abs(b_champ_wr-r_champ_wr)*100:.1f}%")
            else:
                st.write(f"⚠️ {w}")
        if win_caution:
            st.warning("60-65% range — backtest shows ~57% actual accuracy here, be cautious")

        # Win signal breakdown expander
        with st.expander("📊 Win Signal Breakdown", expanded=False):
            show_signal("Team win rate",    b_wr,       r_wr,       0.05, 0.15, blue_team_name, red_team_name)
            show_signal("Recent form",      b_form,     r_form,     0.10, 0.25, blue_team_name, red_team_name, ".0f")
            show_signal("Champion quality", b_champ_wr, r_champ_wr, 0.02, 0.06, blue_team_name, red_team_name)
            show_signal("Player-champ wr",  b_pc_avg,   r_pc_avg,   0.03, 0.08, blue_team_name, red_team_name)

            if b_win_h2h + r_win_h2h > 0:
                h2h_diff  = win_h2h_r - 0.5
                h_str     = "🟢 Strong" if abs(h2h_diff) >= 0.25 else \
                            ("🟡 Moderate" if abs(h2h_diff) >= 0.10 else "⚪ Weak")
                direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 \
                            else f"favours 🔴 {red_team_name}"
                st.write(f"**H2H:** 🔵 {b_win_h2h}–{r_win_h2h} 🔴 — {h_str} {direction}")
            else:
                st.write("**H2H:** No history — ⚪ Neutral")
            st.write(f"**Blue side:** 53.1% historical — ⚪ Slight edge 🔵 {blue_team_name}")

        # Champion ratings expander
        if blue or red:
            with st.expander("🏅 Champion Ratings", expanded=False):
                if blue:
                    st.markdown(f"**🔵 {blue_team_name}**")
                    for i, champ in enumerate(blue):
                        player = blue_players[i] if i < len(blue_players) else ''
                        cwr    = win_champ_rate.get(champ, 0.5)
                        pos    = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
                        rc_val = role_champ_rate.get((pos, champ.strip()), 0.5)
                        pc_val = pc_rate.get((player.strip(), champ.strip()), 0.5) \
                                 if player.strip() else rc_val
                        pcg    = pc_games.get((player.strip(), champ.strip()), 0) \
                                 if player.strip() else 0
                        blended = PC_WEIGHT * pc_val + RC_WEIGHT * rc_val
                        rating = rate_champ(cwr, blended)
                        lbl    = pos_labels[i] if i < len(pos_labels) else ''
                        name   = player if player.strip() else "Unknown"
                        gstr   = f"({pcg}g)" if pcg > 0 else ""
                        st.write(f"**{lbl}** {name} — {champ}: "
                                 f"role {rc_val*100:.0f}% | player {pc_val*100:.0f}% {gstr} → {rating}")
                if red:
                    st.markdown(f"**🔴 {red_team_name}**")
                    for i, champ in enumerate(red):
                        player = red_players[i] if i < len(red_players) else ''
                        cwr    = win_champ_rate.get(champ, 0.5)
                        pos    = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
                        rc_val = role_champ_rate.get((pos, champ.strip()), 0.5)
                        pc_val = pc_rate.get((player.strip(), champ.strip()), 0.5) \
                                 if player.strip() else rc_val
                        pcg    = pc_games.get((player.strip(), champ.strip()), 0) \
                                 if player.strip() else 0
                        blended = PC_WEIGHT * pc_val + RC_WEIGHT * rc_val
                        rating = rate_champ(cwr, blended)
                        lbl    = pos_labels[i] if i < len(pos_labels) else ''
                        name   = player if player.strip() else "Unknown"
                        gstr   = f"({pcg}g)" if pcg > 0 else ""
                        st.write(f"**{lbl}** {name} — {champ}: "
                                 f"role {rc_val*100:.0f}% | player {pc_val*100:.0f}% {gstr} → {rating}")

        st.divider()

        # =============================================================
        # FIRST TO FIVE
        # =============================================================
        st.markdown("### ⚔️ First to Five Kills")
        ft5_color = "🔵" if blue_ft5_conf > red_ft5_conf else "🔴"
        st.markdown(f"#### {ft5_color} Model pick: **{ft5_winner}**")
        st.caption(f"⏱️ Est. ~{est_time:.1f} min ({faster_team} historically faster)")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.metric(f"🔵 {blue_team_name}", f"{blue_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_blue_edge*100:.1f}%")
            st.write(f"Odds: {ft5_blue_odds} | Implied: {ft5_blue_impl*100:.1f}%")
            st.write(odds_label(ft5_blue_odds))
            if blue_ft5_conf > red_ft5_conf:
                st.info(f"💰 {ft5_blue_units}u — {ft5_blue_label}" if ft5_blue_units > 0
                        else "💰 ⛔ SKIP")
        with fc2:
            st.metric(f"🔴 {red_team_name}", f"{red_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_red_edge*100:.1f}%")
            st.write(f"Odds: {ft5_red_odds} | Implied: {ft5_red_impl*100:.1f}%")
            st.write(odds_label(ft5_red_odds))
            if red_ft5_conf > blue_ft5_conf:
                st.info(f"💰 {ft5_red_units}u — {ft5_red_label}" if ft5_red_units > 0
                        else "💰 ⛔ SKIP")

        if blue_draft_ft5 is not None:
            df5 = "🔵" if blue_draft_ft5 > red_draft_ft5 else "🔴"
            dn5 = blue_team_name if blue_draft_ft5 > red_draft_ft5 else red_team_name
            st.caption(f"⚖️ Draft-only: 🔵 {blue_draft_ft5*100:.1f}% vs 🔴 {red_draft_ft5*100:.1f}% — {df5} {dn5} more aggressive draft")

        st.markdown(f"**📊 Confidence: {ft5_conf_level}** — {ft5_conf_desc}")
        for r in ft5_reasons:
            st.write(f"✔ {r}")
        for w in ft5_warnings:
            if "Mixed signals" in w:
                early_dir = f"🔵 {blue_team_name}" if b_early > r_early else f"🔴 {red_team_name}"
                form_dir  = f"🔵 {blue_team_name}" if b_early_form > r_early_form else f"🔴 {red_team_name}"
                agg_dir   = f"🔵 {blue_team_name}" if b_agg > r_agg else f"🔴 {red_team_name}"
                st.write(f"⚠️ Mixed signals — early rate favours {early_dir} ({abs(b_early-r_early)*100:.1f}%), "
                         f"form favours {form_dir} ({abs(b_early_form-r_early_form)*100:.0f}%), "
                         f"aggression favours {agg_dir} ({abs(b_agg-r_agg)*100:.1f}%)")
            elif "Weak signal" in w:
                st.write(f"⚠️ Weak signals — early rate diff: {abs(b_early-r_early)*100:.1f}%, "
                         f"form diff: {abs(b_early_form-r_early_form)*100:.0f}%, "
                         f"aggression diff: {abs(b_agg-r_agg)*100:.1f}%")
            else:
                st.write(f"⚠️ {w}")
        if ft5_caution:
            st.warning("60-65% range — treat with extra caution")

        # FT5 signal breakdown expander
        with st.expander("📊 FT5 Signal Breakdown", expanded=False):
            show_signal("Early game rate", b_early,      r_early,      0.05, 0.15, blue_team_name, red_team_name)
            show_signal("Early form",      b_early_form, r_early_form, 0.10, 0.25, blue_team_name, red_team_name, ".0f")
            show_signal("Aggression",      b_agg,        r_agg,        0.03, 0.08, blue_team_name, red_team_name)

            faster   = blue_team_name if b_speed < r_speed else red_team_name
            spd_diff = abs(b_speed - r_speed)
            if spd_diff >= 2.0:   spd_str = f"🟢 Strong — {faster} significantly faster"
            elif spd_diff >= 0.5: spd_str = f"🟡 Moderate — {faster} slightly faster"
            else:                 spd_str = "⚪ Weak — similar speed"
            st.write(f"**Kill speed:** 🔵 {b_speed:.1f}m vs 🔴 {r_speed:.1f}m — {spd_str}")

            if b_ft5_h2h + r_ft5_h2h > 0:
                h2h_diff  = ft5_h2h_r - 0.5
                h_str     = "🟢 Strong" if abs(h2h_diff) >= 0.25 else \
                            ("🟡 Moderate" if abs(h2h_diff) >= 0.10 else "⚪ Weak")
                direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 \
                            else f"favours 🔴 {red_team_name}"
                st.write(f"**Early H2H:** 🔵 {b_ft5_h2h}–{r_ft5_h2h} 🔴 — {h_str} {direction}")
            else:
                st.write("**Early H2H:** No history — ⚪ Neutral")

        # Champion aggression expander
        if blue or red:
            with st.expander("🔥 Champion Aggression", expanded=False):
                if blue:
                    st.markdown(f"**🔵 {blue_team_name}**")
                    for i, champ in enumerate(blue):
                        player = blue_players[i] if i < len(blue_players) else ''
                        agg    = champ_aggression.get(champ, 0.5)
                        rating = rate_agg(agg)
                        lbl    = pos_labels[i] if i < len(pos_labels) else ''
                        name   = player if player.strip() else "Unknown"
                        st.write(f"**{lbl}** {name} — {champ}: {agg*100:.0f}% → {rating}")
                if red:
                    st.markdown(f"**🔴 {red_team_name}**")
                    for i, champ in enumerate(red):
                        player = red_players[i] if i < len(red_players) else ''
                        agg    = champ_aggression.get(champ, 0.5)
                        rating = rate_agg(agg)
                        lbl    = pos_labels[i] if i < len(pos_labels) else ''
                        name   = player if player.strip() else "Unknown"
                        st.write(f"**{lbl}** {name} — {champ}: {agg*100:.0f}% → {rating}")

        st.divider()

        # =============================================================
        # AI ANALYSIS
        # =============================================================
        if len(blue) == 5 and len(red) == 5:
            with st.expander("🤖 AI Analysis", expanded=False):
                with st.spinner("Generating analysis..."):
                    reasoning = get_claude_reasoning(
                        blue_team_name, red_team_name,
                        blue, red, blue_players, red_players,
                        blue_win_conf, red_win_conf,
                        blue_ft5_conf, red_ft5_conf,
                        b_wr, r_wr, b_form, r_form,
                        b_champ_wr, r_champ_wr,
                        b_pc_avg, r_pc_avg,
                        b_win_h2h, r_win_h2h,
                        b_agg, r_agg, b_early, r_early, b_speed, r_speed,
                        win_blue_odds, win_red_odds,
                        ft5_blue_odds, ft5_red_odds)
                st.write(reasoning)

        st.divider()
        st.caption("~64.68% true accuracy | Trust 65%+ | Best ROI at 2.30+ odds")
