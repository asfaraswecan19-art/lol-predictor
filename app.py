import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import pickle
import requests
import json

st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="🎮",
    layout="centered"
)

st.title("🎮 LoL Pro Match Predictor")
st.caption("Win + First to Five | ~61.88% true accuracy | Calibrated probabilities")

FORM_WINDOW = 5
BLUE_SIDE_WINRATE = 0.5312

def rate_champ(win_rate, pc_rate):
    combined = (win_rate + pc_rate) / 2
    if combined >= 0.58:   return "🟢 Strong"
    elif combined >= 0.50: return "🟡 Average"
    else:                  return "🔴 Weak"

def rate_agg(agg_score):
    if agg_score >= 0.58:   return "🟢 High aggression"
    elif agg_score >= 0.48: return "🟡 Average aggression"
    else:                   return "🔴 Low aggression"

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

# =================================================================
# CLAUDE REASONING
# =================================================================
def get_claude_reasoning(
        blue_team, red_team,
        blue_picks, red_picks,
        blue_players, red_players,
        blue_win_conf, red_win_conf,
        blue_ft5_conf, red_ft5_conf,
        b_wr, r_wr, b_form, r_form,
        b_champ_wr, r_champ_wr,
        b_pc_avg, r_pc_avg,
        b_win_h2h, r_win_h2h,
        b_agg, r_agg,
        b_early, r_early,
        b_speed, r_speed,
        win_blue_odds, win_red_odds,
        ft5_blue_odds, ft5_red_odds):

    positions = ['Top', 'Jng', 'Mid', 'ADC', 'Sup']

    blue_roster = ', '.join([
        f"{blue_players[i]} ({positions[i]}: {blue_picks[i]}, "
        f"champ wr {win_champ_rate.get(blue_picks[i], 0.5)*100:.0f}%, "
        f"agg {champ_aggression.get(blue_picks[i], 0.5)*100:.0f}%)"
        for i in range(len(blue_picks))
    ])

    red_roster = ', '.join([
        f"{red_players[i]} ({positions[i]}: {red_picks[i]}, "
        f"champ wr {win_champ_rate.get(red_picks[i], 0.5)*100:.0f}%, "
        f"agg {champ_aggression.get(red_picks[i], 0.5)*100:.0f}%)"
        for i in range(len(red_picks))
    ])

    prompt = f"""You are an expert League of Legends analyst. Analyze this pro match and explain the predictions in 3-4 sentences each. Be specific about champion synergies, early/late game dynamics, and player strengths.

MATCH: {blue_team} (Blue) vs {red_team} (Red)

BLUE TEAM - {blue_team}:
{blue_roster}
Team win rate: {b_wr*100:.1f}% | Form (L5): {b_form*100:.0f}% | H2H: {b_win_h2h}-{r_win_h2h} vs opponent
Champion quality avg: {b_champ_wr*100:.1f}% | Player-champ avg: {b_pc_avg*100:.1f}%
Early game rate: {b_early*100:.1f}% | Avg kill speed: {b_speed:.1f} min | Aggression: {b_agg*100:.1f}%

RED TEAM - {red_team}:
{red_roster}
Team win rate: {r_wr*100:.1f}% | Form (L5): {r_form*100:.0f}% | H2H: {r_win_h2h}-{b_win_h2h} vs opponent
Champion quality avg: {r_champ_wr*100:.1f}% | Player-champ avg: {r_pc_avg*100:.1f}%
Early game rate: {r_early*100:.1f}% | Avg kill speed: {r_speed:.1f} min | Aggression: {r_agg*100:.1f}%

MODEL PREDICTIONS:
Match winner: {blue_team} {blue_win_conf*100:.1f}% vs {red_team} {red_win_conf*100:.1f}%
First to 5 kills: {blue_team} {blue_ft5_conf*100:.1f}% vs {red_team} {red_ft5_conf*100:.1f}%

Please provide:
1. MATCH WINNER REASONING (3-4 sentences): Why does the model favour {blue_team if blue_win_conf > red_win_conf else red_team}? Mention specific champion picks, player strengths, team stats, and game style.
2. FIRST TO 5 KILLS REASONING (3-4 sentences): Which team has the more aggressive early game composition and why? Mention specific champions known for early kills and jungle pressure.

Keep it concise and analytical. Focus on what actually matters for these specific picks."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": st.secrets["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        data = response.json()
        return data['content'][0]['text']
    except Exception as e:
        return f"Reasoning unavailable: {str(e)}"

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
    if st.button("🔄 Swap Sides", use_container_width=True):
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
    st.markdown("### 🔵 Blue Side")
    prev_blue_team = st.session_state.get('_prev_blue_team', None)
    blue_team = st.selectbox(
        "Team (optional)", options=[None] + all_teams,
        format_func=lambda x: "— no team —" if x is None else x,
        key='blue_team')
    if blue_team != prev_blue_team:
        st.session_state['_prev_blue_team'] = blue_team
        if blue_team and blue_team in team_lineups:
            autofill_players(team_lineups[blue_team], 'blue')

    blue_top = st.selectbox("Top (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='blue_top')
    blue_p_top = st.text_input("Top player (optional)", key='blue_p_top')
    blue_jg  = st.selectbox("Jungle (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='blue_jg')
    blue_p_jg = st.text_input("Jungle player (optional)", key='blue_p_jg')
    blue_mid = st.selectbox("Mid (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='blue_mid')
    blue_p_mid = st.text_input("Mid player (optional)", key='blue_p_mid')
    blue_adc = st.selectbox("ADC (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='blue_adc')
    blue_p_adc = st.text_input("ADC player (optional)", key='blue_p_adc')
    blue_sup = st.selectbox("Support (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='blue_sup')
    blue_p_sup = st.text_input("Support player (optional)", key='blue_p_sup')

with col2:
    st.markdown("### 🔴 Red Side")
    prev_red_team = st.session_state.get('_prev_red_team', None)
    red_team = st.selectbox(
        "Team (optional)", options=[None] + all_teams,
        format_func=lambda x: "— no team —" if x is None else x,
        key='red_team')
    if red_team != prev_red_team:
        st.session_state['_prev_red_team'] = red_team
        if red_team and red_team in team_lineups:
            autofill_players(team_lineups[red_team], 'red')

    red_top  = st.selectbox("Top (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='red_top')
    red_p_top = st.text_input("Top player (optional)", key='red_p_top')
    red_jg   = st.selectbox("Jungle (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='red_jg')
    red_p_jg = st.text_input("Jungle player (optional)", key='red_p_jg')
    red_mid  = st.selectbox("Mid (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='red_mid')
    red_p_mid = st.text_input("Mid player (optional)", key='red_p_mid')
    red_adc  = st.selectbox("ADC (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='red_adc')
    red_p_adc = st.text_input("ADC player (optional)", key='red_p_adc')
    red_sup  = st.selectbox("Support (optional)", options=[None] + all_champs,
        format_func=lambda x: "— no pick —" if x is None else x, key='red_sup')
    red_p_sup = st.text_input("Support player (optional)", key='red_p_sup')

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

    blue_team_name = blue_team if blue_team else "Blue Team"
    red_team_name  = red_team  if red_team  else "Red Team"

    picked = [c for c in blue + red if c]
    if len(picked) != len(set(picked)) and len(picked) > 0:
        st.error("Each champion must be unique across both teams!")
    elif blue_team and red_team and blue_team == red_team:
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
            b_win_enc = pd.DataFrame(
                [[0] * len(win_mlb.classes_)],
                columns=['blue_' + c for c in win_mlb.classes_])

        if len(red) == 5:
            r_win_enc = pd.DataFrame(win_mlb.transform([red]),
                columns=['red_' + c for c in win_mlb.classes_])
        else:
            r_win_enc = pd.DataFrame(
                [[0] * len(win_mlb.classes_)],
                columns=['red_' + c for c in win_mlb.classes_])

        b_wr    = win_team_rate.get(blue_team, 0.5)  if blue_team else 0.5
        r_wr    = win_team_rate.get(red_team,  0.5)  if red_team  else 0.5
        b_games = win_team_games.get(blue_team, 0)   if blue_team else 0
        r_games = win_team_games.get(red_team,  0)   if red_team  else 0

        b_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in blue) / len(blue) \
                     if blue else 0.5
        r_champ_wr = sum(win_champ_rate.get(c, 0.5) for c in red)  / len(red)  \
                     if red  else 0.5

        win_h2h_r  = get_h2h_rate(win_h2h, blue_team, red_team) \
                     if blue_team and red_team else 0.5
        b_form     = get_form(win_team_recent, blue_team) if blue_team else 0.5
        r_form     = get_form(win_team_recent, red_team)  if red_team  else 0.5
        h2h_total  = get_h2h_total(win_h2h, blue_team, red_team) \
                     if blue_team and red_team else 0
        b_win_h2h, r_win_h2h = get_h2h_record(win_h2h, blue_team, red_team) \
                                if blue_team and red_team else (0, 0)

        if blue and blue_players:
            b_pc_avg = sum(pc_rate.get((p.strip(), c.strip()), 0.5)
                           for p, c in zip(blue_players, blue)
                           if p.strip() != '') / max(len(blue), 1)
        else:
            b_pc_avg = 0.5

        if red and red_players:
            r_pc_avg = sum(pc_rate.get((p.strip(), c.strip()), 0.5)
                           for p, c in zip(red_players, red)
                           if p.strip() != '') / max(len(red), 1)
        else:
            r_pc_avg = 0.5

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
            b_ft5_enc = pd.DataFrame(
                [[0] * len(ft5_mlb.classes_)],
                columns=['blue_' + c for c in ft5_mlb.classes_])

        if len(red) == 5:
            r_ft5_enc = pd.DataFrame(ft5_mlb.transform([red]),
                columns=['red_' + c for c in ft5_mlb.classes_])
        else:
            r_ft5_enc = pd.DataFrame(
                [[0] * len(ft5_mlb.classes_)],
                columns=['red_' + c for c in ft5_mlb.classes_])

        b_agg        = sum(champ_aggression.get(c, 0.5) for c in blue) / len(blue) \
                       if blue else 0.5
        r_agg        = sum(champ_aggression.get(c, 0.5) for c in red)  / len(red)  \
                       if red  else 0.5
        b_early      = team_early_rate.get(blue_team, 0.5)  if blue_team else 0.5
        r_early      = team_early_rate.get(red_team,  0.5)  if red_team  else 0.5
        b_speed      = team_kill_speed.get(blue_team, 10.0) if blue_team else 10.0
        r_speed      = team_kill_speed.get(red_team,  10.0) if red_team  else 10.0
        ft5_h2h_r    = get_h2h_rate(ft5_h2h, blue_team, red_team) \
                       if blue_team and red_team else 0.5
        ft5_h2h_tot  = get_h2h_total(ft5_h2h, blue_team, red_team) \
                       if blue_team and red_team else 0
        b_ft5_h2h, r_ft5_h2h = get_h2h_record(ft5_h2h, blue_team, red_team) \
                                if blue_team and red_team else (0, 0)
        b_early_form = get_form(ft5_team_recent, blue_team) if blue_team else 0.5
        r_early_form = get_form(ft5_team_recent, red_team)  if red_team  else 0.5

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

        win_blue_edge, win_blue_units, win_blue_label, win_blue_impl = calc_edge(blue_win_conf, win_blue_odds)
        win_red_edge,  win_red_units,  win_red_label,  win_red_impl  = calc_edge(red_win_conf,  win_red_odds)
        ft5_blue_edge, ft5_blue_units, ft5_blue_label, ft5_blue_impl = calc_edge(blue_ft5_conf, ft5_blue_odds)
        ft5_red_edge,  ft5_red_units,  ft5_red_label,  ft5_red_impl  = calc_edge(red_ft5_conf,  ft5_red_odds)

        win_winner  = blue_team_name if blue_win_conf > red_win_conf else red_team_name
        ft5_winner  = blue_team_name if blue_ft5_conf > red_ft5_conf else red_team_name
        faster_team = blue_team_name if b_speed < r_speed else red_team_name
        est_time    = (b_speed + r_speed) / 2
        positions   = ['Top', 'Jng', 'Mid', 'ADC', 'Sup']

        st.divider()

        # Team stats
        st.markdown("### 📋 Team Stats")
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(f"**🔵 {blue_team_name}**")
            st.write(f"Win rate: {b_wr*100:.1f}%")
            st.write(f"Form (L5): {b_form*100:.0f}%")
            st.write(f"Early game rate: {b_early*100:.1f}%")
            st.write(f"Avg kill time: {b_speed:.1f} min")
        with sc2:
            st.markdown(f"**🔴 {red_team_name}**")
            st.write(f"Win rate: {r_wr*100:.1f}%")
            st.write(f"Form (L5): {r_form*100:.0f}%")
            st.write(f"Early game rate: {r_early*100:.1f}%")
            st.write(f"Avg kill time: {r_speed:.1f} min")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.write(f"Win H2H: {blue_team_name} {b_win_h2h} — {r_win_h2h} {red_team_name}")
        with hc2:
            st.write(f"Early H2H: {blue_team_name} {b_ft5_h2h} — {r_ft5_h2h} {red_team_name}")

        st.divider()

        # Win signals
        st.markdown("### 📊 Win Signal Breakdown")
        show_signal("Team win rate",    b_wr,       r_wr,       0.05, 0.15, blue_team_name, red_team_name)
        show_signal("Recent form",      b_form,     r_form,     0.10, 0.25, blue_team_name, red_team_name, ".0f")
        show_signal("Champion quality", b_champ_wr, r_champ_wr, 0.02, 0.06, blue_team_name, red_team_name)
        show_signal("Player-champ wr",  b_pc_avg,   r_pc_avg,   0.03, 0.08, blue_team_name, red_team_name)

        if b_win_h2h + r_win_h2h > 0:
            h2h_diff = win_h2h_r - 0.5
            if abs(h2h_diff) >= 0.25:   h_str = "🟢 Strong"
            elif abs(h2h_diff) >= 0.10: h_str = "🟡 Moderate"
            else:                       h_str = "⚪ Weak"
            direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 \
                        else f"favours 🔴 {red_team_name}"
            st.write(f"**H2H record:** 🔵 {b_win_h2h} - {r_win_h2h} 🔴 — {h_str} {direction}")
        else:
            st.write(f"**H2H record:** No history — ⚪ Neutral")
        st.write(f"**Blue side advantage:** 53.1% historical — ⚪ Slight edge 🔵 {blue_team_name}")

        if blue:
            st.markdown(f"#### 🔵 {blue_team_name} Champion Ratings")
            for i, champ in enumerate(blue):
                player = blue_players[i] if i < len(blue_players) else ''
                cwr    = win_champ_rate.get(champ, 0.5)
                pcr    = pc_rate.get((player.strip(), champ.strip()), 0.5) \
                         if player.strip() else cwr
                pcg    = pc_games.get((player.strip(), champ.strip()), 0) \
                         if player.strip() else 0
                rating = rate_champ(cwr, pcr)
                pos    = positions[i] if i < len(positions) else ''
                gstr   = f"({pcg} games)" if pcg > 0 else "(no player data)"
                name   = player if player.strip() else "Unknown"
                st.write(f"**{pos}** {name} — {champ}: "
                         f"champ {cwr*100:.0f}% | player {pcr*100:.0f}% {gstr} → {rating}")

        if red:
            st.markdown(f"#### 🔴 {red_team_name} Champion Ratings")
            for i, champ in enumerate(red):
                player = red_players[i] if i < len(red_players) else ''
                cwr    = win_champ_rate.get(champ, 0.5)
                pcr    = pc_rate.get((player.strip(), champ.strip()), 0.5) \
                         if player.strip() else cwr
                pcg    = pc_games.get((player.strip(), champ.strip()), 0) \
                         if player.strip() else 0
                rating = rate_champ(cwr, pcr)
                pos    = positions[i] if i < len(positions) else ''
                gstr   = f"({pcg} games)" if pcg > 0 else "(no player data)"
                name   = player if player.strip() else "Unknown"
                st.write(f"**{pos}** {name} — {champ}: "
                         f"champ {cwr*100:.0f}% | player {pcr*100:.0f}% {gstr} → {rating}")

        st.divider()

        # Match winner
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

        st.divider()

        # FT5 signals
        st.markdown("### 📊 FT5 Signal Breakdown")
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
            h2h_diff = ft5_h2h_r - 0.5
            if abs(h2h_diff) >= 0.25:   h_str = "🟢 Strong"
            elif abs(h2h_diff) >= 0.10: h_str = "🟡 Moderate"
            else:                       h_str = "⚪ Weak"
            direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 \
                        else f"favours 🔴 {red_team_name}"
            st.write(f"**Early H2H:** 🔵 {b_ft5_h2h} - {r_ft5_h2h} 🔴 — {h_str} {direction}")
        else:
            st.write(f"**Early H2H:** No history — ⚪ Neutral")

        if blue:
            st.markdown(f"#### 🔵 {blue_team_name} Champion Aggression")
            for i, champ in enumerate(blue):
                player = blue_players[i] if i < len(blue_players) else ''
                agg    = champ_aggression.get(champ, 0.5)
                rating = rate_agg(agg)
                pos    = positions[i] if i < len(positions) else ''
                name   = player if player.strip() else "Unknown"
                st.write(f"**{pos}** {name} — {champ}: aggression {agg*100:.0f}% → {rating}")

        if red:
            st.markdown(f"#### 🔴 {red_team_name} Champion Aggression")
            for i, champ in enumerate(red):
                player = red_players[i] if i < len(red_players) else ''
                agg    = champ_aggression.get(champ, 0.5)
                rating = rate_agg(agg)
                pos    = positions[i] if i < len(positions) else ''
                name   = player if player.strip() else "Unknown"
                st.write(f"**{pos}** {name} — {champ}: aggression {agg*100:.0f}% → {rating}")

        st.divider()

        # First to five
        st.markdown("### ⚔️ First to Five Kills")
        ft5_color = "🔵" if blue_ft5_conf > red_ft5_conf else "🔴"
        st.markdown(f"#### {ft5_color} Model pick: **{ft5_winner}**")
        st.caption(f"⏱️ Est. 5 kills at ~minute {est_time:.1f} ({faster_team} historically faster)")
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

        st.divider()

        # Claude reasoning — only show if both teams and all picks are entered
        if blue_team and red_team and len(blue) == 5 and len(red) == 5:
            st.markdown("### 🤖 AI Analysis")
            with st.spinner("Generating analysis..."):
                reasoning = get_claude_reasoning(
                    blue_team_name, red_team_name,
                    blue, red,
                    blue_players, red_players,
                    blue_win_conf, red_win_conf,
                    blue_ft5_conf, red_ft5_conf,
                    b_wr, r_wr, b_form, r_form,
                    b_champ_wr, r_champ_wr,
                    b_pc_avg, r_pc_avg,
                    b_win_h2h, r_win_h2h,
                    b_agg, r_agg,
                    b_early, r_early,
                    b_speed, r_speed,
                    win_blue_odds, win_red_odds,
                    ft5_blue_odds, ft5_red_odds)
            st.write(reasoning)
            st.divider()

        st.caption("~61.88% true accuracy | Trust 65%+ | Best ROI at 2.30+ odds")
