import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import pickle
import requests
import difflib
import re
import json
from datetime import datetime

st.set_page_config(
    page_title="LoL Match Predictor",
    page_icon="🎮",
    layout="centered"
)

st.markdown("""
<style>
  header[data-testid="stHeader"] { display: none !important; }
  #MainMenu { display: none !important; }
  footer { display: none !important; }
  .stDeployButton { display: none !important; }
  div[data-testid="stToolbar"] { display: none !important; }
  .stApp { background: #0a0c10; color: #d0d8f0; }
  .block-container { padding-top: 0.75rem !important; }
  hr { border-color: #1e2535 !important; }
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stNumberInput > div > div > input {
    background: #0f1218 !important;
    border: 1px solid #2a3050 !important;
    color: #d0d8f0 !important;
    border-radius: 6px !important;
    outline: none !important;
    box-shadow: none !important;
    font-family: 'SF Mono','Fira Code','Consolas',monospace !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus,
  .stNumberInput > div > div > input:focus {
    border-color: #3a6a30 !important;
    box-shadow: 0 0 0 2px rgba(80,160,40,0.15) !important;
    outline: none !important;
  }
  .stTextInput > div > div, .stTextArea > div > div, .stNumberInput > div > div {
    border: none !important; box-shadow: none !important; background: transparent !important;
  }
  .stNumberInput > div > div > div { background: #0f1218 !important; }
  button[data-testid="stNumberInputStepDown"],
  button[data-testid="stNumberInputStepUp"] {
    background: #0f1218 !important; border-color: #2a3050 !important; color: #4a6a30 !important;
  }
  button[data-testid="stNumberInputStepDown"]:hover,
  button[data-testid="stNumberInputStepUp"]:hover {
    background: #1a2a10 !important; color: #80d040 !important;
  }
  div[data-baseweb="base-input"] { background: #0f1218 !important; }
  div[data-baseweb="input"] { background: #0f1218 !important; border-color: #2a3050 !important; }
  label { color: #3a4a6a !important; font-size: 0.78rem !important;
          font-family: 'SF Mono','Fira Code','Consolas',monospace !important;
          text-transform: uppercase !important; letter-spacing: 0.08em !important; }
  div[data-testid="stButton"] > button[kind="primary"] {
    background: #0d1f05 !important; border: 1px solid #2a5a10 !important;
    color: #80d040 !important; font-family: 'SF Mono','Fira Code',monospace !important;
    font-weight: 700 !important; border-radius: 5px !important; letter-spacing: 0.06em !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"] {
    background: #0f1218 !important; border: 1px solid #1e2535 !important;
    color: #4a5a7a !important; font-family: 'SF Mono','Fira Code',monospace !important;
    border-radius: 5px !important;
  }
  div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #3a4a6a !important; color: #8090b0 !important;
  }
  div[data-testid="stExpander"] {
    background: #0f1218 !important; border: 1px solid #1e2535 !important; border-radius: 6px !important;
  }
  div[data-testid="stExpander"] summary {
    color: #4a5a7a !important; font-family: 'SF Mono','Fira Code',monospace !important;
  }
  div[data-testid="metric-container"] {
    background: #0f1218; border: 1px solid #1e2535; border-radius: 6px; padding: 0.6rem 0.8rem;
  }
  div[data-testid="metric-container"] label { color: #3a4060 !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #c0f060 !important; font-size: 1.4rem !important;
    font-family: 'SF Mono','Fira Code',monospace !important;
  }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] svg { display:none; }
  .stCheckbox label { color: #4a5a7a !important; text-transform: none !important; letter-spacing: 0 !important; }
  p, .stMarkdown p { color: #8090b0; font-family: 'SF Mono','Fira Code',monospace; font-size: 0.82rem; }
  h1,h2,h3 { font-family: 'SF Mono','Fira Code',monospace !important; color: #c0f060 !important; }
  small, .stCaption { color: #3a4a6a !important; font-family: 'SF Mono','Fira Code',monospace !important; }
  div[data-testid="stTextInput"] > div > div { padding: 4px 8px !important; min-height: 0 !important; height: 30px !important; }
  .stTextArea > div > div > textarea { padding: 4px 8px !important; }
  .stNumberInput > div > div > input { padding: 4px 6px !important; height: 30px !important; }
  .stNumberInput button { height: 30px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown('''
<div style="border-bottom:1px solid #1e2535;padding-bottom:10px;margin-bottom:8px;">
  <span style="color:#c0f060;font-size:1.3rem;font-weight:700;font-family:'SF Mono',monospace;letter-spacing:0.06em;">&#9672; LOL MATCH PREDICTOR v8</span>
  <span style="color:#3a4a6a;font-size:0.72rem;font-family:'SF Mono',monospace;margin-left:12px;">Win ~67.09% / AUC 0.7227 &middot; FT5 57.16% &middot; Gold trajectory + Grid search params</span>
</div>
''', unsafe_allow_html=True)

FORM_WINDOW       = 8
RECENT_WINDOW     = 20
RECENT_WEIGHT     = 0.2
BLUE_SIDE_WINRATE = 0.5312
DISCORD_USER_ID   = "465715665584652315"

TEAM_ALIASES = {
    'Team BDS': 'Team Shifters',
    'BDS':      'Team Shifters',
}

LEAGUE_MAP = {
    'T1': 'LCK', 'Gen.G': 'LCK', 'Hanwha Life Esports': 'LCK', 'Dplus KIA': 'LCK',
    'KT Rolster': 'LCK', 'DRX': 'LCK', 'BNK FEARX': 'LCK', 'DN SOOPers': 'LCK',
    'Nongshim RedForce': 'LCK', 'HANJIN BRION': 'LCK',
    'G2 Esports': 'LEC', 'Fnatic': 'LEC', 'Team Vitality': 'LEC', 'Karmine Corp': 'LEC',
    'Team Heretics': 'LEC', 'Movistar KOI': 'LEC', 'GIANTX': 'LEC', 'SK Gaming': 'LEC',
    'Natus Vincere': 'LEC', 'Team Shifters': 'LEC',
    'Cloud9': 'LCS', 'Team Liquid': 'LCS', 'FlyQuest': 'LCS', 'Sentinels': 'LCS',
    'Shopify Rebellion': 'LCS', 'Dignitas': 'LCS', 'LYON': 'LCS', 'Disguised': 'LCS',
    'RED Canids': 'CBLOL', 'Fluxo W7M': 'CBLOL', 'FURIA': 'CBLOL', 'Keyd Stars': 'CBLOL',
    'LOUD': 'CBLOL', 'paiN Gaming': 'CBLOL', 'Leviatán': 'CBLOL', 'LOS': 'CBLOL',
    "Anyone's Legend": 'LPL', 'Bilibili Gaming': 'LPL', 'JD Gaming': 'LPL',
    'Top Esports': 'LPL', 'Weibo Gaming': 'LPL', 'Invictus Gaming': 'LPL',
    'EDward Gaming': 'LPL', 'Ninjas in Pyjamas': 'LPL', 'Team WE': 'LPL',
    'ThunderTalk Gaming': 'LPL', 'LGD Gaming': 'LPL', 'LNG Esports': 'LPL',
    'Oh My God': 'LPL', 'Ultra Prime': 'LPL',
    'FunPlus Phoenix': 'LPL', 'Rare Atom': 'LPL', 'Wolves Esports': 'LPL',
}

def normalize_team(name):
    return TEAM_ALIASES.get(str(name), str(name))

def get_league(team_name):
    if not team_name: return ''
    return LEAGUE_MAP.get(team_name, '')

def conf_short(level):
    if 'HIGH'   in level: return 'High'
    if 'MEDIUM' in level: return 'Med'
    return 'Low'

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
        score += 3; reasons.append(f"Strong team history ({min_games}+ games each)")
    elif min_games >= 20:
        score += 2; reasons.append(f"Decent team history ({min_games}+ games each)")
    elif min_games >= 10:
        score += 1; reasons.append(f"Limited team history ({min_games} games)")
    else:
        warnings_list.append(f"Very little team data ({min_games} games)")
    if h2h_total >= 10:
        score += 3; reasons.append(f"Strong H2H record ({h2h_total} games)")
    elif h2h_total >= 5:
        score += 2; reasons.append(f"Some H2H history ({h2h_total} games)")
    elif h2h_total >= 2:
        score += 1; reasons.append(f"Limited H2H ({h2h_total} games)")
    else:
        warnings_list.append("No H2H history — using defaults")
    signals = [
        1 if winrate_diff > 0.05 else (-1 if winrate_diff < -0.05 else 0),
        1 if form_diff > 0.1    else (-1 if form_diff    < -0.1  else 0),
        1 if champ_diff > 0.02  else (-1 if champ_diff   < -0.02 else 0),
    ]
    non_zero = [s for s in signals if s != 0]
    if len(non_zero) >= 2:
        if all(s == non_zero[0] for s in non_zero):
            score += 3; reasons.append("All signals agree")
        else:
            score += 1; warnings_list.append("Mixed signals")
    else:
        score += 1; warnings_list.append("Weak signal strength")
    if score >= 7:   level, desc = "🟢 HIGH",   "Strong data, clear favourite"
    elif score >= 4: level, desc = "🟡 MEDIUM", "Reasonable data, some uncertainty"
    else:            level, desc = "🔴 LOW",    "Limited data or conflicting signals"
    return level, desc, reasons, warnings_list

def ft5_confidence(blue_prob, b_early_rate, r_early_rate,
                   b_kill_speed, r_kill_speed, b_agg, r_agg,
                   h2h_total, b_early_form, r_early_form):
    """
    FT5 confidence calibrated to BETTING VALUE from backtest:

    Backtest findings:
    - Strong red signal (blue_prob < 0.48): 61% accuracy but 7.2% ROI
      → good for accuracy, poor for value due to odds compression
    - LOW conf + 2.30+ odds: 39.6% ROI — best betting value
    - Overall model edge: +4.04% over always-blue (53.3% baseline)

    Confidence reflects expected betting value, not raw model accuracy.
    HIGH = best ROI conditions | MEDIUM = decent value | LOW = weak signal
    """
    reasons  = []
    warnings = []

    # Strong red signal — highest accuracy but note odds compression
    if blue_prob < 0.48:
        red_prob = 1 - blue_prob
        reasons.append(f"Strong red signal — model blue conf {blue_prob*100:.1f}% (below 48%)")
        reasons.append("Backtest: 61% red accuracy across 123 games")
        warnings.append("⚠️ Check odds — red signal games often get compressed odds (7.2% ROI backtest)")
        level = "🟡 MEDIUM"
        desc  = "High accuracy red pick — verify odds offer value before betting"
        return level, desc, reasons, warnings

    # Model signal strength beyond baseline
    deviation = blue_prob - 0.53  # distance from 53% blue baseline
    blue_pick = blue_prob >= 0.50

    # Kill speed — most direct signal
    speed_gap = 0
    if b_kill_speed > 0 and r_kill_speed > 0:
        speed_gap = abs(b_kill_speed - r_kill_speed)
        if speed_gap >= 3:
            faster = "Blue" if b_kill_speed < r_kill_speed else "Red"
            reasons.append(f"{faster} gets kills {speed_gap:.1f} min faster historically")
        elif speed_gap >= 1.5:
            reasons.append("Moderate kill speed gap between teams")
        else:
            warnings.append("Similar kill speeds — FT5 hard to call")

    # Early rate gap
    early_diff = abs(b_early_rate - r_early_rate)
    if early_diff >= 0.15:
        faster = "Blue" if b_early_rate > r_early_rate else "Red"
        reasons.append(f"{faster} wins FT5 races significantly more often")
    elif early_diff >= 0.08:
        reasons.append("Moderate early rate advantage")
    else:
        warnings.append("Teams have similar FT5 win rates")

    # Aggression
    agg_diff = abs(b_agg - r_agg)
    if agg_diff >= 0.08:
        more_agg = "Blue" if b_agg > r_agg else "Red"
        reasons.append(f"{more_agg} comp meaningfully more aggressive")
    elif agg_diff < 0.03:
        warnings.append("Similar comp aggression")

    # H2H
    if h2h_total >= 8:
        reasons.append(f"Good early H2H sample ({h2h_total} games)")
    elif h2h_total < 3:
        warnings.append("Limited H2H early game data")

    n_reasons  = len(reasons)
    n_warnings = len(warnings)

    # Calibrate to ROI findings:
    # Best ROI comes from 2.30+ odds + any edge — flag when model has clear signal
    # at high odds that's where 39.6% ROI comes from
    strong_signal  = deviation > 0.10 or deviation < -0.10  # clear lean from baseline
    medium_signal  = 0.05 < abs(deviation) <= 0.10
    supporting_ev  = n_reasons >= 2 and n_warnings <= 1

    if strong_signal and supporting_ev:
        level = "🟢 HIGH"
        desc  = "Strong model signal with supporting evidence — best value at 2.30+ odds"
    elif strong_signal or (medium_signal and supporting_ev):
        level = "🟡 MEDIUM"
        desc  = "Moderate signal — look for 2.30+ odds for best ROI"
    else:
        level = "🔴 LOW"
        desc  = "Weak FT5 signal — near baseline (53% blue), only bet with strong odds edge"
        if not warnings:
            warnings.append("Model near always-blue baseline — limited predictive value")

    return level, desc, reasons, warnings


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
pc_games_d       = p['pc_games']
role_champ_rate  = p['role_champ_rate']
ft5_model        = p['ft5_model']
ft5_mlb          = p['ft5_mlb']
champ_aggression = p['champ_aggression']
team_early_rate  = p['team_early_rate']
team_kill_speed      = p['team_kill_speed']
team_avg_gamelength  = p.get('team_avg_gamelength', {})
team_avg_kills       = p.get('team_avg_kills', {})
ft5_h2h          = p['ft5_h2h']
ft5_team_recent  = p['ft5_team_recent']
ft5_team_games   = p['ft5_team_games']
team_lineups     = p['team_lineups']
all_teams        = p['all_teams']
all_champs       = p['all_champs']
PC_WEIGHT        = p.get('pc_weight', 0.10)
RC_WEIGHT        = p.get('rc_weight', 0.90)
H2H_CAP          = p.get('h2h_cap',  0.60)
gold_lookup      = p.get('gold_lookup', {})
GOLD_WINDOW      = p.get('gold_window', 15)

POSITIONS  = ['top', 'jng', 'mid', 'adc', 'sup']
POS_LABELS = ['Top', 'Jng', 'Mid', 'ADC', 'Sup']

CHAMP_LOOKUP = {
    c.lower().replace("'","").replace(" ","").replace("&","").replace(".","").replace("-",""): c
    for c in all_champs
}

def fuzzy_match_champion(raw, threshold=0.6):
    if not raw or not raw.strip(): return None
    clean = raw.strip().lower().replace("'","").replace(" ","").replace("&","").replace(".","").replace("-","")
    if clean in CHAMP_LOOKUP: return CHAMP_LOOKUP[clean]
    matches = difflib.get_close_matches(clean, CHAMP_LOOKUP.keys(), n=1, cutoff=threshold)
    return CHAMP_LOOKUP[matches[0]] if matches else None

def fuzzy_match_team(raw):
    if not raw or not raw.strip(): return None
    raw_strip = raw.strip()
    normed = normalize_team(raw_strip)
    if normed != raw_strip and normed in all_teams: return normed
    for t in all_teams:
        if raw_strip.lower() == t.lower(): return t
    raw_clean = raw_strip.lower().replace(" ","")
    team_map  = {t.lower().replace(" ",""): t for t in all_teams}
    matches   = difflib.get_close_matches(raw_clean, team_map.keys(), n=1, cutoff=0.7)
    return team_map[matches[0]] if matches else None

def parse_champion_input(text):
    if not text or not text.strip(): return []
    parts = re.split(r'[\t,]+', text.strip())
    if len(parts) < 5: parts = re.split(r'\s+', text.strip())
    champs = []
    i = 0
    while i < len(parts) and len(champs) < 5:
        word = parts[i].strip()
        if not word: i += 1; continue
        match = fuzzy_match_champion(word)
        if match: champs.append(match); i += 1; continue
        if i + 1 < len(parts):
            m2 = fuzzy_match_champion(word + " " + parts[i+1].strip())
            if m2: champs.append(m2); i += 2; continue
        if i + 2 < len(parts):
            m3 = fuzzy_match_champion(word + " " + parts[i+1].strip() + " " + parts[i+2].strip())
            if m3: champs.append(m3); i += 3; continue
        i += 1
    return champs

st.markdown('<span style="color:#3a6a20;font-size:0.75rem;font-family:monospace;">&#9654; MODELS LOADED</span>', unsafe_allow_html=True)

defaults = {
    'blue_team_input': '', 'red_team_input': '',
    'blue_comp_input': '', 'red_comp_input': '',
    'blue_p_top': '', 'blue_p_jg': '', 'blue_p_mid': '',
    'blue_p_adc': '', 'blue_p_sup': '',
    'red_p_top':  '', 'red_p_jg':  '', 'red_p_mid':  '',
    'red_p_adc':  '', 'red_p_sup':  '',
    'game_number': '',
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
    h = recent[-FORM_WINDOW:]
    if not h: return 0.5
    weights = list(range(1, len(h) + 1))
    return sum(v * w for v, w in zip(h, weights)) / sum(weights)

def get_recent_wr(recent_dict, team):
    recent = recent_dict.get(team, [])
    h = recent[-RECENT_WINDOW:]
    return sum(h) / len(h) if h else 0.5

def get_gold_features(team_name, match_date=None):
    """Get avg_gd20 and late_scaling from gold_lookup using today's date as fallback."""
    if not team_name: return 0.0, 0.0
    # Try today first, then scan recent entries for this team
    from datetime import datetime, timedelta
    if match_date is None:
        match_date = datetime.now().strftime('%Y-%m-%d')
    # Try exact date, then walk back up to 30 days
    dt = datetime.strptime(match_date, '%Y-%m-%d')
    for i in range(30):
        ds = (dt - timedelta(days=i)).strftime('%Y-%m-%d')
        key = (ds, team_name)
        if key in gold_lookup:
            entry = gold_lookup[key]
            return entry.get('avg_gd20', 0.0), entry.get('late_scaling', 0.0)
    return 0.0, 0.0

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

def show_signal(label, b_val, r_val, low_t, high_t, blue_name, red_name, fmt=".1f"):
    diff     = b_val - r_val
    abs_diff = abs(diff)
    if abs_diff >= high_t:   strength = "🟢 Strong"
    elif abs_diff >= low_t:  strength = "🟡 Moderate"
    else:                    strength = "⚪ Weak"
    direction = f"favours 🔵 {blue_name}" if diff > 0 else \
                (f"favours 🔴 {red_name}" if diff < 0 else "even")
    st.write(f"**{label}:** 🔵 {b_val*100:{fmt}}% vs 🔴 {r_val*100:{fmt}}% — {strength} {direction}")

def get_blended_avg(players, picks):
    rates = []
    for i, (player, champ) in enumerate(zip(players, picks)):
        pos    = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
        pc_key = (player.strip(), champ.strip())
        rc_key = (pos, champ.strip())
        pc_val = pc_rate.get(pc_key, 0.5) if player.strip() else 0.5
        rc_val = role_champ_rate.get(rc_key, 0.5)
        rates.append(PC_WEIGHT * pc_val + RC_WEIGHT * rc_val)
    return sum(rates) / len(rates) if rates else 0.5

def get_draft_only_prediction(blue, red, b_champ_wr, r_champ_wr, b_pc_avg, r_pc_avg):
    b_win_enc = pd.DataFrame(win_mlb.transform([blue]),
        columns=['blue_' + c for c in win_mlb.classes_])
    r_win_enc = pd.DataFrame(win_mlb.transform([red]),
        columns=['red_'  + c for c in win_mlb.classes_])
    neutral_row = pd.DataFrame([[
        0.5, 0.5, 0.0, 50, 50,
        b_champ_wr, r_champ_wr, b_champ_wr - r_champ_wr,
        0.5, 0.5, 0.5, 0.0,
        0.5, 0.5, 0.0,
        BLUE_SIDE_WINRATE,
        b_pc_avg, r_pc_avg, b_pc_avg - r_pc_avg,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]], columns=[
        'blue_team_winrate','red_team_winrate','team_winrate_diff',
        'blue_team_games','red_team_games',
        'blue_avg_winrate','red_avg_winrate','winrate_diff',
        'h2h_winrate','blue_form','red_form','form_diff',
        'blue_recent_wr','red_recent_wr','recent_wr_diff',
        'blue_side_advantage','blue_pc_avg','red_pc_avg','pc_avg_diff',
        'blue_avg_gd20','red_avg_gd20','gd20_diff',
        'blue_late_scaling','red_late_scaling','late_scaling_diff',
    ])
    win_prob = win_model.predict_proba(pd.concat([b_win_enc, r_win_enc, neutral_row], axis=1))[0]
    b_ft5_enc = pd.DataFrame(ft5_mlb.transform([blue]),
        columns=['blue_' + c for c in ft5_mlb.classes_])
    r_ft5_enc = pd.DataFrame(ft5_mlb.transform([red]),
        columns=['red_'  + c for c in ft5_mlb.classes_])
    b_agg = sum(champ_aggression.get(c, 0.5) for c in blue) / len(blue)
    r_agg = sum(champ_aggression.get(c, 0.5) for c in red)  / len(red)
    neutral_ft5 = pd.DataFrame([[
        b_agg, r_agg, b_agg-r_agg,
        0.5, 0.5, 0.0, 10.0, 10.0, 0.0,
        0.5, 0.5, 0.5, 0.0,
    ]], columns=[
        'blue_aggression','red_aggression','aggression_diff',
        'blue_early_rate','red_early_rate','early_rate_diff',
        'blue_kill_speed','red_kill_speed','speed_diff',
        'h2h_early_rate','blue_early_form','red_early_form','early_form_diff',
    ])
    ft5_prob = ft5_model.predict_proba(pd.concat([b_ft5_enc, r_ft5_enc, neutral_ft5], axis=1))[0]
    # Clip to 5%-95%
    bw = min(max(win_prob[1], 0.05), 0.95)
    rw = min(max(win_prob[0], 0.05), 0.95)
    bf = min(max(ft5_prob[1], 0.05), 0.95)
    rf = min(max(ft5_prob[0], 0.05), 0.95)
    return bw, rw, bf, rf

def send_discord_dm(message):
    try:
        token   = st.secrets["DISCORD_BOT_TOKEN"]
        headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
        dm_resp = requests.post(
            "https://discord.com/api/v10/users/@me/channels",
            headers=headers, json={"recipient_id": DISCORD_USER_ID}, timeout=10)
        if dm_resp.status_code not in [200, 201]: return False
        channel_id = dm_resp.json()["id"]
        msg_resp = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers=headers, json={"content": message}, timeout=10)
        return msg_resp.status_code in [200, 201]
    except Exception:
        return False

def log_to_sheets(row_data, sheet_id):
    try:
        creds_json = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_info(
            creds_json,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service  = build("sheets", "v4", credentials=creds)
        result   = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range="Dashboard!A:A").execute()
        values   = result.get("values", [])
        next_row = max(9, len(values) + 1)
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"Dashboard!A{next_row}:J{next_row}",
            valueInputOption="USER_ENTERED",
            body={"values": [row_data]}
        ).execute()
        return True
    except Exception:
        return False

def fetch_tracker_history(conf, conf_level, sheet_id):
    """Pull completed rows from a tracker sheet and return W-L for matching band+confidence."""
    try:
        creds_json = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT"])
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        creds = service_account.Credentials.from_service_account_info(
            creds_json, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        service = build("sheets", "v4", credentials=creds)
        result  = service.spreadsheets().values().get(
            spreadsheetId=sheet_id, range="Dashboard!H9:K1000").execute()
        rows = result.get("values", [])

        # Columns: H=Bot Confidence, I=Model%, J=Odds, K=Result
        # Band is ±2.5% around the prediction (5% window)
        lo = conf - 0.025
        hi = conf + 0.025
        conf_short_map = {"High": "High", "Med": "Med", "Low": "Low"}
        target_conf = conf_short(conf_level)

        band_conf_w, band_conf_l = 0, 0  # same band + same confidence
        band_w,      band_l      = 0, 0  # same band only
        conf_w,      conf_l      = 0, 0  # same confidence only

        for r in rows:
            if len(r) < 4: continue
            raw_conf  = r[0].strip() if r[0] else ""
            raw_pct   = r[1].strip() if len(r) > 1 else ""
            result    = r[3].strip() if len(r) > 3 else ""
            if result not in ("Win", "Loss"): continue
            try:
                pct = float(raw_pct)
                if pct > 1: pct = pct / 100
            except:
                continue
            in_band = lo <= pct <= hi
            in_conf = raw_conf == target_conf
            won = result == "Win"
            if in_band and in_conf:
                if won: band_conf_w += 1
                else:   band_conf_l += 1
            if in_band:
                if won: band_w += 1
                else:   band_l += 1
            if in_conf:
                if won: conf_w += 1
                else:   conf_l += 1

        return {
            "band_conf": (band_conf_w, band_conf_l),
            "band":      (band_w, band_l),
            "conf":      (conf_w, conf_l),
            "target_conf": target_conf,
            "band_label": f"{lo*100:.0f}%-{hi*100:.0f}%",
        }
    except Exception:
        return None

def format_history(data, label):
    """Format tracker history into a readable string."""
    if not data:
        return "📊 No tracker data available"

    parts = []

    bc_w, bc_l = data["band_conf"]
    bc_total = bc_w + bc_l
    if bc_total >= 3:
        bc_pct = bc_w / bc_total * 100
        emoji  = "🟢" if bc_pct >= 65 else ("🟡" if bc_pct >= 50 else "🔴")
        parts.append(f"{emoji} **{data['band_label']} + {data['target_conf']} conf:** {bc_w}W-{bc_l}L ({bc_pct:.0f}%) — {bc_total} games")
    elif bc_total > 0:
        parts.append(f"⚪ **{data['band_label']} + {data['target_conf']} conf:** {bc_w}W-{bc_l}L — only {bc_total} game{'s' if bc_total > 1 else ''}, too small")
    else:
        parts.append(f"⚪ **{data['band_label']} + {data['target_conf']} conf:** No completed games yet")

    b_w, b_l = data["band"]
    b_total = b_w + b_l
    if b_total >= 5 and b_total != bc_total:
        b_pct  = b_w / b_total * 100
        emoji  = "🟢" if b_pct >= 65 else ("🟡" if b_pct >= 50 else "🔴")
        parts.append(f"{emoji} **{data['band_label']} (all confidence):** {b_w}W-{b_l}L ({b_pct:.0f}%) — {b_total} games")

    c_w, c_l = data["conf"]
    c_total = c_w + c_l
    if c_total >= 5:
        c_pct  = c_w / c_total * 100
        emoji  = "🟢" if c_pct >= 65 else ("🟡" if c_pct >= 50 else "🔴")
        parts.append(f"{emoji} **{data['target_conf']} confidence (all %):** {c_w}W-{c_l}L ({c_pct:.0f}%) — {c_total} games")

    return "\n".join(parts) if parts else "📊 Not enough data yet"

def get_claude_reasoning(
        blue_team, red_team, blue_picks, red_picks,
        blue_players, red_players,
        blue_win_conf, red_win_conf,
        blue_ft5_conf, red_ft5_conf,
        b_wr, r_wr, b_form, r_form,
        b_champ_wr, r_champ_wr, b_pc_avg, r_pc_avg,
        b_win_h2h, r_win_h2h,
        b_agg, r_agg, b_early, r_early, b_speed, r_speed,
        win_blue_odds, win_red_odds, ft5_blue_odds, ft5_red_odds):

    # Build roster descriptions focused on champion roles/synergies
    def describe_roster(players, picks, team):
        lines = []
        for i in range(len(picks)):
            player = players[i].strip() if i < len(players) and players[i].strip() else "Unknown"
            champ  = picks[i]
            pos    = POS_LABELS[i] if i < len(POS_LABELS) else ''
            lines.append(f"{pos}: {player} on {champ}")
        return ', '.join(lines)

    blue_roster = describe_roster(blue_players, blue_picks, blue_team)
    red_roster  = describe_roster(red_players,  red_picks,  red_team)

    # Determine comp styles based on archetypes
    def comp_style(picks):
        dive = sum(1 for c in picks if champ_aggression.get(c, 0.5) >= 0.58)
        poke = sum(1 for c in picks if champ_aggression.get(c, 0.5) <= 0.45)
        if dive >= 3: return "dive/engage heavy"
        if poke >= 3: return "poke/siege oriented"
        if dive >= 2 and poke == 0: return "dive with peel"
        return "teamfight/balanced"

    b_style = comp_style(blue_picks)
    r_style = comp_style(red_picks)

    # Gold trajectory context
    b_gd20, b_late = get_gold_features(blue_team)
    r_gd20, r_late = get_gold_features(red_team)

    def gold_context(team, gd20, late):
        if gd20 > 500:
            trend = "typically ahead at 20 min"
        elif gd20 < -500:
            trend = "typically behind at 20 min"
        else:
            trend = "even gold at 20 min"
        if late > 300:
            scaling = "and scales well into late game"
        elif late < -300:
            scaling = "but tends to fall off late"
        else:
            scaling = "with neutral late game scaling"
        return f"{trend} {scaling}"

    b_gold_ctx = gold_context(blue_team, b_gd20, b_late) if b_gd20 != 0 else "no gold trend data"
    r_gold_ctx = gold_context(red_team,  r_gd20, r_late) if r_gd20 != 0 else "no gold trend data"

    winner = blue_team if blue_win_conf > red_win_conf else red_team
    ft5_winner = blue_team if blue_ft5_conf > red_ft5_conf else red_team
    loser  = red_team  if blue_win_conf > red_win_conf else blue_team

    prompt = f"""You are an expert League of Legends pro play analyst. Analyze how this match should play out based on the draft and team styles. Focus on the game narrative — how teamfights will develop, which phase of the game each team wins, and what win conditions each team has. Do NOT mention win rates, H2H records, or champion mastery stats.

MATCH: {blue_team} (Blue) vs {red_team} (Red)

BLUE SIDE — {blue_team}
Draft: {blue_roster}
Comp style: {b_style}
Gold trend: {b_gold_ctx}
Early aggression: {'high' if b_agg > 0.55 else 'low' if b_agg < 0.48 else 'average'} | Avg first kill: {b_speed:.1f} min

RED SIDE — {red_team}
Draft: {red_roster}
Comp style: {r_style}
Gold trend: {r_gold_ctx}
Early aggression: {'high' if r_agg > 0.55 else 'low' if r_agg < 0.48 else 'average'} | Avg first kill: {r_speed:.1f} min

MODEL PREDICTION: {winner} wins ({max(blue_win_conf, red_win_conf)*100:.0f}%) | FT5: {ft5_winner} ({max(blue_ft5_conf, red_ft5_conf)*100:.0f}%)

Write two paragraphs:
1. MATCH FLOW (3-4 sentences): How should this game play out? Describe the draft interaction — does one comp counter the other, which phase of the game does each team win, what are the key win conditions? Explain why {winner} is favoured.
2. FIRST TO 5 KILLS (2-3 sentences): Which team gets first blood and early kills, and why? Describe the early game matchups and which comp forces early fights.
Be direct and specific. No bullet points. No statistics."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": st.secrets["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-sonnet-4-5", "max_tokens": 600,
                  "messages": [{"role": "user", "content": prompt}]},
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
        st.session_state['blue_team_input'], st.session_state['red_team_input'] = \
            st.session_state['red_team_input'], st.session_state['blue_team_input']
        st.session_state['blue_comp_input'], st.session_state['red_comp_input'] = \
            st.session_state['red_comp_input'], st.session_state['blue_comp_input']
        for pos in ['top', 'jg', 'mid', 'adc', 'sup']:
            st.session_state[f'blue_p_{pos}'], st.session_state[f'red_p_{pos}'] = \
                st.session_state[f'red_p_{pos}'], st.session_state[f'blue_p_{pos}']
        st.rerun()

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div style="color:#4a90d9;font-size:11px;font-weight:700;font-family:monospace;letter-spacing:0.08em;margin-bottom:4px;">&#9679; BLUE SIDE</div>', unsafe_allow_html=True)
    blue_team_raw   = st.text_input("Team name", key='blue_team_input',
                                     placeholder="e.g. T1, Gen.G, Cloud9...", label_visibility="collapsed")
    blue_team_match = fuzzy_match_team(blue_team_raw) if blue_team_raw else None
    if blue_team_match and blue_team_match in team_lineups:
        lineup = team_lineups[blue_team_match]
        if isinstance(lineup, dict):
            if not st.session_state['blue_p_top']: st.session_state['blue_p_top'] = lineup.get('top','')
            if not st.session_state['blue_p_jg']:  st.session_state['blue_p_jg']  = lineup.get('jng','')
            if not st.session_state['blue_p_mid']: st.session_state['blue_p_mid'] = lineup.get('mid','')
            if not st.session_state['blue_p_adc']: st.session_state['blue_p_adc'] = lineup.get('adc','')
            if not st.session_state['blue_p_sup']: st.session_state['blue_p_sup'] = lineup.get('sup','')
    if blue_team_raw and blue_team_match:
        st.markdown(f'<div style="color:#3a6a20;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#10003; {blue_team_match}</div>', unsafe_allow_html=True)
    elif blue_team_raw:
        st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#9900; unknown — using averages</div>', unsafe_allow_html=True)
    blue_comp_raw = st.text_area("Champions", key='blue_comp_input',
                                  placeholder="e.g. Gnar Nocturne Ahri Caitlyn Bard",
                                  height=58, label_visibility="collapsed")
    blue_parsed = parse_champion_input(blue_comp_raw)
    if blue_comp_raw:
        if len(blue_parsed) == 5:
            st.markdown(f'<div style="color:#3a6a20;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#10003; {' &middot; '.join([f'{POS_LABELS[i]}: {blue_parsed[i]}' for i in range(5)])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#8a6020;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#9888; {len(blue_parsed)}/5 parsed</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;letter-spacing:0.08em;margin:4px 0 2px;">PLAYERS (optional)</div>', unsafe_allow_html=True)
    pb1, pb2, pb3 = st.columns(3)
    with pb1:
        blue_p_top = st.text_input("Top", key='blue_p_top', placeholder="Top")
        blue_p_adc = st.text_input("ADC", key='blue_p_adc', placeholder="ADC")
    with pb2:
        blue_p_jg  = st.text_input("Jng", key='blue_p_jg',  placeholder="Jng")
        blue_p_sup = st.text_input("Sup", key='blue_p_sup', placeholder="Sup")
    with pb3:
        blue_p_mid = st.text_input("Mid", key='blue_p_mid', placeholder="Mid")

with col2:
    st.markdown('<div style="color:#e05454;font-size:11px;font-weight:700;font-family:monospace;letter-spacing:0.08em;margin-bottom:4px;">&#9679; RED SIDE</div>', unsafe_allow_html=True)
    red_team_raw   = st.text_input("Team name", key='red_team_input',
                                    placeholder="e.g. T1, Gen.G, Cloud9...", label_visibility="collapsed")
    red_team_match = fuzzy_match_team(red_team_raw) if red_team_raw else None
    if red_team_match and red_team_match in team_lineups:
        lineup = team_lineups[red_team_match]
        if isinstance(lineup, dict):
            if not st.session_state['red_p_top']: st.session_state['red_p_top'] = lineup.get('top','')
            if not st.session_state['red_p_jg']:  st.session_state['red_p_jg']  = lineup.get('jng','')
            if not st.session_state['red_p_mid']: st.session_state['red_p_mid'] = lineup.get('mid','')
            if not st.session_state['red_p_adc']: st.session_state['red_p_adc'] = lineup.get('adc','')
            if not st.session_state['red_p_sup']: st.session_state['red_p_sup'] = lineup.get('sup','')
    if red_team_raw and red_team_match:
        st.markdown(f'<div style="color:#3a6a20;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#10003; {red_team_match}</div>', unsafe_allow_html=True)
    elif red_team_raw:
        st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#9900; unknown — using averages</div>', unsafe_allow_html=True)
    red_comp_raw = st.text_area("Champions", key='red_comp_input',
                                 placeholder="e.g. Ambessa Pantheon Aurora Jhin Neeko",
                                 height=58, label_visibility="collapsed")
    red_parsed = parse_champion_input(red_comp_raw)
    if red_comp_raw:
        if len(red_parsed) == 5:
            st.markdown(f'<div style="color:#3a6a20;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#10003; {' &middot; '.join([f'{POS_LABELS[i]}: {red_parsed[i]}' for i in range(5)])}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="color:#8a6020;font-size:10px;font-family:monospace;margin:-2px 0 2px;">&#9888; {len(red_parsed)}/5 parsed</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;letter-spacing:0.08em;margin:4px 0 2px;">PLAYERS (optional)</div>', unsafe_allow_html=True)
    pr1, pr2, pr3 = st.columns(3)
    with pr1:
        red_p_top = st.text_input("Top", key='red_p_top', placeholder="Top")
        red_p_adc = st.text_input("ADC", key='red_p_adc', placeholder="ADC")
    with pr2:
        red_p_jg  = st.text_input("Jng", key='red_p_jg',  placeholder="Jng")
        red_p_sup = st.text_input("Sup", key='red_p_sup', placeholder="Sup")
    with pr3:
        red_p_mid = st.text_input("Mid", key='red_p_mid', placeholder="Mid")

gc1, gc2, gc3, gc4 = st.columns([1, 1, 1, 1])
with gc1:
    game_number = st.text_input("Game #", key='game_number', placeholder="1, 2, 3...")
with gc2:
    st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;letter-spacing:0.08em;margin-bottom:2px;">WIN ODDS</div>', unsafe_allow_html=True)
    wo1, wo2 = st.columns(2)
    with wo1:
        win_blue_odds = st.number_input("Blue", min_value=1.01, max_value=10.0, value=1.85, step=0.05, key="wbo")
    with wo2:
        win_red_odds  = st.number_input("Red",  min_value=1.01, max_value=10.0, value=1.95, step=0.05, key="wro")
with gc3:
    st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;letter-spacing:0.08em;margin-bottom:2px;">FT5 ODDS</div>', unsafe_allow_html=True)
    fo1, fo2 = st.columns(2)
    with fo1:
        ft5_blue_odds = st.number_input("Blue", min_value=1.01, max_value=10.0, value=1.85, step=0.05, key="fbo")
    with fo2:
        ft5_red_odds  = st.number_input("Red",  min_value=1.01, max_value=10.0, value=1.95, step=0.05, key="fro")
with gc4:
    st.markdown('<div style="color:#3a4a6a;font-size:10px;font-family:monospace;letter-spacing:0.08em;margin-bottom:2px;">LOG TO</div>', unsafe_allow_html=True)
    send_discord   = st.checkbox("Discord",   value=True)
    send_ft5_sheet = st.checkbox("FT5 Sheet", value=True)
    send_win_sheet = st.checkbox("Win Sheet", value=True)

predict_btn = st.button("&#9672;  PREDICT", type="primary", use_container_width=True)

# =================================================================
# PREDICTION
# =================================================================
if predict_btn:
    blue         = blue_parsed
    red          = red_parsed
    blue_players = [blue_p_top, blue_p_jg, blue_p_mid, blue_p_adc, blue_p_sup]
    red_players  = [red_p_top,  red_p_jg,  red_p_mid,  red_p_adc,  red_p_sup]

    blue_team_norm = blue_team_match
    red_team_norm  = red_team_match
    blue_team_name = blue_team_raw.strip() if blue_team_raw.strip() else "Blue Team"
    red_team_name  = red_team_raw.strip()  if red_team_raw.strip()  else "Red Team"
    game_label     = f"Game {game_number.strip()}" if game_number.strip() else ""

    picked = [c for c in blue + red if c]
    if len(picked) != len(set(picked)) and len(picked) > 0:
        st.error("Each champion must be unique across both teams!")
    elif blue_team_norm and red_team_norm and blue_team_norm == red_team_norm:
        st.error("Blue and red team can't be the same!")
    else:
        missing = []
        if not blue_team_raw.strip(): missing.append("blue team")
        if not red_team_raw.strip():  missing.append("red team")
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

        b_champ_wr = sum(win_champ_rate.get(c,0.5) for c in blue)/len(blue) if blue else 0.5
        r_champ_wr = sum(win_champ_rate.get(c,0.5) for c in red) /len(red)  if red  else 0.5

        win_h2h_r  = get_h2h_rate(win_h2h, blue_team_norm, red_team_norm) \
                     if blue_team_norm and red_team_norm else 0.5
        b_form     = get_form(win_team_recent, blue_team_norm) if blue_team_norm else 0.5
        r_form     = get_form(win_team_recent, red_team_norm)  if red_team_norm  else 0.5
        h2h_total  = get_h2h_total(win_h2h, blue_team_norm, red_team_norm) \
                     if blue_team_norm and red_team_norm else 0
        b_win_h2h, r_win_h2h = get_h2h_record(win_h2h, blue_team_norm, red_team_norm) \
                                if blue_team_norm and red_team_norm else (0,0)

        b_pc_avg = get_blended_avg(blue_players, blue) if blue else 0.5
        r_pc_avg = get_blended_avg(red_players,  red)  if red  else 0.5

        b_recent_wr = get_recent_wr(win_team_recent, blue_team_norm) if blue_team_norm else 0.5
        r_recent_wr = get_recent_wr(win_team_recent, red_team_norm)  if red_team_norm  else 0.5

        # Gold trajectory features
        b_gd20, b_late = get_gold_features(blue_team_norm) if blue_team_norm else (0.0, 0.0)
        r_gd20, r_late = get_gold_features(red_team_norm)  if red_team_norm  else (0.0, 0.0)

        win_extra = pd.DataFrame([[
            b_wr, r_wr, b_wr-r_wr, b_games, r_games,
            b_champ_wr, r_champ_wr, b_champ_wr-r_champ_wr,
            win_h2h_r, b_form, r_form, b_form-r_form,
            b_recent_wr, r_recent_wr, b_recent_wr-r_recent_wr,
            BLUE_SIDE_WINRATE, b_pc_avg, r_pc_avg, b_pc_avg-r_pc_avg,
            b_gd20, r_gd20, b_gd20-r_gd20,
            b_late, r_late, b_late-r_late,
        ]], columns=[
            'blue_team_winrate','red_team_winrate','team_winrate_diff',
            'blue_team_games','red_team_games',
            'blue_avg_winrate','red_avg_winrate','winrate_diff',
            'h2h_winrate','blue_form','red_form','form_diff',
            'blue_recent_wr','red_recent_wr','recent_wr_diff',
            'blue_side_advantage','blue_pc_avg','red_pc_avg','pc_avg_diff',
            'blue_avg_gd20','red_avg_gd20','gd20_diff',
            'blue_late_scaling','red_late_scaling','late_scaling_diff',
        ])
        win_prob_raw  = win_model.predict_proba(pd.concat([b_win_enc,r_win_enc,win_extra],axis=1))[0]
        # Clip to 5%-95% — prevents 100% confidence outputs
        blue_win_conf = min(max(win_prob_raw[1], 0.05), 0.95)
        red_win_conf  = min(max(win_prob_raw[0], 0.05), 0.95)

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

        b_agg        = sum(champ_aggression.get(c,0.5) for c in blue)/len(blue) if blue else 0.5
        r_agg        = sum(champ_aggression.get(c,0.5) for c in red) /len(red)  if red  else 0.5
        b_early      = team_early_rate.get(blue_team_norm,0.5)  if blue_team_norm else 0.5
        r_early      = team_early_rate.get(red_team_norm, 0.5)  if red_team_norm  else 0.5
        b_speed      = team_kill_speed.get(blue_team_norm,10.0) if blue_team_norm else 10.0
        r_speed      = team_kill_speed.get(red_team_norm, 10.0) if red_team_norm  else 10.0
        ft5_h2h_r    = get_h2h_rate(ft5_h2h, blue_team_norm, red_team_norm) \
                       if blue_team_norm and red_team_norm else 0.5
        ft5_h2h_tot  = get_h2h_total(ft5_h2h, blue_team_norm, red_team_norm) \
                       if blue_team_norm and red_team_norm else 0
        b_ft5_h2h, r_ft5_h2h = get_h2h_record(ft5_h2h, blue_team_norm, red_team_norm) \
                                if blue_team_norm and red_team_norm else (0,0)
        b_early_form = get_form(ft5_team_recent, blue_team_norm) if blue_team_norm else 0.5
        r_early_form = get_form(ft5_team_recent, red_team_norm)  if red_team_norm  else 0.5

        ft5_extra = pd.DataFrame([[
            b_agg, r_agg, b_agg-r_agg,
            b_early, r_early, b_early-r_early,
            b_speed, r_speed, r_speed-b_speed,
            ft5_h2h_r, b_early_form, r_early_form, b_early_form-r_early_form,
        ]], columns=[
            'blue_aggression','red_aggression','aggression_diff',
            'blue_early_rate','red_early_rate','early_rate_diff',
            'blue_kill_speed','red_kill_speed','speed_diff',
            'h2h_early_rate','blue_early_form','red_early_form','early_form_diff',
        ])
        ft5_prob_raw  = ft5_model.predict_proba(pd.concat([b_ft5_enc,r_ft5_enc,ft5_extra],axis=1))[0]
        # Clip to 5%-95%
        blue_ft5_conf = min(max(ft5_prob_raw[1], 0.05), 0.95)
        red_ft5_conf  = min(max(ft5_prob_raw[0], 0.05), 0.95)

        if len(blue)==5 and len(red)==5:
            bdw, rdw, bdf, rdf = get_draft_only_prediction(
                blue, red, b_champ_wr, r_champ_wr, b_pc_avg, r_pc_avg)
        else:
            bdw = rdw = bdf = rdf = None

        win_blue_edge, win_blue_units, win_blue_label, win_blue_impl = calc_edge(blue_win_conf, win_blue_odds)
        win_red_edge,  win_red_units,  win_red_label,  win_red_impl  = calc_edge(red_win_conf,  win_red_odds)
        ft5_blue_edge, ft5_blue_units, ft5_blue_label, ft5_blue_impl = calc_edge(blue_ft5_conf, ft5_blue_odds)
        ft5_red_edge,  ft5_red_units,  ft5_red_label,  ft5_red_impl  = calc_edge(red_ft5_conf,  ft5_red_odds)

        # Strong red signal detection
        # Backtest: blue conf < 48% → red picks are 66% accurate, 14.9% ROI WITHOUT boost
        # Unit boost was tested and HURTS ROI (10.2% vs 14.9%) — let calc_edge decide
        ft5_strong_red = blue_ft5_conf < 0.48

        win_winner  = blue_team_name if blue_win_conf > red_win_conf else red_team_name
        ft5_winner  = blue_team_name if blue_ft5_conf > red_ft5_conf else red_team_name
        faster_team = blue_team_name if b_speed < r_speed else red_team_name
        est_time    = (b_speed + r_speed) / 2

        win_conf_level, win_conf_desc, win_reasons, win_warnings = model_confidence(
            b_games, r_games, h2h_total, b_form-r_form, b_wr-r_wr, b_champ_wr-r_champ_wr)
        ft5_conf_level, ft5_conf_desc, ft5_reasons, ft5_warnings = ft5_confidence(
            blue_ft5_conf,
            b_early, r_early,
            b_speed, r_speed,
            b_agg,   r_agg,
            ft5_h2h_tot,
            b_early_form, r_early_form)

        win_caution = 0.60 <= max(blue_win_conf, red_win_conf) < 0.65
        ft5_caution = 0.60 <= max(blue_ft5_conf, red_ft5_conf) < 0.65

        # Shared fields
        series_str  = f"{blue_team_name} vs {red_team_name}"
        league_str  = get_league(blue_team_norm or red_team_norm)
        map_str     = game_number.strip() if game_number.strip() else ""
        today_str   = datetime.now().strftime("%m/%d/%Y")

        # FT5 sheet row
        ft5_pick       = blue_team_name if blue_ft5_conf > red_ft5_conf else red_team_name
        ft5_pick_conf  = max(blue_ft5_conf, red_ft5_conf)
        ft5_pick_odds  = ft5_blue_odds if blue_ft5_conf > red_ft5_conf else ft5_red_odds
        ft5_pick_units = ft5_blue_units if blue_ft5_conf > red_ft5_conf else ft5_red_units
        ft5_pick_label = ft5_blue_label if blue_ft5_conf > red_ft5_conf else ft5_red_label
        ft5_bot_rec    = str(ft5_pick_units) if ft5_pick_units > 0 else "Skip"

        ft5_row = [
            today_str, series_str, map_str, league_str, ft5_pick,
            "", ft5_bot_rec, conf_short(ft5_conf_level),
            round(ft5_pick_conf, 4), ft5_pick_odds,
        ]

        # Winner sheet row
        win_pick       = blue_team_name if blue_win_conf > red_win_conf else red_team_name
        win_pick_conf  = max(blue_win_conf, red_win_conf)
        win_pick_odds  = win_blue_odds if blue_win_conf > red_win_conf else win_red_odds
        win_pick_units = win_blue_units if blue_win_conf > red_win_conf else win_red_units
        win_pick_label = win_blue_label if blue_win_conf > red_win_conf else win_red_label
        win_bot_rec    = str(win_pick_units) if win_pick_units > 0 else "Skip"

        winner_row = [
            today_str, series_str, map_str, league_str, win_pick,
            "", win_bot_rec, conf_short(win_conf_level),
            round(win_pick_conf, 4), win_pick_odds,
        ]

        ft5_sheets_ok    = log_to_sheets(ft5_row,    st.secrets["GOOGLE_SHEETS_ID"])        if send_ft5_sheet else None
        winner_sheets_ok = log_to_sheets(winner_row, st.secrets["GOOGLE_WINNER_SHEETS_ID"]) if send_win_sheet else None

        # Discord
        win_edge_d  = max(win_blue_edge, win_red_edge) * 100
        win_units_d = win_blue_units if blue_win_conf > red_win_conf else win_red_units
        win_label_d = win_blue_label if blue_win_conf > red_win_conf else win_red_label
        win_odds_d  = win_blue_odds  if blue_win_conf > red_win_conf else win_red_odds
        ft5_edge_d  = max(ft5_blue_edge, ft5_red_edge) * 100
        game_str    = f" — {game_label}" if game_label else ""

        draft_win_str = ""
        draft_ft5_str = ""
        if bdw is not None:
            dw_pick = blue_team_name if bdw > rdw else red_team_name
            df_pick = blue_team_name if bdf > rdf else red_team_name
            draft_win_str = f"\n⚖️ Draft Win: 🔵 {bdw*100:.1f}% vs 🔴 {rdw*100:.1f}% — {dw_pick}"
            draft_ft5_str = f"\n⚖️ Draft FT5: 🔵 {bdf*100:.1f}% vs 🔴 {rdf*100:.1f}% — {df_pick}"

        red_signal_str = ""
        if ft5_strong_red:
            red_signal_str = "\n🚨 STRONG RED SIGNAL — blue conf below 48% (66% historical red accuracy, 14.9% ROI)"

        discord_msg = f"""🎮 **{blue_team_name} vs {red_team_name}**{game_str}

🏆 **WINNER: {win_pick}** {win_pick_conf*100:.1f}% | Edge: +{win_edge_d:.1f}% | Odds: {win_odds_d} | {win_units_d}u {win_label_d}
⚔️ **FT5: {ft5_pick}** {ft5_pick_conf*100:.1f}% | Edge: +{ft5_edge_d:.1f}% | Odds: {ft5_pick_odds} | {ft5_bot_rec}u {ft5_pick_label}
📊 Win confidence: {win_conf_level} | FT5 confidence: {ft5_conf_level}{draft_win_str}{draft_ft5_str}{red_signal_str}"""

        discord_sent = send_discord_dm(discord_msg) if send_discord else None


        # ── helpers ──
        def conf_display(level):
            if 'HIGH'   in level: return ('HIGH',   '#0a1f05', '#80d040', '#2a5010')
            if 'MEDIUM' in level: return ('MEDIUM', '#1f1a05', '#d0a040', '#5a4010')
            return                        ('LOW',    '#1a0505', '#f06060', '#5a1010')

        def agg_tag(agg_score):
            if agg_score >= 0.58: return ('high', '#c0f060')
            if agg_score >= 0.48: return ('avg',  '#8090b0')
            return                       ('low',  '#60a0f0')

        def fmt_gl(seconds):
            if not seconds or seconds == 0: return 'N/A'
            return f"{int(seconds//60)}:{int(seconds%60):02d}"

        # ── game style data ──
        b_avg_gl = team_avg_gamelength.get(blue_team_norm, 0)
        r_avg_gl = team_avg_gamelength.get(red_team_norm, 0)
        b_avg_k  = team_avg_kills.get(blue_team_norm, 0)
        r_avg_k  = team_avg_kills.get(red_team_norm, 0)
        total_kills_est = (b_avg_k + r_avg_k) if (b_avg_k > 0 and r_avg_k > 0) else 0

        win_conf_txt, win_bg, win_fg, win_br = conf_display(win_conf_level)
        ft5_conf_txt, ft5_bg, ft5_fg, ft5_br = conf_display(ft5_conf_level)

        win_pick_edge = win_blue_edge if blue_win_conf > red_win_conf else win_red_edge
        ft5_pick_edge = ft5_blue_edge if blue_ft5_conf > red_ft5_conf else ft5_red_edge

        win_pick_show = win_pick_units > 0
        ft5_pick_show = ft5_pick_units > 0

        if win_pick_show:
            win_rec_str = f"{win_winner} WIN &middot; {win_pick_units}u {win_pick_label.replace('✅ ','').replace('🔥 ','')} &middot; @{win_pick_odds}"
        else:
            win_rec_str = f"{win_winner} WIN &middot; SKIP (edge {win_pick_edge*100:.1f}%)"

        if ft5_pick_show:
            ft5_rec_str = f"{ft5_winner} FT5 &middot; {ft5_pick_units}u {ft5_pick_label.replace('✅ ','').replace('🔥 ','')} &middot; @{ft5_pick_odds} &middot; est ~{est_time:.1f} min"
        else:
            ft5_rec_str = f"{ft5_winner} FT5 &middot; SKIP (edge {ft5_pick_edge*100:.1f}%)"

        win_pick_color  = '#c0f060' if win_pick_show else '#4a5a7a'
        win_pick_bg     = '#0d1f05' if win_pick_show else '#0f1218'
        win_pick_border = '#2a5a10' if win_pick_show else '#1e2535'
        ft5_pick_color  = '#60a0f0' if ft5_pick_show else '#4a5a7a'
        ft5_pick_bg     = '#05101f' if ft5_pick_show else '#0f1218'
        ft5_pick_border = '#103a6a' if ft5_pick_show else '#1e2535'

        def draft_rows_html(picks, players):
            if not picks: return '<div style="color:#3a4060;font-size:10px;">No picks entered</div>'
            pos_labels = ['TOP','JNG','MID','ADC','SUP']
            rows = ''
            for i, champ in enumerate(picks):
                pos = pos_labels[i] if i < len(pos_labels) else ''
                agg = champ_aggression.get(champ, 0.5)
                tag, tagcol = agg_tag(agg)
                player = players[i].strip() if i < len(players) and players[i].strip() else ''
                pl_html = f' <span style="color:#5a6a8a;font-size:9px;">({player})</span>' if player else ''
                rows += (f'<div style="display:grid;grid-template-columns:28px 1fr auto;align-items:center;gap:5px;padding:2px 0;">'
                         f'<span style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.06em;">{pos}</span>'
                         f'<span style="font-size:11px;color:#8090b0;">{champ}{pl_html}</span>'
                         f'<span style="font-size:9px;color:{tagcol};">{tag}</span>'
                         f'</div>')
            return rows

        blue_draft_html = draft_rows_html(blue, blue_players)
        red_draft_html  = draft_rows_html(red,  red_players)

        league_detected = league_str if league_str else get_league(blue_team_norm or red_team_norm)
        ft5_league_tips = {
            'LCK':   'LCK FT5: Best league — +6.9% edge. Red signal 69% accurate.',
            'LPL':   'LPL FT5: Model not trained on LPL — rough guide only.',
            'LEC':   'LEC FT5: Weak edge (+3.1%). Only bet 60%+ or strong red signal.',
            'LCS':   'LCS FT5: 0% model edge. Blue baseline (55%) is main edge.',
            'CBLOL': 'CBLOL FT5: Solid edge (+3.2%). Red signal 63% accurate.',
            'FST':   'FST FT5: Small sample. Red signal weak (25%). Favour blue.',
        }
        league_tip = next((tip for lg, tip in ft5_league_tips.items()
                           if lg.lower() in (league_detected or '').lower()), '')

        red_signal_html = (
            '<div style="background:#1a0505;border-left:2px solid #6a1010;padding:5px 8px;'
            'border-radius:0 4px 4px 0;font-size:10px;color:#f06060;margin-bottom:8px;">'
            f'STRONG RED SIGNAL — blue conf {blue_ft5_conf*100:.1f}% (below 48%). '
            'Backtest: 66% red accuracy / 14.9% ROI.</div>'
        ) if ft5_strong_red else ''

        win_caution_html = (
            '<div style="background:#1f1a05;border-left:2px solid #5a4010;padding:5px 8px;'
            'border-radius:0 4px 4px 0;font-size:10px;color:#c0a040;margin-top:6px;">'
            '60-65% range — backtest ~57% actual accuracy, be cautious</div>'
        ) if win_caution else ''

        ft5_caution_html = (
            '<div style="background:#1f1a05;border-left:2px solid #5a4010;padding:5px 8px;'
            'border-radius:0 4px 4px 0;font-size:10px;color:#c0a040;margin-top:6px;">'
            '60-65% range — treat with extra caution</div>'
        ) if ft5_caution else ''

        draft_win_cap = ''
        if bdw is not None:
            dw_name = blue_team_name if bdw > rdw else red_team_name
            draft_win_cap = f'<div style="color:#3a4060;font-size:10px;margin-top:4px;">Draft-only win: {blue_team_name} {bdw*100:.1f}% vs {red_team_name} {rdw*100:.1f}% &mdash; {dw_name} has better draft</div>'
        draft_ft5_cap = ''
        if bdf is not None:
            df5_name = blue_team_name if bdf > rdf else red_team_name
            draft_ft5_cap = f'<div style="color:#3a4060;font-size:10px;margin-top:4px;">Draft-only FT5: {blue_team_name} {bdf*100:.1f}% vs {red_team_name} {rdf*100:.1f}% &mdash; {df5_name} more aggressive</div>'

        game_lbl_html = f' &mdash; {game_label}' if game_label else ''
        b_avg_k_str = f"{b_avg_k:.1f}" if b_avg_k else 'N/A'
        r_avg_k_str = f"{r_avg_k:.1f}" if r_avg_k else 'N/A'
        draft_only_str = f"{(blue_team_name if bdw and bdw > rdw else red_team_name)[:5]} {max(bdw,rdw)*100:.1f}%" if bdw is not None else 'N/A'

        st.markdown(f"""
<div style="background:#0a0c10;border:1px solid #1e2330;border-radius:10px;padding:16px;
     font-family:'SF Mono','Fira Code','Consolas',monospace;font-size:12px;margin-top:16px;">

  <div style="display:flex;align-items:center;justify-content:space-between;
              border-bottom:1px solid #1e2535;padding-bottom:10px;margin-bottom:14px;">
    <span style="color:#c0f060;font-size:13px;font-weight:700;letter-spacing:0.06em;">&#9672; LOL MATCH PREDICTOR v8</span>
    <span style="background:#1a2a10;color:#6db33f;font-size:10px;padding:2px 8px;border-radius:3px;border:1px solid #2a4a1a;">
      {(league_detected or 'N/A').upper()}{game_lbl_html}
    </span>
  </div>

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
    <div>
      <div style="font-size:15px;font-weight:700;color:#e0e8ff;">{blue_team_name}</div>
      <div style="color:#3a4a6a;margin-top:2px;font-size:11px;">BLUE &middot; {b_wr*100:.1f}% WR &middot; form {b_form*100:.0f}%</div>
    </div>
    <div style="color:#2a3050;font-size:11px;font-weight:700;padding:0 10px;">VS</div>
    <div style="text-align:right;">
      <div style="font-size:15px;font-weight:700;color:#e0e8ff;">{red_team_name}</div>
      <div style="color:#3a4a6a;margin-top:2px;font-size:11px;">RED &middot; {r_wr*100:.1f}% WR &middot; form {r_form*100:.0f}%</div>
    </div>
  </div>

  <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">DRAFT</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px;">
    <div style="background:#0f1218;border:1px solid #1e2535;border-left:2px solid #1e4a80;border-radius:5px;padding:8px 10px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#2a5a90;margin-bottom:6px;">BLUE &middot; {blue_team_name}</div>
      {blue_draft_html}
    </div>
    <div style="background:#0f1218;border:1px solid #1e2535;border-left:2px solid #801e1e;border-radius:5px;padding:8px 10px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#903030;margin-bottom:6px;">RED &middot; {red_team_name}</div>
      {red_draft_html}
    </div>
  </div>

  <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">MATCH WINNER</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:6px;">
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">BLUE &middot; {blue_team_name}</div>
      <div style="font-size:22px;font-weight:700;color:#c0f060;line-height:1;">{blue_win_conf*100:.1f}%</div>
      <div style="margin-top:5px;font-size:10px;color:#3a4a6a;">odds {win_blue_odds} &middot; impl {win_blue_impl*100:.1f}% &middot; edge {'+' if win_blue_edge>=0 else ''}{win_blue_edge*100:.1f}%</div>
    </div>
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">RED &middot; {red_team_name}</div>
      <div style="font-size:22px;font-weight:700;color:#f06060;line-height:1;">{red_win_conf*100:.1f}%</div>
      <div style="margin-top:5px;font-size:10px;color:#3a4a6a;">odds {win_red_odds} &middot; impl {win_red_impl*100:.1f}% &middot; edge {'+' if win_red_edge>=0 else ''}{win_red_edge*100:.1f}%</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;">MODEL CONFIDENCE</span>
    <span style="background:{win_bg};color:{win_fg};border:1px solid {win_br};font-size:9px;padding:2px 8px;border-radius:3px;font-weight:700;">{win_conf_txt}</span>
    <span style="color:#3a4a6a;font-size:10px;">&mdash; {win_conf_desc}</span>
  </div>
  {win_caution_html}
  <div style="background:{win_pick_bg};border:1px solid {win_pick_border};border-radius:5px;
              padding:9px 14px;display:flex;align-items:center;justify-content:space-between;margin-top:8px;">
    <div>
      <div style="font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{'#4a8020' if win_pick_show else '#3a4a6a'};">RECOMMENDED BET &mdash; MATCH WINNER</div>
      <div style="font-size:14px;font-weight:700;color:{win_pick_color};margin-top:2px;">{win_rec_str}</div>
    </div>
    <span style="background:{win_bg};color:{win_fg};border:1px solid {win_br};font-size:11px;padding:4px 12px;border-radius:3px;font-weight:700;">{'PICK' if win_pick_show else 'SKIP'}</span>
  </div>
  {draft_win_cap}

  <div style="border-top:1px solid #1e2535;margin:14px 0;"></div>

  <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:6px;">FIRST TO FIVE KILLS</div>
  {red_signal_html}
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:6px;">
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">BLUE &middot; {blue_team_name}</div>
      <div style="font-size:22px;font-weight:700;color:#60a0f0;line-height:1;">{blue_ft5_conf*100:.1f}%</div>
      <div style="margin-top:5px;font-size:10px;color:#3a4a6a;">odds {ft5_blue_odds} &middot; impl {ft5_blue_impl*100:.1f}% &middot; edge {'+' if ft5_blue_edge>=0 else ''}{ft5_blue_edge*100:.1f}%</div>
    </div>
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:4px;">RED &middot; {red_team_name}</div>
      <div style="font-size:22px;font-weight:700;color:#f06060;line-height:1;">{red_ft5_conf*100:.1f}%</div>
      <div style="margin-top:5px;font-size:10px;color:#3a4a6a;">odds {ft5_red_odds} &middot; impl {ft5_red_impl*100:.1f}% &middot; edge {'+' if ft5_red_edge>=0 else ''}{ft5_red_edge*100:.1f}%</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
    <span style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;">FT5 CONFIDENCE</span>
    <span style="background:{ft5_bg};color:{ft5_fg};border:1px solid {ft5_br};font-size:9px;padding:2px 8px;border-radius:3px;font-weight:700;">{ft5_conf_txt}</span>
    <span style="color:#3a4a6a;font-size:10px;">&mdash; {ft5_conf_desc}</span>
  </div>
  {ft5_caution_html}
  <div style="background:{ft5_pick_bg};border:1px solid {ft5_pick_border};border-radius:5px;
              padding:9px 14px;display:flex;align-items:center;justify-content:space-between;margin-top:8px;">
    <div>
      <div style="font-size:9px;letter-spacing:0.12em;text-transform:uppercase;color:{'#2a6090' if ft5_pick_show else '#3a4a6a'};">RECOMMENDED BET &mdash; FIRST TO FIVE</div>
      <div style="font-size:14px;font-weight:700;color:{ft5_pick_color};margin-top:2px;">{ft5_rec_str}</div>
    </div>
    <span style="background:{ft5_bg};color:{ft5_fg};border:1px solid {ft5_br};font-size:11px;padding:4px 12px;border-radius:3px;font-weight:700;">{'PICK' if ft5_pick_show else 'SKIP'}</span>
  </div>
  {draft_ft5_cap}

  <div style="border-top:1px solid #1e2535;margin:14px 0;"></div>

  <div style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:#3a4a6a;margin-bottom:8px;">GAME STYLE</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px;">
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#2a5a90;margin-bottom:6px;">BLUE &middot; {blue_team_name}</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <div><div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">AVG GAME</div><div style="font-size:15px;font-weight:700;color:#d0d8f0;">{fmt_gl(b_avg_gl)}</div></div>
        <div style="text-align:right;"><div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">AVG KILLS</div><div style="font-size:15px;font-weight:700;color:#d0d8f0;">{b_avg_k_str}</div></div>
      </div>
    </div>
    <div style="background:#0f1218;border:1px solid #1e2535;border-radius:5px;padding:10px 12px;">
      <div style="font-size:9px;letter-spacing:0.1em;text-transform:uppercase;color:#903030;margin-bottom:6px;">RED &middot; {red_team_name}</div>
      <div style="display:flex;justify-content:space-between;align-items:baseline;">
        <div><div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">AVG GAME</div><div style="font-size:15px;font-weight:700;color:#d0d8f0;">{fmt_gl(r_avg_gl)}</div></div>
        <div style="text-align:right;"><div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">AVG KILLS</div><div style="font-size:15px;font-weight:700;color:#d0d8f0;">{r_avg_k_str}</div></div>
      </div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;">
    <div style="background:#0f1218;border-radius:4px;padding:7px 10px;">
      <div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">EXP TOTAL KILLS</div>
      <div style="font-size:12px;font-weight:600;color:#f0c060;">{'~' + str(round(total_kills_est)) if total_kills_est else 'N/A'}</div>
    </div>
    <div style="background:#0f1218;border-radius:4px;padding:7px 10px;">
      <div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">H2H WIN</div>
      <div style="font-size:12px;font-weight:600;color:#d0d8f0;">{blue_team_name[:4]} {b_win_h2h}&ndash;{r_win_h2h} {red_team_name[:4]}</div>
    </div>
    <div style="background:#0f1218;border-radius:4px;padding:7px 10px;">
      <div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">H2H FT5</div>
      <div style="font-size:12px;font-weight:600;color:#d0d8f0;">{blue_team_name[:4]} {b_ft5_h2h}&ndash;{r_ft5_h2h} {red_team_name[:4]}</div>
    </div>
    <div style="background:#0f1218;border-radius:4px;padding:7px 10px;">
      <div style="font-size:9px;color:#3a4060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:3px;">DRAFT-ONLY</div>
      <div style="font-size:12px;font-weight:600;color:#60a0f0;">{draft_only_str}</div>
    </div>
  </div>
  {"<div style='background:#0a150a;border-left:2px solid #3a6020;padding:5px 8px;border-radius:0 4px 4px 0;font-size:10px;color:#6a8a50;margin-top:10px;'>" + league_tip + "</div>" if league_tip else ""}

</div>
""", unsafe_allow_html=True)

        with st.spinner("Fetching tracker history..."):
            win_history = fetch_tracker_history(win_pick_conf, win_conf_level, st.secrets["GOOGLE_WINNER_SHEETS_ID"])
            ft5_history = fetch_tracker_history(ft5_pick_conf, ft5_conf_level, st.secrets["GOOGLE_SHEETS_ID"])

        if win_history or ft5_history:
            with st.expander("📈 Tracker History", expanded=True):
                if win_history:
                    st.markdown("**Winner picks**")
                    st.markdown(format_history(win_history, "Winner"))
                if ft5_history:
                    st.markdown("**FT5 picks**")
                    st.markdown(format_history(ft5_history, "FT5"))

        with st.expander("📊 Signal Breakdown", expanded=False):
            st.markdown("**Match Winner**")
            show_signal("Team win rate",    b_wr,       r_wr,       0.05, 0.15, blue_team_name, red_team_name)
            show_signal("Recent form",      b_form,     r_form,     0.10, 0.25, blue_team_name, red_team_name, ".0f")
            show_signal("Champion quality", b_champ_wr, r_champ_wr, 0.02, 0.06, blue_team_name, red_team_name)
            show_signal("Player-champ wr",  b_pc_avg,   r_pc_avg,   0.03, 0.08, blue_team_name, red_team_name)
            if b_win_h2h + r_win_h2h > 0:
                h2h_diff  = win_h2h_r - 0.5
                h_str     = "🟢 Strong" if abs(h2h_diff) >= 0.25 else ("🟡 Moderate" if abs(h2h_diff) >= 0.10 else "⚪ Weak")
                direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 else f"favours 🔴 {red_team_name}"
                st.write(f"**H2H:** 🔵 {b_win_h2h}–{r_win_h2h} 🔴 — {h_str} {direction}")
            else:
                st.write("**H2H:** No history — ⚪ Neutral")
            st.write(f"**Blue side:** 53.1% historical — ⚪ Slight edge 🔵 {blue_team_name}")
            st.markdown("**FT5**")
            show_signal("Early game rate", b_early,      r_early,      0.05, 0.15, blue_team_name, red_team_name)
            show_signal("Early form",      b_early_form, r_early_form, 0.10, 0.25, blue_team_name, red_team_name, ".0f")
            show_signal("Aggression",      b_agg,        r_agg,        0.03, 0.08, blue_team_name, red_team_name)
            faster   = blue_team_name if b_speed < r_speed else red_team_name
            spd_diff = abs(b_speed - r_speed)
            spd_str  = f"🟢 Strong — {faster} significantly faster" if spd_diff >= 2.0 else (f"🟡 Moderate — {faster} slightly faster" if spd_diff >= 0.5 else "⚪ Weak — similar speed")
            st.write(f"**Kill speed:** 🔵 {b_speed:.1f}m vs 🔴 {r_speed:.1f}m — {spd_str}")

        if blue or red:
            with st.expander("🏅 Champion Ratings", expanded=False):
                for team_name, picks, players_list in [(blue_team_name, blue, blue_players), (red_team_name, red, red_players)]:
                    if picks:
                        st.markdown(f"**{'🔵' if picks is blue else '🔴'} {team_name}**")
                        for i, champ in enumerate(picks):
                            player  = players_list[i] if i < len(players_list) else ''
                            pos     = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
                            rc_val  = role_champ_rate.get((pos, champ.strip()), 0.5)
                            pc_val  = pc_rate.get((player.strip(), champ.strip()), 0.5) if player.strip() else rc_val
                            pcg     = pc_games_d.get((player.strip(), champ.strip()), 0) if player.strip() else 0
                            cwr     = win_champ_rate.get(champ, 0.5)
                            rating  = rate_champ(cwr, PC_WEIGHT * pc_val + RC_WEIGHT * rc_val)
                            lbl     = POS_LABELS[i] if i < len(POS_LABELS) else ''
                            name    = player if player.strip() else "Unknown"
                            gstr    = f"({pcg}g)" if pcg > 0 else ""
                            st.write(f"**{lbl}** {name} — {champ}: role {rc_val*100:.0f}% | player {pc_val*100:.0f}% {gstr} → {rating}")

        if len(blue) == 5 and len(red) == 5:
            with st.expander("🤖 AI Analysis", expanded=False):
                with st.spinner("Generating analysis..."):
                    reasoning = get_claude_reasoning(
                        blue_team_name, red_team_name,
                        blue, red, blue_players, red_players,
                        blue_win_conf, red_win_conf,
                        blue_ft5_conf, red_ft5_conf,
                        b_wr, r_wr, b_form, r_form,
                        b_champ_wr, r_champ_wr, b_pc_avg, r_pc_avg,
                        b_win_h2h, r_win_h2h,
                        b_agg, r_agg, b_early, r_early, b_speed, r_speed,
                        win_blue_odds, win_red_odds,
                        ft5_blue_odds, ft5_red_odds)
                st.write(reasoning)

        status_parts = []
        if send_discord:
            status_parts.append("📨 Discord sent" if discord_sent else "⚠️ Discord failed")
        if send_ft5_sheet:
            status_parts.append("📊 FT5 logged" if ft5_sheets_ok is True else "⚠️ FT5 sheet failed")
        if send_win_sheet:
            status_parts.append("🏆 Winner logged" if winner_sheets_ok is True else "⚠️ Winner sheet failed")
        if status_parts:
            st.markdown(
                '<div style="color:#3a4a6a;font-size:0.75rem;font-family:monospace;text-align:center;margin-top:8px;">'
                + " &nbsp;|&nbsp; ".join(status_parts) + '</div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="color:#1e2a1e;font-size:0.72rem;font-family:monospace;text-align:center;margin-top:12px;">'
            'V8 | Win 67.09% / AUC 0.7227 | FT5 57.16% | Best ROI at 2.30+ odds</div>',
            unsafe_allow_html=True)
