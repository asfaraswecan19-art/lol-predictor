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

st.title("🎮 LoL Pro Match Predictor")
st.caption("V8.1 | Win + FT5 + Total Kills O/U | ~65.4% win (leak-free) / 57.9% FT5 / +5.7% kills O/U edge")

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
team_kill_speed  = p['team_kill_speed']
KILL_SPEED_DEFAULT = p.get('kill_speed_default', 22.0)  # league mean, not 10.0
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

# V8.1 NEW: patch features
patch_release    = p.get('patch_release', {})  # str patch -> ISO date string
patch_champ      = p.get('patch_champ', {})    # (patch, champ) -> [wins, games]
champ_recent_pkl = p.get('champ_recent', {})   # champ -> [(iso_date, win), ...]
PATCH_WINDOW_DAYS = p.get('patch_window_days', 30)
# Convert ISO strings back to datetimes for runtime use
patch_release_dt = {k: datetime.fromisoformat(v) for k, v in patch_release.items()} if patch_release else {}
champ_recent     = {c: [(datetime.fromisoformat(d), w) for d, w in lst]
                    for c, lst in champ_recent_pkl.items()} if champ_recent_pkl else {}

# V8.1 NEW: isotonic calibrators (FT5 only by default — win-model raw was better in backtest)
win_calibrator   = p.get('win_calibrator', None)
ft5_calibrator   = p.get('ft5_calibrator', None)
APPLY_WIN_CALIB  = p.get('apply_win_calib', False)
APPLY_FT5_CALIB  = p.get('apply_ft5_calib', True)

# V8.2 NEW: kills predictor models and lookups
kls_point       = p.get('kls_point', None)
kls_q10         = p.get('kls_q10', None)
kls_q50         = p.get('kls_q50', None)
kls_q90         = p.get('kls_q90', None)
kls_ou_models   = p.get('kls_ou_models', {})  # {line: classifier}
kls_feat_cols   = p.get('kls_feat_cols', [])
champ_dur_score   = p.get('champ_dur_score', {})
champ_kills_score = p.get('champ_kills_score', {})
kls_global_dur   = p.get('kls_global_dur', 32.0)
kls_global_kills = p.get('kls_global_kills', 27.0)
kls_team_dur_hist   = p.get('kls_team_dur_hist', {})
kls_team_kills_hist = p.get('kls_team_kills_hist', {})
kls_team_ckpm_hist  = p.get('kls_team_ckpm_hist', {})
kls_h2h_dur   = p.get('kls_h2h_dur', {})
kls_h2h_kills = p.get('kls_h2h_kills', {})
kls_patch_dur   = p.get('kls_patch_dur', {})
kls_patch_kills = p.get('kls_patch_kills', {})
KLS_RECENT_WINDOW = p.get('kls_recent_window', 20)
KLS_LINES         = p.get('kls_lines', [22.5, 24.5, 26.5, 28.5, 30.5])
dur_point = p.get('dur_point', None)
dur_q10   = p.get('dur_q10', None)
dur_q50   = p.get('dur_q50', None)
dur_q90   = p.get('dur_q90', None)

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

st.success("Models ready!")

defaults = {
    'blue_team_input': '', 'red_team_input': '',
    'blue_comp_input': '', 'red_comp_input': '',
    'blue_p_top': '', 'blue_p_jg': '', 'blue_p_mid': '',
    'blue_p_adc': '', 'blue_p_sup': '',
    'red_p_top':  '', 'red_p_jg':  '', 'red_p_mid':  '',
    'red_p_adc':  '', 'red_p_sup':  '',
    'game_number': '',
    'current_patch': '',
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

# V8.1 NEW: patch feature helpers
def get_champ_patch_winrate(champ, patch, ref_date=None):
    """Champion winrate on current patch if >=5 games, else 30-day rolling, else 0.5.
    Gracefully handles brand-new patches with no data."""
    from datetime import datetime, timedelta
    if ref_date is None: ref_date = datetime.now()
    elif isinstance(ref_date, str): ref_date = datetime.strptime(ref_date[:10], '%Y-%m-%d')
    w_g = patch_champ.get((patch, champ))
    if w_g and w_g[1] >= 5:
        return w_g[0] / w_g[1]
    cutoff = ref_date - timedelta(days=PATCH_WINDOW_DAYS)
    recent = [w for d, w in champ_recent.get(champ, []) if d >= cutoff]
    if len(recent) >= 5:
        return sum(recent) / len(recent)
    return 0.5

def get_patch_age_days(patch, ref_date=None):
    """Days since this patch first appeared. 0 if brand new (unseen)."""
    from datetime import datetime
    if ref_date is None: ref_date = datetime.now()
    elif isinstance(ref_date, str): ref_date = datetime.strptime(ref_date[:10], '%Y-%m-%d')
    release = patch_release_dt.get(patch)
    if release is None: return 0
    return max(0, (ref_date - release).days)

def get_team_patch_features(picks, patch, ref_date=None):
    """Returns (avg_patch_wr, patch_age_days, is_new_patch) for a 5-champ team."""
    if not picks or not patch:
        return 0.5, 0, 0
    wrs = [get_champ_patch_winrate(c, patch, ref_date) for c in picks]
    age = get_patch_age_days(patch, ref_date)
    return (sum(wrs)/len(wrs), age, 1 if age < 7 else 0)

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

# V8.2 NEW: kills feature builder (mirrors kills model training)
def _rmean(h, w, d): return sum(h[-w:])/len(h[-w:]) if h else d
def _rstd(h, w, d):  return float(np.std(h[-w:])) if len(h) >= 3 else d

def build_kills_features(blue, red, blue_picks, red_picks, patch, league):
    """Build a feature row matching kls_feat_cols. Returns DataFrame with 1 row."""
    gdm = kls_global_dur; gkm = kls_global_kills
    b_dur_h = kls_team_dur_hist.get(blue, [])
    r_dur_h = kls_team_dur_hist.get(red, [])
    b_kls_h = kls_team_kills_hist.get(blue, [])
    r_kls_h = kls_team_kills_hist.get(red, [])
    b_ck_h  = kls_team_ckpm_hist.get(blue, [])
    r_ck_h  = kls_team_ckpm_hist.get(red, [])

    b_dur  = _rmean(b_dur_h, KLS_RECENT_WINDOW, gdm)
    r_dur  = _rmean(r_dur_h, KLS_RECENT_WINDOW, gdm)
    b_kls  = _rmean(b_kls_h, KLS_RECENT_WINDOW, gkm)
    r_kls  = _rmean(r_kls_h, KLS_RECENT_WINDOW, gkm)
    b_ckpm = _rmean(b_ck_h, KLS_RECENT_WINDOW, gkm/gdm)
    r_ckpm = _rmean(r_ck_h, KLS_RECENT_WINDOW, gkm/gdm)
    b_dur_s = _rstd(b_dur_h, KLS_RECENT_WINDOW, 5.0)
    r_dur_s = _rstd(r_dur_h, KLS_RECENT_WINDOW, 5.0)
    b_kls_s = _rstd(b_kls_h, KLS_RECENT_WINDOW, 8.0)
    r_kls_s = _rstd(r_kls_h, KLS_RECENT_WINDOW, 8.0)

    mk = tuple(sorted([blue or "", red or ""]))
    hd = kls_h2h_dur.get(mk, [])
    hk = kls_h2h_kills.get(mk, [])
    h_dur = _rmean(hd, 10, (b_dur+r_dur)/2)
    h_kls = _rmean(hk, 10, (b_kls+r_kls)/2)

    b_ad = np.mean([champ_dur_score.get(c, gdm) for c in blue_picks]) if blue_picks else gdm
    b_ak = np.mean([champ_kills_score.get(c, gkm) for c in blue_picks]) if blue_picks else gkm
    r_ad = np.mean([champ_dur_score.get(c, gdm) for c in red_picks]) if red_picks else gdm
    r_ak = np.mean([champ_kills_score.get(c, gkm) for c in red_picks]) if red_picks else gkm

    p_dur = kls_patch_dur.get(patch, gdm)
    p_kls = kls_patch_kills.get(patch, gkm)
    from datetime import datetime
    rd = patch_release_dt.get(patch) if patch_release_dt else None
    p_age = 0 if rd is None else max(0, (datetime.now() - rd).days)

    feat = {
        'b_dur_mean':b_dur, 'r_dur_mean':r_dur, 'avg_dur_mean':(b_dur+r_dur)/2,
        'b_kills_mean':b_kls, 'r_kills_mean':r_kls, 'avg_kills_mean':(b_kls+r_kls)/2,
        'b_ckpm_mean':b_ckpm, 'r_ckpm_mean':r_ckpm, 'avg_ckpm_mean':(b_ckpm+r_ckpm)/2,
        'b_dur_std':b_dur_s, 'r_dur_std':r_dur_s,
        'b_kills_std':b_kls_s, 'r_kills_std':r_kls_s,
        'min_games_seen': min(len(b_dur_h), len(r_dur_h)),
        'h2h_dur_mean':h_dur, 'h2h_kills_mean':h_kls, 'h2h_n':len(hd),
        'b_arch_dur':b_ad, 'r_arch_dur':r_ad, 'avg_arch_dur':(b_ad+r_ad)/2,
        'b_arch_kills':b_ak, 'r_arch_kills':r_ak, 'avg_arch_kills':(b_ak+r_ak)/2,
        'patch_dur':p_dur, 'patch_kills':p_kls,
        'patch_age_days':p_age, 'is_new_patch': 1 if p_age < 7 else 0,
        'league_idx': hash(league) % 1000,
    }
    return pd.DataFrame([feat])[kls_feat_cols]

def predict_kills_ou(X_row, target_line):
    """Predict probability that total kills exceeds target_line.
    Picks the closest trained O/U classifier (we trained at 22.5/24.5/26.5/28.5/30.5)
    and adjusts slightly toward the quantile estimate for off-line targets.
    """
    if not kls_ou_models:
        return None
    available_lines = sorted(kls_ou_models.keys())
    closest = min(available_lines, key=lambda x: abs(x - target_line))
    model = kls_ou_models[closest]
    prob_over_closest = float(model.predict_proba(X_row)[0, 1])

    # If user's line is far from any trained line, blend with point prediction
    # to handle the offset (e.g. line=27.0 between 26.5 and 28.5)
    if abs(target_line - closest) <= 0.5:
        return prob_over_closest, closest
    # For larger gaps, use the q50 + assumption of approx normal residuals
    if kls_q50 is not None:
        median = float(kls_q50.predict(X_row)[0])
        # Crude blend: shift the closest-line probability based on how target compares to median
        offset = (target_line - closest) * 0.05  # ~5pp per kill of line shift, empirically reasonable
        if target_line > closest:
            return max(0.05, prob_over_closest - offset), closest
        else:
            return min(0.95, prob_over_closest + offset), closest
    return prob_over_closest, closest

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
        0.5, 0.5, 0.0, KILL_SPEED_DEFAULT, KILL_SPEED_DEFAULT, 0.0,
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
    st.markdown("### 🔵 Blue Side")
    blue_team_raw   = st.text_input("Team name", key='blue_team_input',
                                     placeholder="e.g. T1, Gen.G, Cloud9...")
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
        st.caption(f"✅ Matched: {blue_team_match}")
    elif blue_team_raw and not blue_team_match:
        st.caption("⚪ Unknown team — using average stats")

    blue_comp_raw = st.text_area("Champion comp (Top Jng Mid ADC Sup)",
                                  key='blue_comp_input',
                                  placeholder="e.g. Gnar Nocturne Ahri Caitlyn Bard",
                                  height=80)
    blue_parsed = parse_champion_input(blue_comp_raw)
    if blue_comp_raw:
        if len(blue_parsed) == 5:
            st.caption(f"✅ {' | '.join([f'{POS_LABELS[i]}: {blue_parsed[i]}' for i in range(5)])}")
        else:
            st.caption(f"⚠️ Parsed {len(blue_parsed)}/5 — {', '.join(blue_parsed) if blue_parsed else 'none recognized'}")

    st.markdown("**Players (optional)**")
    blue_p_top = st.text_input("Top",     key='blue_p_top')
    blue_p_jg  = st.text_input("Jungle",  key='blue_p_jg')
    blue_p_mid = st.text_input("Mid",     key='blue_p_mid')
    blue_p_adc = st.text_input("ADC",     key='blue_p_adc')
    blue_p_sup = st.text_input("Support", key='blue_p_sup')

with col2:
    st.markdown("### 🔴 Red Side")
    red_team_raw   = st.text_input("Team name", key='red_team_input',
                                    placeholder="e.g. T1, Gen.G, Cloud9...")
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
        st.caption(f"✅ Matched: {red_team_match}")
    elif red_team_raw and not red_team_match:
        st.caption("⚪ Unknown team — using average stats")

    red_comp_raw = st.text_area("Champion comp (Top Jng Mid ADC Sup)",
                                 key='red_comp_input',
                                 placeholder="e.g. Ambessa Pantheon Aurora Jhin Neeko",
                                 height=80)
    red_parsed = parse_champion_input(red_comp_raw)
    if red_comp_raw:
        if len(red_parsed) == 5:
            st.caption(f"✅ {' | '.join([f'{POS_LABELS[i]}: {red_parsed[i]}' for i in range(5)])}")
        else:
            st.caption(f"⚠️ Parsed {len(red_parsed)}/5 — {', '.join(red_parsed) if red_parsed else 'none recognized'}")

    st.markdown("**Players (optional)**")
    red_p_top = st.text_input("Top",     key='red_p_top')
    red_p_jg  = st.text_input("Jungle",  key='red_p_jg')
    red_p_mid = st.text_input("Mid",     key='red_p_mid')
    red_p_adc = st.text_input("ADC",     key='red_p_adc')
    red_p_sup = st.text_input("Support", key='red_p_sup')

gc1, gc2, gc3 = st.columns([1, 2, 2])
with gc1:
    game_number = st.text_input("Game #", key='game_number', placeholder="1, 2, 3...")
    # V8.1 NEW: patch input — auto-fills latest known patch, can be overridden for new ones
    latest_patch = max(patch_release_dt.keys(), key=lambda p: patch_release_dt[p]) if patch_release_dt else ""
    current_patch = st.text_input("Patch", key='current_patch',
                                   value=latest_patch, placeholder="e.g. 26.05",
                                   help="Defaults to latest known patch. Override for new patches; "
                                        "model falls back to 30-day rolling winrate if patch is unknown.")
with gc2:
    st.markdown("**Match Winner**")
    win_blue_odds = st.number_input("Blue odds", min_value=1.01, max_value=10.0,
                                     value=1.85, step=0.05, key="wbo")
    win_red_odds  = st.number_input("Red odds",  min_value=1.01, max_value=10.0,
                                     value=1.95, step=0.05, key="wro")
with gc3:
    st.markdown("**First to Five**")
    ft5_blue_odds = st.number_input("Blue odds", min_value=1.01, max_value=10.0,
                                     value=1.85, step=0.05, key="fbo")
    ft5_red_odds  = st.number_input("Red odds",  min_value=1.01, max_value=10.0,
                                     value=1.95, step=0.05, key="fro")

# V8.2 NEW: Kills O/U input row
st.markdown("**💀 Total Kills Over/Under** *(optional — leave blank to skip)*")
kc1, kc2, kc3 = st.columns([1, 1, 1])
with kc1:
    kills_line = st.number_input("Kill line", min_value=15.0, max_value=45.0,
                                  value=26.5, step=0.5, key="kls_line",
                                  help="The over/under line offered by the book (e.g. 26.5)")
with kc2:
    kills_over_odds = st.number_input("Over odds", min_value=1.01, max_value=10.0,
                                       value=1.90, step=0.05, key="klo")
with kc3:
    kills_under_odds = st.number_input("Under odds", min_value=1.01, max_value=10.0,
                                        value=1.90, step=0.05, key="klu")

chk1, chk2, chk3, chk4 = st.columns(4)
with chk1:
    send_discord   = st.checkbox("📨 Discord",      value=True)
with chk2:
    send_ft5_sheet = st.checkbox("📊 FT5 Sheet",    value=True)
with chk3:
    send_win_sheet = st.checkbox("🏆 Winner Sheet", value=True)
with chk4:
    st.empty()

predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

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
        # V8.1: optional isotonic calibration (off by default for win model — backtest showed raw is better)
        if APPLY_WIN_CALIB and win_calibrator is not None:
            blue_raw = win_prob_raw[1]
            blue_cal = float(win_calibrator.predict([blue_raw])[0])
            win_prob_raw = [1 - blue_cal, blue_cal]
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
        b_speed      = team_kill_speed.get(blue_team_norm, KILL_SPEED_DEFAULT) if blue_team_norm else KILL_SPEED_DEFAULT
        r_speed      = team_kill_speed.get(red_team_norm,  KILL_SPEED_DEFAULT) if red_team_norm  else KILL_SPEED_DEFAULT
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
        # V8.1: optional isotonic calibration (on by default for FT5 — small improvement on 50-60% bands)
        if APPLY_FT5_CALIB and ft5_calibrator is not None:
            blue_raw = ft5_prob_raw[1]
            blue_cal = float(ft5_calibrator.predict([blue_raw])[0])
            ft5_prob_raw = [1 - blue_cal, blue_cal]
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
        ft5_conf_level, ft5_conf_desc, ft5_reasons, ft5_warnings = model_confidence(
            ft5_team_games.get(blue_team_norm,0) if blue_team_norm else 0,
            ft5_team_games.get(red_team_norm, 0) if red_team_norm  else 0,
            ft5_h2h_tot, b_early_form-r_early_form, b_early-r_early, b_agg-r_agg)

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

        st.divider()

        match_title = f"### {blue_team_name} vs {red_team_name}"
        if game_label: match_title += f" — {game_label}"
        st.markdown(match_title)

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
                st.info(f"💰 {win_blue_units}u — {win_blue_label}" if win_blue_units > 0 else "💰 ⛔ SKIP")
        with wc2:
            st.metric(f"🔴 {red_team_name}", f"{red_win_conf*100:.1f}%",
                      delta=f"Edge: {win_red_edge*100:.1f}%")
            st.write(f"Odds: {win_red_odds} | Implied: {win_red_impl*100:.1f}%")
            st.write(odds_label(win_red_odds))
            if red_win_conf > blue_win_conf:
                st.info(f"💰 {win_red_units}u — {win_red_label}" if win_red_units > 0 else "💰 ⛔ SKIP")

        if bdw is not None:
            dw = "🔵" if bdw > rdw else "🔴"
            dn = blue_team_name if bdw > rdw else red_team_name
            st.caption(f"⚖️ Draft-only: 🔵 {bdw*100:.1f}% vs 🔴 {rdw*100:.1f}% — {dw} {dn} has better draft")

        st.markdown(f"**📊 Confidence: {win_conf_level}** — {win_conf_desc}")

        with st.spinner("Fetching your tracker history..."):
            win_history = fetch_tracker_history(
                win_pick_conf, win_conf_level,
                st.secrets["GOOGLE_WINNER_SHEETS_ID"])
        if win_history:
            with st.expander("📈 Your Tracker History (Winner)", expanded=True):
                st.markdown(format_history(win_history, "Winner"))

        for r in win_reasons: st.write(f"✔ {r}")
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

        with st.expander("📊 Win Signal Breakdown", expanded=False):
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

        if blue or red:
            with st.expander("🏅 Champion Ratings", expanded=False):
                if blue:
                    st.markdown(f"**🔵 {blue_team_name}**")
                    for i, champ in enumerate(blue):
                        player  = blue_players[i] if i < len(blue_players) else ''
                        pos     = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
                        rc_val  = role_champ_rate.get((pos, champ.strip()), 0.5)
                        pc_val  = pc_rate.get((player.strip(), champ.strip()), 0.5) if player.strip() else rc_val
                        pcg     = pc_games_d.get((player.strip(), champ.strip()), 0) if player.strip() else 0
                        blended = PC_WEIGHT * pc_val + RC_WEIGHT * rc_val
                        cwr     = win_champ_rate.get(champ, 0.5)
                        rating  = rate_champ(cwr, blended)
                        lbl     = POS_LABELS[i] if i < len(POS_LABELS) else ''
                        name    = player if player.strip() else "Unknown"
                        gstr    = f"({pcg}g)" if pcg > 0 else ""
                        st.write(f"**{lbl}** {name} — {champ}: role {rc_val*100:.0f}% | player {pc_val*100:.0f}% {gstr} → {rating}")
                if red:
                    st.markdown(f"**🔴 {red_team_name}**")
                    for i, champ in enumerate(red):
                        player  = red_players[i] if i < len(red_players) else ''
                        pos     = POSITIONS[i] if i < len(POSITIONS) else 'unknown'
                        rc_val  = role_champ_rate.get((pos, champ.strip()), 0.5)
                        pc_val  = pc_rate.get((player.strip(), champ.strip()), 0.5) if player.strip() else rc_val
                        pcg     = pc_games_d.get((player.strip(), champ.strip()), 0) if player.strip() else 0
                        blended = PC_WEIGHT * pc_val + RC_WEIGHT * rc_val
                        cwr     = win_champ_rate.get(champ, 0.5)
                        rating  = rate_champ(cwr, blended)
                        lbl     = POS_LABELS[i] if i < len(POS_LABELS) else ''
                        name    = player if player.strip() else "Unknown"
                        gstr    = f"({pcg}g)" if pcg > 0 else ""
                        st.write(f"**{lbl}** {name} — {champ}: role {rc_val*100:.0f}% | player {pc_val*100:.0f}% {gstr} → {rating}")

        st.divider()

        st.markdown("### ⚔️ First to Five Kills")
        ft5_color = "🔵" if blue_ft5_conf > red_ft5_conf else "🔴"
        st.markdown(f"#### {ft5_color} Model pick: **{ft5_winner}**")

        # Strong red signal — informational only, no unit boost
        # Backtest: 66% red accuracy, 14.9% ROI when edge exists (normal calc_edge handles units)
        if ft5_strong_red:
            st.error(f"🚨 **STRONG RED SIGNAL** — Model blue confidence {blue_ft5_conf*100:.1f}% "
                     f"(below 48%). Backtest shows red picks in this range are **66% accurate** "
                     f"with **14.9% ROI**. Trust the unit recommendation below.")

        st.caption(f"⏱️ Est. ~{est_time:.1f} min ({faster_team} historically faster)")
        fc1, fc2 = st.columns(2)
        with fc1:
            st.metric(f"🔵 {blue_team_name}", f"{blue_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_blue_edge*100:.1f}%")
            st.write(f"Odds: {ft5_blue_odds} | Implied: {ft5_blue_impl*100:.1f}%")
            st.write(odds_label(ft5_blue_odds))
            if blue_ft5_conf > red_ft5_conf:
                st.info(f"💰 {ft5_blue_units}u — {ft5_blue_label}" if ft5_blue_units > 0 else "💰 ⛔ SKIP")
        with fc2:
            st.metric(f"🔴 {red_team_name}", f"{red_ft5_conf*100:.1f}%",
                      delta=f"Edge: {ft5_red_edge*100:.1f}%")
            st.write(f"Odds: {ft5_red_odds} | Implied: {ft5_red_impl*100:.1f}%")
            st.write(odds_label(ft5_red_odds))
            if red_ft5_conf > blue_ft5_conf:
                st.info(f"💰 {ft5_red_units}u — {ft5_red_label}" if ft5_red_units > 0 else "💰 ⛔ SKIP")

        if bdf is not None:
            df5 = "🔵" if bdf > rdf else "🔴"
            dn5 = blue_team_name if bdf > rdf else red_team_name
            st.caption(f"⚖️ Draft-only: 🔵 {bdf*100:.1f}% vs 🔴 {rdf*100:.1f}% — {df5} {dn5} more aggressive draft")

        st.markdown(f"**📊 Confidence: {ft5_conf_level}** — {ft5_conf_desc}")

        with st.spinner("Fetching your FT5 tracker history..."):
            ft5_history = fetch_tracker_history(
                ft5_pick_conf, ft5_conf_level,
                st.secrets["GOOGLE_SHEETS_ID"])
        if ft5_history:
            with st.expander("📈 Your Tracker History (FT5)", expanded=True):
                st.markdown(format_history(ft5_history, "FT5"))

        for r in ft5_reasons: st.write(f"✔ {r}")
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
                h_str     = "🟢 Strong" if abs(h2h_diff) >= 0.25 else ("🟡 Moderate" if abs(h2h_diff) >= 0.10 else "⚪ Weak")
                direction = f"favours 🔵 {blue_team_name}" if h2h_diff > 0 else f"favours 🔴 {red_team_name}"
                st.write(f"**Early H2H:** 🔵 {b_ft5_h2h}–{r_ft5_h2h} 🔴 — {h_str} {direction}")
            else:
                st.write("**Early H2H:** No history — ⚪ Neutral")

        if blue or red:
            with st.expander("🔥 Champion Aggression", expanded=False):
                if blue:
                    st.markdown(f"**🔵 {blue_team_name}**")
                    for i, champ in enumerate(blue):
                        player = blue_players[i] if i < len(blue_players) else ''
                        agg    = champ_aggression.get(champ, 0.5)
                        rating = rate_agg(agg)
                        lbl    = POS_LABELS[i] if i < len(POS_LABELS) else ''
                        name   = player if player.strip() else "Unknown"
                        st.write(f"**{lbl}** {name} — {champ}: {agg*100:.0f}% → {rating}")
                if red:
                    st.markdown(f"**🔴 {red_team_name}**")
                    for i, champ in enumerate(red):
                        player = red_players[i] if i < len(red_players) else ''
                        agg    = champ_aggression.get(champ, 0.5)
                        rating = rate_agg(agg)
                        lbl    = POS_LABELS[i] if i < len(POS_LABELS) else ''
                        name   = player if player.strip() else "Unknown"
                        st.write(f"**{lbl}** {name} — {champ}: {agg*100:.0f}% → {rating}")

        # =====================================================
        # V8.2: TOTAL KILLS OVER/UNDER PREDICTION
        # =====================================================
        if kls_point is not None and len(blue) == 5 and len(red) == 5:
            st.divider()
            st.markdown(f"### 💀 Total Kills Over/Under — Line {kills_line}")

            # Build features and get predictions
            patch_for_kls = current_patch.strip() if current_patch.strip() else (
                max(patch_release_dt.keys(), key=lambda p: patch_release_dt[p])
                if patch_release_dt else ""
            )
            league_for_kls = "LCK"  # neutral default — model uses hashed league_idx
            try:
                X_kls = build_kills_features(blue_team_norm, red_team_norm,
                                              blue, red, patch_for_kls, league_for_kls)
                # Point + interval estimates
                k_point = float(kls_point.predict(X_kls)[0])
                k_p10   = float(kls_q10.predict(X_kls)[0]) if kls_q10 is not None else k_point - 5
                k_p90   = float(kls_q90.predict(X_kls)[0]) if kls_q90 is not None else k_point + 5

                # O/U probability at user's line
                ou_result = predict_kills_ou(X_kls, kills_line)
                if ou_result:
                    prob_over, closest_line = ou_result
                    prob_under = 1 - prob_over

                    # Cap predictions at sensible bounds
                    prob_over  = min(max(prob_over,  0.05), 0.95)
                    prob_under = min(max(prob_under, 0.05), 0.95)

                    # Edge calculations
                    over_edge,  over_units,  over_label,  over_impl  = calc_edge(prob_over,  kills_over_odds)
                    under_edge, under_units, under_label, under_impl = calc_edge(prob_under, kills_under_odds)

                    # Display
                    kc1, kc2 = st.columns(2)
                    with kc1:
                        st.metric(f"⬆️ OVER {kills_line}", f"{prob_over*100:.1f}%",
                                  delta=f"Edge: {over_edge*100:.1f}%")
                        st.write(f"Odds: {kills_over_odds} | Implied: {over_impl*100:.1f}%")
                        st.write(odds_label(kills_over_odds))
                        if prob_over > prob_under:
                            st.info(f"💰 {over_units}u — {over_label}" if over_units > 0 else "💰 ⛔ SKIP")
                    with kc2:
                        st.metric(f"⬇️ UNDER {kills_line}", f"{prob_under*100:.1f}%",
                                  delta=f"Edge: {under_edge*100:.1f}%")
                        st.write(f"Odds: {kills_under_odds} | Implied: {under_impl*100:.1f}%")
                        st.write(odds_label(kills_under_odds))
                        if prob_under > prob_over:
                            st.info(f"💰 {under_units}u — {under_label}" if under_units > 0 else "💰 ⛔ SKIP")

                    # Context info
                    st.caption(f"📊 Model expects ~{k_point:.0f} kills (range {k_p10:.0f}-{k_p90:.0f}) | "
                               f"O/U classifier trained on line {closest_line}" +
                               (f" (your line {kills_line} is within range)" if abs(kills_line - closest_line) <= 0.5
                                else f" (your line {kills_line} extrapolated)"))
                    st.caption("⚠️ Kills model has modest edge (~5% over baseline). Best at lines 24.5-30.5. "
                               "Treat lower-confidence picks (<58%) as marginal.")
                else:
                    st.warning("Kills O/U model unavailable for this line")
            except Exception as e:
                st.warning(f"Could not compute kills prediction: {e}")

        st.divider()

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
            st.caption(" | ".join(status_parts))

        st.divider()
        st.caption("V8.1 | Win 65.4% (leak-free) / AUC 0.6944 | FT5 57.9% | Kills O/U at lines 22-30")
