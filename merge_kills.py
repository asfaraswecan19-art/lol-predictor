"""
merge_kills.py — Join lolesports kill_data/*.json against proplay_matches.csv
to produce kill_timelines_v2.csv with PRECISE kill timings.

STRATEGY: Date-anchored co-occurrence matching.
We do NOT try to parse team names. Instead:
  1. Build a team alias table by date co-occurrence:
     Oracle says X plays on dates {D1, D2, ...}, lolesports says C plays
     on dates {D1, D2, D3, ...}. If overlap is high enough, X <-> C.
  2. Validate aliases: for each Oracle match, check the alias-derived
     pair appears in lolesports on the same date. Drop bad aliases.
  3. Merge using the validated alias table.

USAGE:
    python merge_kills.py

OUTPUT:
    kill_timelines_v2.csv  — merged FT5 training data (precise + proxy fallback)
    merge_report.txt       — diagnostics
    team_aliases.json      — the discovered Oracle->lol_code mapping
"""
import csv, json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

KILL_DATA_DIR = Path("kill_data")
PROPLAY_CSV   = Path("proplay_matches.csv")
GAME_IDS_CSV  = Path("game_ids.csv")
KILL_OLD_CSV  = Path("kill_timelines.csv")
OUT_CSV       = Path("kill_timelines_v2.csv")
REPORT_FILE   = Path("merge_report.txt")
ALIASES_FILE  = Path("team_aliases.json")

DATE_TOLERANCE_DAYS = 1
MIN_DATE_OVERLAP = 3
MIN_OVERLAP_RATIO = 0.30


def date_only(ts):
    if not ts: return None
    return ts[:10]


def _shift(d, days):
    try:
        return (datetime.fromisoformat(d) + timedelta(days=days)).strftime('%Y-%m-%d')
    except Exception:
        return d


def build_aliases(oracle_team_dates, lol_team_dates, oracle_match_dates, lol_match_dates):
    tentative = {}
    for o_team, o_dates in oracle_team_dates.items():
        best_score = 0; best_lol = None; best_overlap = 0
        for lol_code, lol_dates in lol_team_dates.items():
            overlap = len(o_dates & lol_dates)
            if overlap < MIN_DATE_OVERLAP: continue
            score = overlap / max(len(o_dates), 1)
            if score > best_score:
                best_score = score; best_lol = lol_code; best_overlap = overlap
        if best_lol and best_score >= MIN_OVERLAP_RATIO:
            tentative[o_team] = (best_lol, best_score, best_overlap)

    # Verify by matchup — but ONLY count dates where the lol_code has ANY game
    # (otherwise absence isn't evidence of wrong mapping)
    hits   = defaultdict(int)
    misses = defaultdict(int)
    for d, oracle_matches in oracle_match_dates.items():
        lol_on_d_all = (lol_match_dates.get(d, [])
                    + lol_match_dates.get(_shift(d, -1), [])
                    + lol_match_dates.get(_shift(d, 1),  []))
        lol_pairs = {frozenset({m['blue'], m['red']}) for m in lol_on_d_all}
        # Which lol codes appear on this date at all?
        lol_codes_on_d = set()
        for m in lol_on_d_all:
            lol_codes_on_d.add(m['blue']); lol_codes_on_d.add(m['red'])
        for om in oracle_matches:
            ab = tentative.get(om['blue']); ar = tentative.get(om['red'])
            if not ab or not ar: continue
            ab_code, ar_code = ab[0], ar[0]
            # Skip verification if the alias target wasn't even active on this date
            if ab_code not in lol_codes_on_d or ar_code not in lol_codes_on_d:
                continue
            if frozenset({ab_code, ar_code}) in lol_pairs:
                hits[om['blue']] += 1; hits[om['red']] += 1
            else:
                misses[om['blue']] += 1; misses[om['red']] += 1

    final = {}
    for o_team, (lol_code, score, overlap) in tentative.items():
        h = hits[o_team]; m = misses[o_team]; total = h + m
        if total == 0:
            # No verifiable matchups — keep based on overlap strength
            if score >= 0.5:
                final[o_team] = lol_code
        elif h / total >= 0.5:
            final[o_team] = lol_code
    return final


def main():
    if not KILL_DATA_DIR.exists():
        print(f"ERROR: {KILL_DATA_DIR} not found. Run fetch_kills.py first."); return
    if not PROPLAY_CSV.exists() or not GAME_IDS_CSV.exists():
        print(f"ERROR: missing {PROPLAY_CSV} or {GAME_IDS_CSV}."); return

    print("Loading proplay_matches.csv...")
    proplay_rows = list(csv.DictReader(open(PROPLAY_CSV, encoding='utf-8')))
    print(f"  {len(proplay_rows)} Oracle games")

    print("Loading game_ids.csv...")
    lol_rows = [r for r in csv.DictReader(open(GAME_IDS_CSV, encoding='utf-8'))
                if r.get('state') == 'completed']
    print(f"  {len(lol_rows)} lolesports completed games")

    oracle_team_dates = defaultdict(set)
    oracle_match_dates = defaultdict(list)
    for r in proplay_rows:
        d = date_only(r.get('date',''))
        if not d: continue
        oracle_team_dates[r['blue_team']].add(d)
        oracle_team_dates[r['red_team']].add(d)
        oracle_match_dates[d].append({'blue': r['blue_team'], 'red': r['red_team']})

    lol_team_dates = defaultdict(set)
    lol_match_dates = defaultdict(list)
    for r in lol_rows:
        d = date_only(r.get('start_time',''))
        if not d: continue
        lol_team_dates[r['blue_team']].add(d)
        lol_team_dates[r['red_team']].add(d)
        lol_match_dates[d].append({'blue': r['blue_team'], 'red': r['red_team']})

    print("\nBuilding team aliases via date co-occurrence + matchup verification...")
    aliases = build_aliases(oracle_team_dates, lol_team_dates,
                             oracle_match_dates, lol_match_dates)
    print(f"  Verified aliases: {len(aliases)}/{len(oracle_team_dates)}")
    ALIASES_FILE.write_text(json.dumps(aliases, indent=2), encoding='utf-8')

    print("\nLoading kill_data/*.json...")
    kill_files = list(KILL_DATA_DIR.glob("*.json"))
    print(f"  {len(kill_files)} files")
    games_per_match_pair = defaultdict(list)
    for fp in kill_files:
        try:
            data = json.loads(fp.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  WARN: cant read {fp.name}: {e}"); continue
        d = date_only(data.get('start_time'))
        bt = data.get('blue_team'); rt = data.get('red_team')
        if not d or not bt or not rt: continue
        games_per_match_pair[(d, frozenset({bt, rt}))].append({
            'game_id': data['game_id'], 'start_time': data['start_time'],
            'blue_team': bt, 'red_team': rt, 'summary': data['summary'],
        })
    # Sort each series by start_time so series_idx works correctly
    for key in games_per_match_pair:
        games_per_match_pair[key].sort(key=lambda g: g['start_time'])
    print(f"  Indexed {len(games_per_match_pair)} unique (date, pair) keys")

    proxy_by_gameid = {}
    if KILL_OLD_CSV.exists():
        print(f"\nLoading {KILL_OLD_CSV} as fallback...")
        for r in csv.DictReader(open(KILL_OLD_CSV, encoding='utf-8')):
            proxy_by_gameid[r['game_id']] = r
        print(f"  {len(proxy_by_gameid)} proxy records")

    print("\nMerging...")
    n_precise = n_proxy = n_blank = 0
    series_idx = defaultdict(int)
    unmatched_samples = []
    out_rows = []

    for r in proplay_rows:
        d = date_only(r.get('date',''))
        ob, orr = r['blue_team'], r['red_team']
        ab = aliases.get(ob); ar = aliases.get(orr)

        out = {
            'game_id':     r.get('game_id',''),
            'tournament':  r.get('league', r.get('tournament','')),
            'year':        r.get('year',''),
            'blue_team':   ob,
            'red_team':    orr,
            'blue_picks':  r.get('blue_picks',''),
            'red_picks':   r.get('red_picks',''),
            'blue_players':r.get('blue_players',''),
            'red_players': r.get('red_players',''),
        }

        precise = None
        if ab and ar and d:
            try:
                d0 = datetime.fromisoformat(d)
                for delta in range(-DATE_TOLERANCE_DAYS, DATE_TOLERANCE_DAYS + 1):
                    dd = (d0 + timedelta(days=delta)).strftime('%Y-%m-%d')
                    key = (dd, frozenset({ab, ar}))
                    games = games_per_match_pair.get(key, [])
                    if games:
                        idx = series_idx[key]
                        if idx < len(games):
                            precise = games[idx]; series_idx[key] += 1
                        else:
                            precise = games[-1]
                        break
            except Exception:
                pass

        if precise:
            s = precise['summary']
            same_side = (precise['blue_team'] == ab)
            f25 = s.get('first_to_five')
            bt5 = s.get('blue_time_to_5'); rt5 = s.get('red_time_to_5')
            # Swap times if Oracle's blue is lolesports' red
            if not same_side:
                bt5, rt5 = rt5, bt5
                if f25 == 'blue': f25 = 'red'
                elif f25 == 'red': f25 = 'blue'

            # Match proxy schema: when a team didn't reach 5 kills, cap at 30 minutes
            # and keep is_ambiguous=0 (the proxy uses this same convention).
            # ambiguous=1 only when we genuinely don't know who got there first
            # (i.e. neither side reached 5 kills).
            blue_unknown = (bt5 is None)
            red_unknown  = (rt5 is None)
            if blue_unknown and red_unknown:
                out['is_ambiguous'] = 1
                out['blue_time'] = 30.0
                out['red_time']  = 30.0
                out['first_to_five'] = ''
            else:
                out['is_ambiguous'] = 0
                out['blue_time'] = round(bt5, 4) if bt5 is not None else 30.0
                out['red_time']  = round(rt5, 4) if rt5 is not None else 30.0
                out['first_to_five'] = (f25 if f25 in ('blue','red') else '')

            bk10 = s.get('blue_kills_at_10min', 0) or 0
            rk10 = s.get('red_kills_at_10min',  0) or 0
            if not same_side: bk10, rk10 = rk10, bk10
            out['blue_kills10']  = bk10
            out['red_kills10']   = rk10
            out['blue_golddiff10'] = r.get('blue_golddiff10','')
            out['red_golddiff10']  = r.get('red_golddiff10','')
            out['blue_fb']         = r.get('blue_fb','')
            out['red_fb']          = r.get('red_fb','')
            try:
                dur = float(r.get('game_duration_min', 0))
                bk = s.get('blue_total_kills',0); rk = s.get('red_total_kills',0)
                if not same_side: bk, rk = rk, bk
                if dur > 0:
                    out['blue_ckpm'] = round(bk/dur, 4)
                    out['red_ckpm']  = round(rk/dur, 4)
                else:
                    out['blue_ckpm'] = out['red_ckpm'] = ''
            except (TypeError, ValueError):
                out['blue_ckpm'] = out['red_ckpm'] = ''
            n_precise += 1
        else:
            proxy = proxy_by_gameid.get(out['game_id'])
            if proxy:
                for col in ['first_to_five','blue_time','red_time','is_ambiguous',
                            'blue_kills10','red_kills10','blue_golddiff10','red_golddiff10',
                            'blue_fb','red_fb','blue_ckpm','red_ckpm']:
                    out[col] = proxy.get(col, '')
                n_proxy += 1
            else:
                for col in ['first_to_five','blue_time','red_time','is_ambiguous',
                            'blue_kills10','red_kills10','blue_golddiff10','red_golddiff10',
                            'blue_fb','red_fb','blue_ckpm','red_ckpm']:
                    out[col] = ''
                n_blank += 1
                if len(unmatched_samples) < 30:
                    unmatched_samples.append((d, ob, orr, out['game_id']))

        out_rows.append(out)

    fieldnames = ['game_id','tournament','year','blue_team','red_team',
                  'blue_picks','red_picks','blue_players','red_players',
                  'first_to_five','blue_time','red_time','is_ambiguous',
                  'blue_kills10','red_kills10','blue_golddiff10','red_golddiff10',
                  'blue_fb','red_fb','blue_ckpm','red_ckpm']
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(out_rows)

    total = len(out_rows)
    report = []
    report.append("="*60)
    report.append("  MERGE REPORT")
    report.append("="*60)
    report.append(f"Total Oracle games:          {total}")
    report.append(f"Precise (lolesports):        {n_precise} ({n_precise/total*100:.1f}%)")
    report.append(f"Proxy fallback:              {n_proxy} ({n_proxy/total*100:.1f}%)")
    report.append(f"No data at all:              {n_blank} ({n_blank/total*100:.1f}%)")
    report.append("")
    report.append(f"Team aliases discovered:     {len(aliases)}/{len(oracle_team_dates)}")
    report.append("")
    report.append("Sample aliases (Oracle -> lol_code):")
    for o, l in sorted(aliases.items())[:25]:
        report.append(f"  '{o}' -> '{l}'")
    report.append("")
    rejected = [t for t in oracle_team_dates if t not in aliases]
    if rejected:
        report.append(f"Teams without alias ({len(rejected)}):")
        for t in rejected[:15]:
            report.append(f"  '{t}' ({len(oracle_team_dates[t])} Oracle dates)")
        if len(rejected) > 15:
            report.append(f"  ... and {len(rejected)-15} more")
    report.append("")
    if unmatched_samples:
        report.append("Sample unmatched games (precise data not found):")
        for d, b, r, gid in unmatched_samples[:10]:
            report.append(f"  {d}  {b} vs {r}  ({gid})")

    text = "\n".join(report)
    REPORT_FILE.write_text(text, encoding='utf-8')
    print("\n" + text)
    print(f"\nWrote: {OUT_CSV}")
    print(f"Aliases: {ALIASES_FILE}")
    print(f"Report:  {REPORT_FILE}")


if __name__ == '__main__':
    main()
