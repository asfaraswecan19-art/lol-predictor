"""Audit specific games to see what's actually different between proxy and v2."""
import pandas as pd

proxy = pd.read_csv("kill_timelines.csv")
v2 = pd.read_csv("kill_timelines_v2.csv")

# Stats first
print("=== Schema comparison ===")
print(f"Proxy rows: {len(proxy)}")
print(f"V2 rows: {len(v2)}")
print()
print("first_to_five value counts:")
print("PROXY:", proxy['first_to_five'].value_counts(dropna=False).to_dict())
print("V2:   ", v2['first_to_five'].value_counts(dropna=False).to_dict())
print()
print("is_ambiguous value counts:")
print("PROXY:", proxy['is_ambiguous'].value_counts(dropna=False).to_dict())
print("V2:   ", v2['is_ambiguous'].value_counts(dropna=False).to_dict())

# Merge by game_id, look at where they differ
m = proxy.merge(v2, on='game_id', how='inner', suffixes=('_p','_v'))
print(f"\nGames in both: {len(m)}")

# Where first_to_five disagrees
disagree_f25 = m[m['first_to_five_p'] != m['first_to_five_v']]
print(f"\nfirst_to_five disagrees: {len(disagree_f25)}")
if len(disagree_f25) > 0:
    print("Sample disagreements:")
    print(disagree_f25[['game_id','blue_team_p','red_team_p','first_to_five_p','first_to_five_v',
                        'blue_time_p','blue_time_v','red_time_p','red_time_v']].head(10).to_string())

# Where times differ significantly (precise replaced proxy)
m['blue_time_v_num'] = pd.to_numeric(m['blue_time_v'], errors='coerce')
m['red_time_v_num']  = pd.to_numeric(m['red_time_v'],  errors='coerce')
changed = m[(m['blue_time_p'] != m['blue_time_v_num']) | (m['red_time_p'] != m['red_time_v_num'])]
print(f"\nGames where times changed (proxy != v2): {len(changed)}")
print("Sample (first 10):")
print(changed[['game_id','blue_team_p','red_team_p','first_to_five_p','first_to_five_v',
               'blue_time_p','blue_time_v','red_time_p','red_time_v']].head(10).to_string())

# CRITICAL CHECK: in those games, does first_to_five agree with the time data?
# If first_to_five='red', red_time should be < blue_time.
print("\n=== Sanity: does first_to_five label agree with blue_time vs red_time? ===")
def check(df, prefix):
    df = df[df['first_to_five'+prefix].isin(['blue','red'])].copy()
    df['bt'] = pd.to_numeric(df['blue_time'+prefix], errors='coerce')
    df['rt'] = pd.to_numeric(df['red_time'+prefix], errors='coerce')
    df = df.dropna(subset=['bt','rt'])
    df['blue_first'] = df['bt'] < df['rt']
    df['agrees'] = ((df['first_to_five'+prefix] == 'blue') & df['blue_first']) | \
                   ((df['first_to_five'+prefix] == 'red')  & ~df['blue_first'])
    return df['agrees'].mean(), len(df)

p_acc, p_n = check(m, '_p')
v_acc, v_n = check(m, '_v')
print(f"PROXY: label agrees with times in {p_acc*100:.1f}% of cases (n={p_n})")
print(f"V2:    label agrees with times in {v_acc*100:.1f}% of cases (n={v_n})")
print("(If V2 is much lower than PROXY, the merger has a side-swap bug)")

# Show V2 rows where label disagrees with times
v2_disagree = m[m['first_to_five_v'].isin(['blue','red'])].copy()
v2_disagree['bt'] = pd.to_numeric(v2_disagree['blue_time_v'], errors='coerce')
v2_disagree['rt'] = pd.to_numeric(v2_disagree['red_time_v'], errors='coerce')
v2_disagree = v2_disagree.dropna(subset=['bt','rt'])
v2_disagree['blue_first'] = v2_disagree['bt'] < v2_disagree['rt']
v2_disagree['agrees'] = ((v2_disagree['first_to_five_v'] == 'blue') & v2_disagree['blue_first']) | \
                       ((v2_disagree['first_to_five_v'] == 'red')  & ~v2_disagree['blue_first'])
bad = v2_disagree[~v2_disagree['agrees']]
print(f"\nV2 rows where label DISAGREES with blue_time vs red_time: {len(bad)} of {len(v2_disagree)}")
print("Sample bad rows:")
print(bad[['game_id','blue_team_v','red_team_v','first_to_five_v',
           'blue_time_v','red_time_v']].head(10).to_string())
