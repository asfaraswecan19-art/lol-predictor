import pandas as pd
g = pd.read_csv('game_ids.csv')
print("columns:", list(g.columns))
print(g.head(8).to_string())