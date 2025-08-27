import pandas as pd
import joblib
import json

# === Load Models ===
bat_model = joblib.load(r'G:\SLTC\8th sem\machine learning\mini project\T20\bat_model_t20.pkl')
bowl_model = joblib.load(r'G:\SLTC\8th sem\machine learning\mini project\T20\bowl_model_t20.pkl')

# === Load Data ===
players_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\players.csv')
batting_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\T20\T20_Batting.csv')
bowling_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\T20\T20_Bolling.csv')

# === Merge Player Info with Batting Data ===
batting_df = pd.merge(
    players_df[['PlayerID', 'Full Name', 'Playing Role', 'Age', 'Batting Style', 'Bowling Style']],
    batting_df,
    on='PlayerID',
    how='inner'
)
batting_df.rename(columns={
    'Runs': 'Runs_bat',
    'Ave': 'Avg_bat',
    'SR': 'SR_bat',
    'Mat': 'Mat_bat'
}, inplace=True)

batting_features = ['Runs_bat', 'Avg_bat', 'SR_bat', 'Mat_bat']
batting_df[batting_features] = batting_df[batting_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Predict Batters ===
batting_df['Predicted'] = bat_model.predict(batting_df[batting_features])
predicted_batters = batting_df[batting_df['Predicted'] == 1].drop_duplicates('PlayerID')

selected_batters = predicted_batters.head(6)
if len(selected_batters) < 6:
    fillers = batting_df[~batting_df['PlayerID'].isin(selected_batters['PlayerID'])].drop_duplicates('PlayerID').head(6 - len(selected_batters))
    selected_batters = pd.concat([selected_batters, fillers])

# === Merge Player Info with Bowling Data ===
bowling_df = pd.merge(
    players_df[['PlayerID', 'Full Name', 'Playing Role', 'Age', 'Batting Style', 'Bowling Style']],
    bowling_df,
    on='PlayerID',
    how='inner'
)
bowling_df.rename(columns={
    'Wkts': 'Wkts_bowl',
    'Econ': 'Econ_bowl',
    'Ave': 'Avg_bowl',
    'SR': 'SR_bowl'
}, inplace=True)

bowling_features = ['Wkts_bowl', 'Econ_bowl', 'Avg_bowl', 'SR_bowl']
bowling_df[bowling_features] = bowling_df[bowling_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Predict Bowlers ===
batter_ids = set(selected_batters['PlayerID'])
bowling_df['Predicted'] = bowl_model.predict(bowling_df[bowling_features])
predicted_bowlers = bowling_df[(bowling_df['Predicted'] == 1) & (~bowling_df['PlayerID'].isin(batter_ids))].drop_duplicates('PlayerID')

selected_bowlers = predicted_bowlers.head(5)
if len(selected_bowlers) < 5:
    fillers = bowling_df[~bowling_df['PlayerID'].isin(selected_bowlers['PlayerID']) & (~bowling_df['PlayerID'].isin(batter_ids))].drop_duplicates('PlayerID').head(5 - len(selected_bowlers))
    selected_bowlers = pd.concat([selected_bowlers, fillers])

# === Combine and Format Output ===
# Combine batters and bowlers DataFrames and drop duplicates
combined_df = pd.concat([selected_batters, selected_bowlers]).drop_duplicates('PlayerID')

# Use players_df info to get latest batting and bowling styles and roles
combined_df = combined_df.merge(
    players_df[['PlayerID', 'Batting Style', 'Bowling Style', 'Playing Role', 'Full Name', 'Age']],
    on='PlayerID',
    how='left',
    suffixes=('', '_player')
)

def determine_category(role):
    if pd.isna(role):
        return 'Player'
    role_lower = role.lower()
    if 'allrounder' in role_lower:
        return 'Allrounder'
    elif 'bat' in role_lower:
        return 'Batter'
    elif 'bowl' in role_lower:
        return 'Bowler'
    else:
        return 'Player'

# Prepare final output list
final_players = []
for _, row in combined_df.iterrows():
    final_players.append({
        'name': row['Full Name'],
        'role': row['Playing Role'],
        'age': row['Age'],
        'batting_style': row['Batting Style'] if pd.notna(row['Batting Style']) else 'N/A',
        'bowling_style': row['Bowling Style'] if pd.notna(row['Bowling Style']) else 'N/A',
        'category': determine_category(row['Playing Role'])
    })

# Print JSON list only (no other prints)
print(json.dumps(final_players, indent=4))
