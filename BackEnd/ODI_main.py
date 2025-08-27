import pandas as pd
import joblib
import json

# === Load Models ===
bat_model = joblib.load(r'G:\SLTC\8th sem\machine learning\mini project\ODI\bat_model_odi.pkl')
bowl_model = joblib.load(r'G:\SLTC\8th sem\machine learning\mini project\ODI\bowl_model_odi.pkl')

# === Load Data ===
players_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\players.csv')
batting_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\ODI\ODI_batting.csv')
bowling_df = pd.read_csv(r'G:\SLTC\8th sem\machine learning\mini project\ODI\ODI_bawling.csv')

# === Merge Player Info with Batting Data ===
batting_df = pd.merge(
    players_df[['PlayerID', 'Full Name', 'Playing Role', 'Age', 'Batting Style', 'Bowling Style']],
    batting_df,
    on='PlayerID',
    how='inner'
)
batting_df.rename(columns={'Runs': 'Runs_bat', 'Ave': 'Avg_bat', 'SR': 'SR_bat', 'Mat': 'Mat_bat'}, inplace=True)
batting_features = ['Runs_bat', 'Avg_bat', 'SR_bat', 'Mat_bat']
batting_df[batting_features] = batting_df[batting_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Merge Player Info with Bowling Data ===
bowling_df = pd.merge(
    players_df[['PlayerID', 'Full Name', 'Playing Role', 'Age', 'Batting Style', 'Bowling Style']],
    bowling_df,
    on='PlayerID',
    how='inner'
)
bowling_df.rename(columns={'Wkts': 'Wkts_bowl', 'Econ': 'Econ_bowl', 'Ave': 'Avg_bowl', 'SR': 'SR_bowl'}, inplace=True)
bowling_features = ['Wkts_bowl', 'Econ_bowl', 'Avg_bowl', 'SR_bowl']
bowling_df[bowling_features] = bowling_df[bowling_features].apply(pd.to_numeric, errors='coerce').fillna(0)

# === Predict Batters ===
batting_df['Predicted'] = bat_model.predict(batting_df[batting_features])
predicted_batters = batting_df[batting_df['Predicted'] == 1].drop_duplicates('PlayerID')

# Select up to 6 batters
selected_batters = predicted_batters.head(6)
if len(selected_batters) < 6:
    remaining_needed = 6 - len(selected_batters)
    fillers = batting_df[~batting_df['PlayerID'].isin(selected_batters['PlayerID'])].drop_duplicates('PlayerID').head(remaining_needed)
    selected_batters = pd.concat([selected_batters, fillers])

# === Predict Bowlers ===
batter_ids = set(selected_batters['PlayerID'])
bowling_df['Predicted'] = bowl_model.predict(bowling_df[bowling_features])
predicted_bowlers = bowling_df[(bowling_df['Predicted'] == 1) & (~bowling_df['PlayerID'].isin(batter_ids))].drop_duplicates('PlayerID')

# Select up to 5 bowlers
selected_bowlers = predicted_bowlers.head(5)
if len(selected_bowlers) < 5:
    needed = 5 - len(selected_bowlers)
    fillers = bowling_df[~bowling_df['PlayerID'].isin(set(selected_bowlers['PlayerID']).union(batter_ids))].drop_duplicates('PlayerID').head(needed)
    selected_bowlers = pd.concat([selected_bowlers, fillers])

# === Combine batters and bowlers for final team ===
final_team = pd.concat([selected_batters, selected_bowlers]).drop_duplicates('PlayerID').head(11)

# === Determine category helper ===
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

# === Merge back styles and age for final output ===
final_team = final_team.merge(
    players_df[['PlayerID', 'Batting Style', 'Bowling Style', 'Age']],
    on='PlayerID',
    how='left',
    suffixes=('', '_player')
)

# === Prepare output list ===
output_list = []
for _, row in final_team.iterrows():
    output_list.append({
        'playerId': str(row['PlayerID']),
        'name': row['Full Name'],
        'role': row['Playing Role'] if pd.notna(row['Playing Role']) else 'N/A',
        'age': row['Age'] if pd.notna(row['Age']) else 'N/A',
        'batting_style': row['Batting Style'] if pd.notna(row['Batting Style']) else 'N/A',
        'bowling_style': row['Bowling Style'] if pd.notna(row['Bowling Style']) else 'N/A',
        'category': determine_category(row['Playing Role'])
    })

# === Print JSON only for Flask ===
print(json.dumps(output_list, indent=4))
