import requests
import json
import pandas as pd

uri = 'https://api.football-data.org/v4/matches'
headers = { 'X-Auth-Token': '5412d5e47c5d4c34a313710ed20eecb7' }



url = 'https://api.football-data.org/v4/competitions/PL/matches'  #Premier League
headers = {'X-Auth-Token': '5412d5e47c5d4c34a313710ed20eecb7'}  #api token


response = requests.get(url, headers=headers)


if response.status_code == 200:
    data = response.json()

    if 'matches' in data:
        matches = data['matches']
        
        
        df = pd.DataFrame(matches)
        
        
        if all(col in df.columns for col in ['homeTeam', 'awayTeam', 'score', 'status']):
            df_cleaned = df[['homeTeam', 'awayTeam', 'score', 'status']]

            df_cleaned.dropna(inplace=True)

            df_cleaned.to_csv('football_matches.csv', index=False)
            print("Dataset saved as football_matches.csv")
        else:
            print("Required columns are missing in the response.")
    else:
        print("'matches' key not found in the API response.")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

print(data)
