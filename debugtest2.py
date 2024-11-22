import requests
import pandas as pd

# Define the API endpoint and headers
url = 'https://api.football-data.org/v4/competitions/PL/matches'  # Premier League
headers = {'X-Auth-Token': '5412d5e47c5d4c34a313710ed20eecb7'}  # API token

try:
    # Make the API request
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses

    data = response.json()

    if 'matches' in data:
        matches = data['matches']

        # Extract relevant information
        match_data = []
        for match in matches:
            match_info = {
                'homeTeam': match['homeTeam']['name'],
                'awayTeam': match['awayTeam']['name'],
                'homeScore': match['score']['fullTime']['home'],
                'awayScore': match['score']['fullTime']['away'],
                'status': match['status'],
                'date': match['utcDate']
            }
            match_data.append(match_info)

        # Create a DataFrame
        df = pd.DataFrame(match_data)

        # Save the DataFrame to a CSV file
        df.to_csv('football_matches2.csv', index=False)
        print("Dataset saved as football_matches2.csv")
    else:
        print("'matches' key not found in the API response.")
except requests.exceptions.HTTPError as err:
    print(f"HTTP error occurred: {err}")
except Exception as e:
    print(f"An error occurred: {e}")