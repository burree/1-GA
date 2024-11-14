import pandas as pd


data = pd.read_csv('2023 2024 PL.csv')


print(data.head())

print(data.isnull().sum())

data.dropna(inplace=True) 

#FTHG = Full Time Home Team Goals - FTAG = Full Time Away Team Goals
data['HomeGoals'] = data['FTHG']
data['AwayGoals'] = data['FTAG']