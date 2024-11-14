import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('2023 2024 PL.csv')


print(data.head())

print(data.isnull().sum())

data.dropna(inplace=True) 

#FTHG = Full Time Home Team Goals - FTAG = Full Time Away Team Goals
data['HomeGoals'] = data['FTHG']
data['AwayGoals'] = data['FTAG']
data['GoalDifference'] = data['FTHG'] - data['FTAG']

sns.countplot(x='FTR', data=data)  # visualisera 
plt.title('Match Outcomes')
plt.show()

