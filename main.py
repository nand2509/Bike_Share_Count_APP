import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import pickle

df = pd.read_csv("hour.csv")

df.rename(columns={
    'instant': 'rec_id',
    'dteday': 'datetime',
    'holiday': 'is_holiday',
    'workingday': 'is_workingday',
    'weathersit': 'weather_condition',
    'hum': 'humidity',
    'mnth': 'month',
    'cnt': 'total_count',
    'hr': 'hour',
    'yr': 'year'
}, inplace=True)

df = df[['hour', 'is_holiday', 'weekday', 'total_count']]
df.is_holiday = df.is_holiday.astype('category')
df.weekday = df.weekday.astype('category')
df = pd.get_dummies(df)

x = df.drop(columns=['total_count'])
y = df['total_count']

ada = AdaBoostRegressor()
ada.fit(x, y)

pickle.dump(ada, open('model.pkl', 'wb'))
print("✅ model.pkl saved successfully!")
print("Columns used:", list(x.columns))
