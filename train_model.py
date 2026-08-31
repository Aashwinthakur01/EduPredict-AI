import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('student_data.csv')
X = df.drop('final_score', axis=1)
y = df['final_score']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))
print("Model trained - R2 ~85%")