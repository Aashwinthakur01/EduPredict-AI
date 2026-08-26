import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Generate Synthetic Student Dataset
# --------------------------------------------------

np.random.seed(42)

number_of_students = 1000

attendance = np.random.uniform(50, 100, number_of_students)
study_hours = np.random.uniform(1, 8, number_of_students)
previous_marks = np.random.uniform(35, 95, number_of_students)
assignment_completion = np.random.uniform(40, 100, number_of_students)
sleep_hours = np.random.uniform(4, 9, number_of_students)
participation = np.random.uniform(30, 100, number_of_students)
backlogs = np.random.randint(0, 4, number_of_students)


# --------------------------------------------------
# 2. Calculate Performance Score
# --------------------------------------------------

performance = (
    attendance * 0.20
    + study_hours * 5
    + previous_marks * 0.30
    + assignment_completion * 0.15
    + sleep_hours * 2
    + participation * 0.10
    - backlogs * 3
)

# Add small random variation
noise = np.random.normal(0, 3, number_of_students)

performance = performance + noise

# Keep score between 0 and 100
performance = np.clip(performance, 0, 100)


# --------------------------------------------------
# 3. Create DataFrame
# --------------------------------------------------

data = pd.DataFrame({
    "attendance": attendance,
    "study_hours": study_hours,
    "previous_marks": previous_marks,
    "assignment_completion": assignment_completion,
    "sleep_hours": sleep_hours,
    "participation": participation,
    "backlogs": backlogs,
    "performance": performance
})


# --------------------------------------------------
# 4. Save Dataset
# --------------------------------------------------

data.to_csv("student_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total student records: {len(data)}")


# --------------------------------------------------
# 5. Separate Features and Target
# --------------------------------------------------

X = data.drop("performance", axis=1)
y = data["performance"]


# --------------------------------------------------
# 6. Split Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 7. Create Machine Learning Model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# --------------------------------------------------
# 8. Train Model
# --------------------------------------------------

model.fit(X_train, y_train)

print("Model training completed!")



predictions = model.predict(X_test)




mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

print("\nFeature Importance")
print("-------------------------")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance)

joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")