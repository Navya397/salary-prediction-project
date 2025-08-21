# scripts/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("data/salary_dataset_india.csv")

X = data.drop("Salary", axis=1)
y = data["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)
print(f"R² Score: {pipeline.score(X_test, y_test):.4f}")

joblib.dump(pipeline, "models/salary_prediction_pipeline.joblib")
print("✅ Model saved to models/salary_prediction_pipeline.joblib")
