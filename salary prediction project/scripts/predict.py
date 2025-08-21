import pandas as pd
import joblib
import os

# Paths
model_path = r"models\salary_prediction_pipeline.joblib"
data_path = "data/salary_dataset_india.csv"
   # change if your test file is different
results_path = r"results\predicted_salaries.csv"

# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Load model
model = joblib.load(model_path)

# Load test data
df = pd.read_csv(data_path)

# Run predictions
preds = model.predict(df)

# Save results
df["predicted_salary"] = preds
df.to_csv(results_path, index=False)

print(f"✅ Predictions saved to {results_path}")
