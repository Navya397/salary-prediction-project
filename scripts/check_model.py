# scripts/check_model.py
import os
import joblib

model_path = r"models\salary_prediction_pipeline.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

model = joblib.load(model_path)
print("Model loaded successfully:", type(model))
