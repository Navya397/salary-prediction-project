# scripts/batch_predict.py
import sys
import os
import json
import pandas as pd
import joblib

MODEL_PATH = os.path.join("models", "salary_prediction_pipeline.joblib")
FEATURE_CFG_PATH = os.path.join("models", "feature_config.json")

def main(input_csv, output_csv):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")

    if not os.path.exists(FEATURE_CFG_PATH):
        raise FileNotFoundError(f"Feature config not found at {FEATURE_CFG_PATH}. Train first.")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    target_col = cfg.get("target_col")
    num_features = cfg.get("num_features", [])
    cat_features = cfg.get("cat_features", [])
    expected_cols = num_features + cat_features

    df = pd.read_csv(input_csv)

    # Drop target if present in input
    if target_col in df.columns:
        df_features = df.drop(columns=[target_col]).copy()
    else:
        df_features = df.copy()

    # Ensure all expected columns exist
    missing = [c for c in expected_cols if c not in df_features.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    X = df_features[expected_cols]
    preds = model.predict(X)

    out = df.copy()
    out["Predicted_Salary"] = preds

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/batch_predict.py <input_csv> <output_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
