# scripts/train.py
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join("data", "salary_dataset_india.csv")
MODEL_PATH = os.path.join("models", "salary_prediction_pipeline.joblib")
FEATURE_CFG_PATH = os.path.join("models", "feature_config.json")
RESULTS_DIR = "results"

def find_target_column(df):
    candidates = ["salary_lpa", "Salary", "salary", "target"]
    for c in df.columns:
        if c in candidates:
            return c
        if c.lower() in [x.lower() for x in candidates]:
            return c
    raise ValueError(
        f"Could not find a salary/target column. Available columns: {list(df.columns)}"
    )

def main():
    # Ensure folders exist
    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # 2) Identify target and features
    target_col = find_target_column(df)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split numeric vs categorical automatically
    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]

    if len(num_features) == 0 and len(cat_features) == 0:
        raise ValueError("No usable features found.")

    # 3) Preprocessing + model
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_features),  # with_mean=False works with sparse outputs
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="drop",
        sparse_threshold=0.1,  # keep result sparse if mostly categorical
    )

    model = LinearRegression()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Train
    pipeline.fit(X_train, y_train)

    # 6) Evaluate
    r2 = pipeline.score(X_test, y_test)
    print(f"R^2 on test: {r2:.4f}")

    # 7) Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # 8) Save feature config (so batch_predict knows what to expect)
    cfg = {
        "target_col": target_col,
        "num_features": num_features,
        "cat_features": cat_features,
    }
    with open(FEATURE_CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved feature config to {FEATURE_CFG_PATH}")

    # 9) Save metrics
    metrics_path = os.path.join(RESULTS_DIR, "training_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"R^2: {r2:.6f}\n")
        f.write(f"Target: {target_col}\n")
        f.write(f"Numeric features: {num_features}\n")
        f.write(f"Categorical features: {cat_features}\n")
    print(f"Saved training metrics to {metrics_path}")

if __name__ == "__main__":
    main()
