import pandas as pd

results_path = r"results\predicted_salaries.csv"

df = pd.read_csv(results_path)
print(df.head(10))   # show first 10 rows
print("\nColumns in file:", df.columns.tolist())
