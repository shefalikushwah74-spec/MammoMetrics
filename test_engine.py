
import pandas as pd
from app.ml_engine import run_experiments

df = pd.read_csv("data/data.csv")

# Exclude ID and target from the feature list
features = [
    col for col in df.select_dtypes(include="number").columns
    if col.lower() not in ["id"]
]

experiment = run_experiments(
    df,
    target="diagnosis",
    feature_columns=features
)

print("\nBEST CONFIGURATION")
print(experiment["best_configuration"])

print("\nINDEPENDENT TEST METRICS")
print(experiment["test_metrics"])

print("\nFEATURES USED")
print(experiment["feature_names"])
