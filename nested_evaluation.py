
import time
import warnings
import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ----------------------------------
# 1. Load dataset
# ----------------------------------

df = pd.read_csv("data/data.csv")

X = df.drop(columns=["id", "Unnamed: 32", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

print("Dataset shape:", X.shape)
print("Class distribution:", y.value_counts().to_dict())


# ----------------------------------
# 2. Define classifiers
# ----------------------------------

models = {
    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}


# ----------------------------------
# 3. Define nested CV
# ----------------------------------

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = []


# ----------------------------------
# 4. Define experiment configurations
# ----------------------------------

configurations = {
    "Original": [None],
    "PCA": [1, 2, 3, 5, 10],
    "FastICA": [1, 2, 3, 5, 10]
}


# ----------------------------------
# 5. Nested cross-validation
# ----------------------------------

for method, components_list in configurations.items():

    print(f"\n========== {method} ==========")

    fold = 0

    for train_idx, test_idx in outer_cv.split(X, y):

        fold += 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        best_search = None
        best_score = -np.inf

        # Try every configuration inside training fold
        for n_components in components_list:

            steps = [
                ("scaler", StandardScaler())
            ]

            if method == "PCA":
                steps.append(
                    ("reduction", PCA(
                        n_components=n_components
                    ))
                )

            elif method == "FastICA":
                steps.append(
                    ("reduction", FastICA(
                        n_components=n_components,
                        random_state=42,
                        max_iter=1000,
                        tol=0.001
                    ))
                )

            steps.append(("classifier", LogisticRegression()))

            pipeline = Pipeline(steps)

            param_grid = []

            for model_name, model in models.items():
                param_grid.append({
                    "classifier": [clone(model)]
                })

            start = time.time()

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit="roc_auc",
                cv=inner_cv,
                n_jobs=-1,
                error_score="raise"
            )

            search.fit(X_train, y_train)

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_search = search

        # Evaluate selected configuration on outer test fold
        best_model = best_search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        fold_results = {
            "Method": method,
            "Fold": fold,
            "Best_Model": type(
                best_model.named_steps["classifier"]
            ).__name__,
            "Best_Components": (
                best_model.named_steps["reduction"].n_components
                if method != "Original"
                else 30
            ),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob)
        }

        results.append(fold_results)

        print(
            f"Fold {fold}: "
            f"{fold_results['Best_Model']} | "
            f"Components: {fold_results['Best_Components']} | "
            f"Accuracy: {fold_results['Accuracy']:.4f} | "
            f"ROC-AUC: {fold_results['ROC_AUC']:.4f}"
        )


# ----------------------------------
# 6. Save detailed results
# ----------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/nested_cv_results.csv",
    index=False
)


# ----------------------------------
# 7. Generate summary
# ----------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC"
]

summary = results_df.groupby("Method")[metrics].agg(
    ["mean", "std"]
).round(4)

print("\n========== NESTED CV SUMMARY ==========")
print(summary)

summary.to_csv("results/nested_cv_summary.csv")

print("\nSelected configurations:")
print(
    results_df.groupby(
        ["Method", "Best_Model", "Best_Components"]
    ).size()
)

print("\nResults saved successfully.")
