
import warnings
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, FastICA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
def prepare_dataset(df, target, feature_columns=None):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    data = df.copy().dropna(subset=[target])
    y_raw = data[target].astype(str)

    # Use selected features, or all numeric columns by default
    if feature_columns:
        missing = [col for col in feature_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

        X = data[feature_columns].copy()

        non_numeric = X.select_dtypes(exclude=np.number).columns.tolist()
        if non_numeric:
            raise ValueError(f"These selected columns are not numeric: {non_numeric}")
    else:
        X = data.drop(columns=[target]).select_dtypes(include=np.number)

    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    if X.shape[1] < 2:
        raise ValueError("Select at least two valid numeric features.")

    if y_raw.nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    if y_raw.nunique() > 20:
        raise ValueError("This classification pipeline supports up to 20 classes.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    if pd.Series(y).value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two samples.")

    return X, y, encoder






def run_experiments(df, target, components=(2, 5, 10), feature_columns=None):

    X, y, encoder = prepare_dataset(df, target, feature_columns)

    # Stratified independent holdout set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    min_cv_class = pd.Series(y_train).value_counts().min()
    n_splits = min(5, int(min_cv_class))

    if n_splits < 2:
        raise ValueError("Not enough training samples for cross-validation.")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=42
        )
    }

    # Use only component counts that are valid for this dataset.
    max_components = min(
        X_train.shape[1],
        X_train.shape[0] - 1
    )

    valid_components = sorted(set(
        c for c in components
        if 1 <= c <= max_components
    ))

    configurations = [("Original", None)]

    for c in valid_components:
        configurations.append(("PCA", c))
        configurations.append(("FastICA", c))

    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision_weighted",
        "Recall": "recall_weighted",
        "F1": "f1_weighted"
    }

    rows = []
    warnings.filterwarnings("ignore")

    for method, n_components in configurations:

        for model_name, classifier in models.items():

            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]

            if method == "PCA":
                steps.append((
                    "reducer",
                    PCA(n_components=n_components)
                ))

            elif method == "FastICA":
                steps.append((
                    "reducer",
                    FastICA(
                        n_components=n_components,
                        max_iter=1000,
                        tol=0.01,
                        random_state=42
                    )
                ))

            steps.append(("classifier", classifier))

            pipeline = Pipeline(steps)

            start = time.time()

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise"
            )

            elapsed = time.time() - start

            rows.append({
                "Method": method,
                "Components": (
                    X.shape[1]
                    if n_components is None
                    else n_components
                ),
                "Model": model_name,
                "Accuracy": scores["test_Accuracy"].mean(),
                "Precision": scores["test_Precision"].mean(),
                "Recall": scores["test_Recall"].mean(),
                "F1": scores["test_F1"].mean(),
                "Time": elapsed
            })

    results = pd.DataFrame(rows)

    # Select the best configuration based on CV F1.
    best_row = results.loc[results["F1"].idxmax()]

    best_method = best_row["Method"]
    best_components = int(best_row["Components"])
    best_model_name = best_row["Model"]

    best_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]

    if best_method == "PCA":
        best_steps.append((
            "reducer",
            PCA(n_components=best_components)
        ))

    elif best_method == "FastICA":
        best_steps.append((
            "reducer",
            FastICA(
                n_components=best_components,
                max_iter=1000,
                tol=0.01,
                random_state=42
            )
        ))

    best_steps.append((
        "classifier",
        models[best_model_name]
    ))

    best_pipeline = Pipeline(best_steps)
    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)

    test_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )
    }

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    )

    return {
        "results": results,
        "model": best_pipeline,
        "best_configuration": best_row.to_dict(),
        "test_metrics": test_metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "feature_names": list(X.columns),
        "class_names": list(encoder.classes_),
        "encoder": encoder,
        "target": target
    }
