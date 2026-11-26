
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/data.csv")
X = df.drop(columns=["id", "Unnamed: 32", "diagnosis"])
y = (df["diagnosis"] == "M").astype(int)

# Independent train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Selected model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Cross-validation on training data only
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
)

print("TRAINING CROSS-VALIDATION")
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    print(f"{metric}: {scores['test_' + metric].mean():.4f}")

# Train final model on complete training set
model.fit(X_train, y_train)

# Evaluate once on untouched test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nINDEPENDENT TEST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Benign", "Malignant"]
))
