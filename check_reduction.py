
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

df = pd.read_csv("data/data.csv")
X = df.drop(columns=["id", "Unnamed: 32", "diagnosis"])

X_scaled = StandardScaler().fit_transform(X)

for n in [1, 2, 3, 5, 10]:

    ica = FastICA(
        n_components=n,
        random_state=42,
        max_iter=1000,
        tol=0.001
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ica.fit_transform(X_scaled)

    convergence_warnings = [
        w for w in caught
        if issubclass(w.category, ConvergenceWarning)
    ]

    print(f"\nComponents: {n}")
    print("Iterations:", ica.n_iter_)
    print("Converged:", len(convergence_warnings) == 0)

    for warning in convergence_warnings:
        print("Warning:", warning.message)
