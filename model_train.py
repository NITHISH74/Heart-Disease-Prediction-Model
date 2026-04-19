"""
model_train.py
================
End-to-end training pipeline for the Heart Disease hybrid voting classifier.

Steps:
1. Load the UCI Heart dataset from the official GitHub mirror.
2. Build a VotingClassifier combining Random Forest, Logistic Regression, and SVM.
   (LR and SVM use their own StandardScaler inside sub-pipelines; RF uses raw features.)
3. Evaluate with Stratified K-Fold cross-validation (out-of-fold predictions for the
   confusion matrix; mean accuracy for reporting).
4. Fit on the full dataset for deployment and compute permutation-based feature importance.
5. Persist all artifacts needed by the Streamlit app (model, metrics, SHAP background data).

Run:  python model_train.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
)
ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifacts.joblib"
RANDOM_STATE = 42
N_SPLITS = 5
N_PERM_REPEATS = 15

# Column order must match the dataset and the Streamlit input form.
FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"


def load_heart_data(url: str = DATA_URL) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Fetch CSV from URL. Handles UTF-8 BOM and accidental whitespace in header names.
    """
    df = pd.read_csv(url, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Expected column '{TARGET_COLUMN}' in dataset. Got: {list(df.columns)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].values.ravel()
    return df, X.values.astype(np.float64), y


def build_voting_classifier() -> VotingClassifier:
    """
    Hybrid ensemble: soft voting averages predicted probabilities from three diverse models.

    - Random Forest: captures non-linear interactions without scaling.
    - Logistic Regression + StandardScaler: linear baseline on normalized features.
    - SVM (RBF) + StandardScaler: margin-based classifier; probability=True for soft voting.
    """
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lr_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, C=1.0),
            ),
        ]
    )

    svm_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("lr", lr_pipe), ("svc", svm_pipe)],
        voting="soft",
        weights=[2, 1, 1],
        n_jobs=-1,
    )
    return ensemble


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Loading data...")
    _, X, y = load_heart_data()

    clf = build_voting_classifier()

    # --- Stratified K-Fold: same class proportions in every fold (important for imbalance). ---
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"Running {N_SPLITS}-fold stratified cross-validation...")
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    print(f"CV accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # Out-of-fold predictions align with CV folds for an honest confusion matrix.
    y_pred_oof = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    oof_accuracy = accuracy_score(y, y_pred_oof)
    cm = confusion_matrix(y, y_pred_oof)
    print(f"OOF (stacked folds) accuracy: {oof_accuracy:.4f}")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Final model trained on all samples for deployment.
    print("Fitting final model on full data...")
    clf.fit(X, y)

    # Permutation importance: model-agnostic measure compatible with the full voting model.
    print("Computing permutation importance (may take a minute)...")
    perm = permutation_importance(
        clf,
        X,
        y,
        n_repeats=N_PERM_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="accuracy",
    )
    perm_means = perm.importances_mean

    # Background sample for SHAP (TreeExplainer on RF uses a modest reference set).
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(X.shape[0], size=min(100, X.shape[0]), replace=False)
    X_background = X[idx].copy()

    bundle = {
        "model": clf,
        "feature_names": FEATURE_COLUMNS,
        "target_name": TARGET_COLUMN,
        "class_labels": ("No disease (0)", "Heart disease (1)"),
        "cv_accuracy_mean": cv_mean,
        "cv_accuracy_std": cv_std,
        "oof_accuracy": float(oof_accuracy),
        "confusion_matrix": cm,
        "permutation_importance_mean": perm_means,
        "X_background": X_background,
        "random_state": RANDOM_STATE,
    }

    joblib.dump(bundle, ARTIFACT_PATH)
    print(f"Saved artifacts to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
