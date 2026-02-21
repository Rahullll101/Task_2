# ============================================================
#  IMPORTS
# ============================================================
import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_and_preprocess_data


# ============================================================
#  PATHS
# ============================================================
DATA_PATH = "../data/loan_data.csv"
MODEL_DIR = "../models"


# ============================================================
#  TRAIN ML MODEL (RANDOM FOREST)
# ============================================================
def train_ml():

    # --------------------------------------------------------
    # 1️ Load Data
    # --------------------------------------------------------
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        feature_columns
    ) = load_and_preprocess_data(DATA_PATH)

    # --------------------------------------------------------
    # 2️ Base Model
    # --------------------------------------------------------
    base_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )

    # --------------------------------------------------------
    # 3️ Hyperparameter Grid
    # --------------------------------------------------------
    param_dist = {
        "n_estimators": [200, 300, 500, 700],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
        "class_weight": ["balanced", "balanced_subsample"]
    }

    # --------------------------------------------------------
    # 4️ K-Fold Cross Validation
    # --------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=cv,
        verbose=2,       #  You will see progress
        n_jobs=-1,
        random_state=42
    )

    # --------------------------------------------------------
    # 5️ Train Model
    # --------------------------------------------------------
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    print("\n Best Parameters (Random Forest):")
    print(random_search.best_params_)

    print(f" OOB Score: {best_model.oob_score_:.4f}")

    # --------------------------------------------------------
    # 6️ Evaluation
    # --------------------------------------------------------
    y_pred = best_model.predict(X_test)

    metrics = {
        "model": "Random Forest (Tuned)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "oob_score": best_model.oob_score_
    }

    # --------------------------------------------------------
    # 7️ Save Model
    # --------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "random_forest.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    return {
        "model": best_model,
        "metrics": metrics,
        "scaler": scaler,
        "feature_columns": feature_columns
    }
