# ============================================================
#  IMPORTS
# ============================================================
import os
import joblib
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocessing import load_and_preprocess_data


# ============================================================
#  PATHS
# ============================================================
DATA_PATH = "../data/loan_data.csv"
MODEL_DIR = "../models"


# ============================================================
#  TRAIN DL MODEL (MLP)
# ============================================================
def train_mlp():

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
    base_model = MLPClassifier(
        max_iter=800,
        early_stopping=True,
        random_state=42,
    )

    # --------------------------------------------------------
    # 3️ Hyperparameter Grid
    # --------------------------------------------------------
    param_dist = {
        "hidden_layer_sizes": [
            (128, 64),
            (256, 128),
            (256, 128, 64),
            (256, 128, 64, 32)
        ],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.0005],
        "batch_size": [32, 64]
    }

    # --------------------------------------------------------
    # 4️ K-Fold Cross Validation
    # --------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="f1",
        cv=cv,
        verbose=2,      #  Shows progress
        n_jobs=-1,
        random_state=42
    )

    # --------------------------------------------------------
    # 5️ Train Model
    # --------------------------------------------------------
    random_search.fit(X_train_scaled, y_train)

    best_model = random_search.best_estimator_

    print("\n Best Parameters (MLP):")
    print(random_search.best_params_)

    # --------------------------------------------------------
    # 6️ Evaluation
    # --------------------------------------------------------
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        "model": "MLPClassifier (Tuned)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    # --------------------------------------------------------
    # 7️ Save Model + Scaler + Columns
    # --------------------------------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "mlp_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    return {
        "model": best_model,
        "metrics": metrics,
        "scaler": scaler,
        "feature_columns": feature_columns
    }
