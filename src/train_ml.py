import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import load_and_preprocess_data

DATA_PATH = "../data/loan_data.csv"
MODEL_DIR = "../models"

def train_ml():
    X_train, X_test, _, _, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 🔹 Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "random_forest.pkl"))

    return {
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confidence": f"{int(y_prob.mean() * 100)}%"

    }

if __name__ == "__main__":
    print(train_ml())
