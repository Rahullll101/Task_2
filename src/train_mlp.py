import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import load_and_preprocess_data

DATA_PATH = "../data/loan_data.csv"
MODEL_DIR = "../models"

def train_mlp():
    _, _, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(DATA_PATH)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train_scaled,
        y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
        verbose=0
    )

    y_prob = model.predict(X_test_scaled).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # 🔹 Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "mlp_model.h5"))

    return {
        "model": "MLP (ANN)",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confidence": f"{int(y_prob.mean() * 100)}%"

    }

if __name__ == "__main__":
    print(train_mlp())
