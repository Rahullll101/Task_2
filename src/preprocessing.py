import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

TARGET_COL = "loan_status"

def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    # Encode target if categorical
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    # Handle missing values
    X[numerical_cols] = SimpleImputer(strategy="median").fit_transform(X[numerical_cols])
    X[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[categorical_cols])

    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling (for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    scaler,
    list(X.columns)
    )


# print(load_and_preprocess_data("../data/loan_data.csv"))
