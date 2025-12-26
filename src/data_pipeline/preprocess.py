import os
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

TARGET_COLUMN = "class"


def build_preprocessor():
    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=3))
    ])


def preprocess_and_save(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].replace(-1, 0)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].replace(-1, 0)

    preprocessor = build_preprocessor()
    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    os.makedirs("data/features", exist_ok=True)
    joblib.dump(preprocessor, "data/features/preprocessor.pkl")

    np.save("data/features/train.npy", np.c_[X_train_p, y_train])
    np.save("data/features/test.npy", np.c_[X_test_p, y_test])


if __name__ == "__main__":
    preprocess_and_save(
        train_path="data/processed/train.csv",
        test_path="data/processed/test.csv"
    )