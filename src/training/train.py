import os
import joblib
import numpy as np
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.logger import logger


def load_data():
    train = np.load("data/features/train.npy")
    test = np.load("data/features/test.npy")
    return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]


def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report


def save_model(model):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/network_model.pkl")


if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline")

        X_train, y_train, X_test, y_test = load_data()
        model, report = train_model(X_train, y_train, X_test, y_test)

        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])
        save_model(model)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.exception("Training failed")
        raise e
