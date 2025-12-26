import joblib
import pandas as pd
from sklearn.metrics import f1_score
from utils.logger import logger


def check_model_performance(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    preds = model.predict(X)
    score = f1_score(y, preds)

    logger.info(f"Current model F1 score: {score}")
    return score
