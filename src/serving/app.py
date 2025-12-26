import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from utils.logger import logger

app = FastAPI(title="Network Security Classifier")

MODEL_PATH = "models/network_model.pkl"
model = joblib.load(MODEL_PATH)


class RequestData(BaseModel):
    features: list


@app.post("/predict")
def predict(data: RequestData):
    try:
        arr = np.array(data.features).reshape(1, -1)
        pred = model.predict(arr)[0]
        logger.info("Prediction successful")
        return {"prediction": int(pred)}
    except Exception as e:
        logger.exception("Prediction failed")
        raise e
