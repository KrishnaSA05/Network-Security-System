import os
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


def read_from_mongodb(database: str, collection: str) -> pd.DataFrame:
    client = pymongo.MongoClient(MONGO_DB_URL)
    data = list(client[database][collection].find())
    df = pd.DataFrame(data)
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df


def save_raw_data(df: pd.DataFrame, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def split_and_save(df: pd.DataFrame, train_path: str, test_path: str, ratio: float = 0.2):
    train, test = train_test_split(df, test_size=ratio, random_state=42)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)


if __name__ == "__main__":
    df = read_from_mongodb("network_security", "phishing_data")
    save_raw_data(df, "data/raw/phishing.csv")
    split_and_save(df,
                   train_path="data/processed/train.csv",
                   test_path="data/processed/test.csv")