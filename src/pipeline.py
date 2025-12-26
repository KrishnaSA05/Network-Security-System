import yaml
from data_pipeline.ingest import read_from_mongodb, save_raw_data, split_and_save
from data_pipeline.preprocess import preprocess_and_save
from training.train import train_model
from utils.logger import logger


def run_pipeline():
    logger.info("Pipeline started")

    # Ingest + split
    df = read_from_mongodb("network_security", "phishing_data")
    save_raw_data(df, "data/raw/phishing.csv")
    split_and_save(df, "data/processed/train.csv", "data/processed/test.csv")

    # Preprocess
    preprocess_and_save("data/processed/train.csv", "data/processed/test.csv")

    # Train
    train_model()

    logger.info("Pipeline finished successfully")


if __name__ == "__main__":
    run_pipeline()