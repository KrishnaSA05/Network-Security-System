"""
Script to load CSV data into MongoDB
Run this once to set up your MongoDB database
"""
import pandas as pd
from pymongo import MongoClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from utils.logger import setup_logger, load_config

logger = setup_logger("mongodb_setup", "logs/mongodb_setup.log")

def load_csv_to_mongodb():
    """Load phishing data from CSV to MongoDB"""
    config = load_config("src/config/config.yaml")
    
    # Load CSV
    csv_path = config['data']['raw_data_path']
    logger.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")
    
    # Connect to MongoDB
    mongo_config = config['data']['mongodb']
    client = MongoClient(mongo_config['connection_string'])
    db = client[mongo_config['database_name']]
    collection = db[mongo_config['collection_name']]
    
    # Clear existing data
    logger.info("Clearing existing data...")
    collection.delete_many({})
    
    # Insert data
    logger.info("Inserting data into MongoDB...")
    records = df.to_dict('records')
    collection.insert_many(records)
    
    # Verify
    count = collection.count_documents({})
    logger.info(f"✓ Successfully inserted {count} documents")
    
    client.close()
    logger.info("MongoDB setup complete!")

if __name__ == "__main__":
    load_csv_to_mongodb()
