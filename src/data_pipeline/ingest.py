"""
Data Ingestion Module
Supports loading from CSV or MongoDB
"""
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/data_ingestion.log")

class DataIngestor:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize data ingestor with configuration"""
        self.config = load_config(config_path)
        self.data_source = self.config['data']['source']
        self.target_column = self.config['data']['target_column']
        
        if self.data_source == "csv":
            self.raw_data_path = self.config['data']['raw_data_path']
        elif self.data_source == "mongodb":
            self.mongo_config = self.config['data']['mongodb']
        
    def load_from_csv(self):
        """Load data from CSV file"""
        try:
            logger.info(f"Loading data from CSV: {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"✓ Data loaded successfully from CSV. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def load_from_mongodb(self):
        """Load data from MongoDB"""
        try:
            from pymongo import MongoClient
            
            logger.info("Connecting to MongoDB...")
            
            # Connect to MongoDB
            client = MongoClient(self.mongo_config['connection_string'])
            db = client[self.mongo_config['database_name']]
            collection = db[self.mongo_config['collection_name']]
            
            # Check collection size
            total_docs = collection.count_documents({})
            logger.info(f"Found {total_docs} documents in collection")
            
            # Load data in batches
            batch_size = self.mongo_config.get('batch_size', 1000)
            logger.info(f"Loading data in batches of {batch_size}...")
            
            # Convert MongoDB cursor to DataFrame
            cursor = collection.find({}, {"_id": 0})  # Exclude MongoDB _id field
            df = pd.DataFrame(list(cursor))
            
            # Close connection
            client.close()
            
            logger.info(f"✓ Data loaded successfully from MongoDB. Shape: {df.shape}")
            return df
            
        except ImportError:
            logger.error("pymongo not installed. Run: pip install pymongo dnspython")
            raise
        except Exception as e:
            logger.error(f"Error loading MongoDB data: {str(e)}")
            raise
    
    def load_data(self):
        """Load data from configured source"""
        logger.info(f"Data source: {self.data_source.upper()}")
        
        if self.data_source == "csv":
            return self.load_from_csv()
        elif self.data_source == "mongodb":
            return self.load_from_mongodb()
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def initial_validation(self, df):
        """Perform initial data validation checks"""
        logger.info("Starting initial data validation...")
        
        if df.empty:
            logger.error("DataFrame is empty!")
            raise ValueError("Empty DataFrame")
        
        if self.target_column not in df.columns:
            logger.error(f"Target column '{self.target_column}' not found!")
            raise ValueError(f"Missing target column: {self.target_column}")
        
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Number of columns: {len(df.columns)}")
        logger.info(f"Missing values: {df.isnull().sum().sum()}")
        logger.info(f"Duplicate rows: {df.duplicated().sum()}")
        
        target_dist = df[self.target_column].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")
        
        logger.info(f"Data types:\n{df.dtypes.value_counts()}")
        
        return True
    
    def save_initial_report(self, df, output_path="data/validation/initial_report.txt"):
        """Save initial data report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("INITIAL DATA REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Data Source: {self.data_source.upper()}\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n\n")
            f.write("Data Types:\n")
            f.write(str(df.dtypes) + "\n\n")
            f.write("Missing Values:\n")
            f.write(str(df.isnull().sum()) + "\n\n")
            f.write(f"Duplicate Rows: {df.duplicated().sum()}\n\n")
            f.write("Target Distribution:\n")
            f.write(str(df[self.target_column].value_counts()) + "\n\n")
            f.write("Statistical Summary:\n")
            f.write(str(df.describe()) + "\n")
        
        logger.info(f"Initial report saved to {output_path}")
    
    def run(self):
        """Run the complete ingestion pipeline"""
        logger.info("="*50)
        logger.info("STARTING DATA INGESTION PIPELINE")
        logger.info("="*50)
        
        df = self.load_data()
        self.initial_validation(df)
        self.save_initial_report(df)
        
        logger.info("✓ Data ingestion completed successfully!")
        return df

if __name__ == "__main__":
    ingestor = DataIngestor()
    df = ingestor.run()
    print(f"\nData loaded: {df.shape}")
    print(f"Target distribution:\n{df['Result'].value_counts()}")
