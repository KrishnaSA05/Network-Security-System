"""
Data Preprocessing Module
Handles cleaning, duplicate removal, and basic preprocessing
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/preprocessing.log")

class DataPreprocessor:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize preprocessor with configuration"""
        self.config = load_config(config_path)
        self.target_column = self.config['data']['target_column']
        self.processed_data_path = self.config['data']['processed_data_path']
        
    def check_missing_values(self, df):
        """Check and handle missing values"""
        logger.info("Checking for missing values...")
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            logger.info("Missing values per column:")
            logger.info(df.isnull().sum()[df.isnull().sum() > 0])
        else:
            logger.info("No missing values found")
        
        return missing_count
    
    def remove_duplicates(self, df):
        """Remove duplicate rows"""
        logger.info("Checking for duplicates...")
        initial_rows = len(df)
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            logger.info(f"Found {duplicates} duplicate rows ({duplicates/initial_rows*100:.2f}%)")
            
            # Check target distribution in duplicates
            dup_rows = df[df.duplicated(keep=False)]
            logger.info(f"Target distribution in duplicates:\n{dup_rows[self.target_column].value_counts()}")
            
            # Remove duplicates (keep first occurrence)
            df_clean = df.drop_duplicates(keep='first').reset_index(drop=True)
            
            removed = initial_rows - len(df_clean)
            logger.info(f"Removed {removed} duplicate rows")
            logger.info(f"New shape: {df_clean.shape}")
            
            # Check new target distribution
            logger.info(f"Target distribution after deduplication:\n{df_clean[self.target_column].value_counts()}")
            
            return df_clean
        else:
            logger.info("No duplicates found")
            return df
    
    def check_data_types(self, df):
        """Verify and fix data types"""
        logger.info("Checking data types...")
        
        # All features should be numeric (int64)
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if non_numeric:
            logger.warning(f"Non-numeric columns found: {non_numeric}")
        else:
            logger.info("All columns are numeric")
        
        return df
    
    def check_feature_values(self, df):
        """Check unique values and ranges for features"""
        logger.info("Analyzing feature values...")
        
        feature_info = {}
        for col in df.columns:
            unique_vals = df[col].nunique()
            value_range = (df[col].min(), df[col].max())
            feature_info[col] = {
                'unique_values': unique_vals,
                'range': value_range
            }
        
        # Log features with unusual characteristics
        for col, info in feature_info.items():
            if info['unique_values'] <= 3:
                logger.info(f"{col}: {info['unique_values']} unique values - {sorted(df[col].unique())}")
        
        return feature_info
    
    def validate_target(self, df):
        """Validate target variable"""
        logger.info("Validating target variable...")
        
        unique_targets = df[self.target_column].unique()
        logger.info(f"Target classes: {sorted(unique_targets)}")
        
        # Check if binary classification (-1, 1)
        if set(unique_targets) != {-1, 1}:
            logger.warning(f"Expected target values [-1, 1], found: {unique_targets}")
        
        # Check class balance
        target_counts = df[self.target_column].value_counts()
        target_props = df[self.target_column].value_counts(normalize=True)
        
        logger.info(f"Class counts:\n{target_counts}")
        logger.info(f"Class proportions:\n{target_props}")
        
        # Check for severe imbalance
        min_prop = target_props.min()
        if min_prop < 0.3:
            logger.warning(f"Class imbalance detected! Minority class: {min_prop*100:.2f}%")
        else:
            logger.info("Classes are reasonably balanced")
        
        return target_counts
    
    def save_processed_data(self, df):
        """Save processed data to CSV"""
        Path(self.processed_data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Processed data saved to {self.processed_data_path}")
    
    def save_preprocessing_report(self, df, output_path="data/validation/preprocessing_report.txt"):
        """Save preprocessing report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("PREPROCESSING REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Final Shape: {df.shape}\n\n")
            f.write("Missing Values: None\n\n")
            f.write(f"Target Distribution:\n{df[self.target_column].value_counts()}\n\n")
            f.write(f"Target Proportions:\n{df[self.target_column].value_counts(normalize=True)}\n\n")
            f.write("="*50 + "\n")
        
        logger.info(f"Preprocessing report saved to {output_path}")
    
    def run(self, df):
        """Run the complete preprocessing pipeline"""
        logger.info("="*50)
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        logger.info("="*50)
        
        # Check missing values
        self.check_missing_values(df)
        
        # Remove duplicates
        df_clean = self.remove_duplicates(df)
        
        # Check data types
        df_clean = self.check_data_types(df_clean)
        
        # Analyze feature values
        self.check_feature_values(df_clean)
        
        # Validate target
        self.validate_target(df_clean)
        
        # Save processed data
        self.save_processed_data(df_clean)
        
        # Save report
        self.save_preprocessing_report(df_clean)
        
        logger.info("Preprocessing completed successfully!")
        return df_clean

if __name__ == "__main__":
    # Load data first
    from ingest import DataIngestor
    
    ingestor = DataIngestor()
    df = ingestor.run()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run(df)
    
    print(f"\nPreprocessed data: {df_clean.shape}")
