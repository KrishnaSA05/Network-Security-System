"""
Data Validation Module
Performs comprehensive data quality checks
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/validation.log")

class DataValidator:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize validator with configuration"""
        self.config = load_config(config_path)
        self.target_column = self.config['data']['target_column']
        
    def validate_schema(self, df, expected_columns=None):
        """Validate dataframe schema"""
        logger.info("Validating schema...")
        
        if expected_columns:
            missing_cols = set(expected_columns) - set(df.columns)
            extra_cols = set(df.columns) - set(expected_columns)
            
            if missing_cols:
                logger.error(f"Missing columns: {missing_cols}")
                return False
            if extra_cols:
                logger.warning(f"Extra columns: {extra_cols}")
        
        logger.info("Schema validation passed")
        return True
    
    def validate_data_quality(self, df):
        """Perform comprehensive data quality checks"""
        logger.info("Validating data quality...")
        
        issues = []
        
        # Check for NaN/Inf values
        if df.isnull().any().any():
            issues.append("Contains missing values")
        
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            issues.append("Contains infinite values")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]
        if constant_cols:
            issues.append(f"Constant columns: {constant_cols}")
        
        # Check for high cardinality in supposedly categorical features
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95 and df[col].nunique() > 100:
                    logger.warning(f"High cardinality in {col}: {df[col].nunique()} unique values")
        
        if issues:
            logger.warning(f"Data quality issues found: {issues}")
            return False, issues
        else:
            logger.info("All data quality checks passed")
            return True, []
    
    def validate_value_ranges(self, df):
        """Validate that feature values are within expected ranges"""
        logger.info("Validating value ranges...")
        
        # For phishing dataset, most features should be in {-1, 0, 1}
        issues = []
        
        for col in df.columns:
            if col != self.target_column:
                unique_vals = set(df[col].unique())
                expected_vals = {-1, 0, 1}
                
                if not unique_vals.issubset(expected_vals):
                    unexpected = unique_vals - expected_vals
                    issues.append(f"{col}: unexpected values {unexpected}")
        
        if issues:
            logger.warning(f"Value range issues: {issues}")
            return False, issues
        else:
            logger.info("Value range validation passed")
            return True, []
    
    def validate_correlations(self, df):
        """Check for highly correlated features"""
        logger.info("Checking feature correlations...")
        
        # Calculate correlation matrix
        corr_matrix = df.drop(columns=[self.target_column]).corr().abs()
        
        # Find high correlations (upper triangle only)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation > 0.95
        high_corr_pairs = []
        for column in upper.columns:
            correlated = upper[column][upper[column] > 0.95].index.tolist()
            if correlated:
                for corr_col in correlated:
                    high_corr_pairs.append((column, corr_col, upper.loc[corr_col, column]))
        
        if high_corr_pairs:
            logger.warning(f"Found {len(high_corr_pairs)} highly correlated feature pairs (r > 0.95)")
            for col1, col2, corr_val in high_corr_pairs:
                logger.warning(f"  {col1} - {col2}: {corr_val:.3f}")
        else:
            logger.info("No highly correlated features found (threshold: 0.95)")
        
        return high_corr_pairs
    
    def generate_validation_report(self, df, output_path="data/validation/validation_report.txt"):
        """Generate comprehensive validation report"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("DATA VALIDATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Basic info
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
            
            # Missing values
            f.write("Missing Values:\n")
            missing = df.isnull().sum()
            if missing.sum() == 0:
                f.write("  None\n\n")
            else:
                f.write(f"{missing[missing > 0]}\n\n")
            
            # Duplicates
            f.write(f"Duplicate Rows: {df.duplicated().sum()}\n\n")
            
            # Feature statistics
            f.write("Feature Statistics:\n")
            for col in df.columns:
                unique = df[col].nunique()
                f.write(f"  {col}: {unique} unique values\n")
            
            f.write("\n" + "="*50 + "\n")
        
        logger.info(f"Validation report saved to {output_path}")
    
    def run(self, df):
        """Run complete validation pipeline"""
        logger.info("="*50)
        logger.info("STARTING DATA VALIDATION PIPELINE")
        logger.info("="*50)
        
        validation_results = {
            'schema': True,
            'quality': True,
            'ranges': True,
            'issues': []
        }
        
        # Schema validation
        validation_results['schema'] = self.validate_schema(df)
        
        # Data quality validation
        quality_ok, quality_issues = self.validate_data_quality(df)
        validation_results['quality'] = quality_ok
        validation_results['issues'].extend(quality_issues)
        
        # Value range validation
        ranges_ok, range_issues = self.validate_value_ranges(df)
        validation_results['ranges'] = ranges_ok
        validation_results['issues'].extend(range_issues)
        
        # Check correlations
        high_corr = self.validate_correlations(df)
        
        # Generate report
        self.generate_validation_report(df)
        
        # Final verdict
        all_passed = validation_results['schema'] and validation_results['quality'] and validation_results['ranges']
        
        if all_passed:
            logger.info("✓ All validation checks passed!")
        else:
            logger.warning(f"✗ Validation issues found: {validation_results['issues']}")
        
        logger.info("Validation completed!")
        return validation_results

if __name__ == "__main__":
    # Load processed data
    processed_data_path = "data/processed/cleaned_data.csv"
    df = pd.read_csv(processed_data_path)
    
    # Validate
    validator = DataValidator()
    results = validator.run(df)
    
    print(f"\nValidation Results: {'PASSED' if all(results.values()) else 'FAILED'}")
