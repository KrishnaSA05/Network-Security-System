"""
Feature Engineering Module
Creates engineered features based on domain knowledge
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/feature_engineering.log")

class FeatureEngineer:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize feature engineer with configuration"""
        self.config = load_config(config_path)
        self.drop_features = self.config['features']['drop_features']
        self.feature_data_path = self.config['data']['feature_data_path']
        
    def drop_weak_features(self, df):
        """Drop features with weak correlation or high collinearity"""
        logger.info(f"Dropping weak/collinear features: {self.drop_features}")
        
        # Check if features exist
        existing_features = [f for f in self.drop_features if f in df.columns]
        missing_features = [f for f in self.drop_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Features not found in dataset: {missing_features}")
        
        if existing_features:
            df_clean = df.drop(columns=existing_features)
            logger.info(f"Dropped {len(existing_features)} features")
            logger.info(f"New shape: {df_clean.shape}")
        else:
            df_clean = df.copy()
            logger.warning("No features to drop")
        
        return df_clean
    
    def create_ssl_domain_trust(self, df):
        """
        Create SSLDomainTrust feature
        Interaction between SSL state and domain registration length
        """
        logger.info("Creating SSLDomainTrust feature...")
        
        df['SSLDomainTrust'] = df['SSLfinal_State'] * df['Domain_registeration_length']
        
        logger.info(f"SSLDomainTrust - Unique values: {df['SSLDomainTrust'].nunique()}")
        return df
    
    def create_url_suspicion_score(self, df):
        """
        Create URLSuspicionScore feature
        Combination of URL-related suspicious indicators
        """
        logger.info("Creating URLSuspicionScore feature...")
        
        df['URLSuspicionScore'] = (
            df['having_IP_Address'] + 
            df['URL_Length'] + 
            df['having_At_Symbol'] + 
            df['Prefix_Suffix']
        ) / 4.0
        
        logger.info(f"URLSuspicionScore - Range: [{df['URLSuspicionScore'].min():.2f}, {df['URLSuspicionScore'].max():.2f}]")
        return df
    
    def create_content_credibility(self, df):
        """
        Create ContentCredibility feature
        Based on page content and anchor characteristics
        """
        logger.info("Creating ContentCredibility feature...")
        
        df['ContentCredibility'] = (
            df['URL_of_Anchor'] + 
            df['Links_in_tags'] + 
            df['SFH']
        ) / 3.0
        
        logger.info(f"ContentCredibility - Range: [{df['ContentCredibility'].min():.2f}, {df['ContentCredibility'].max():.2f}]")
        return df
    
    def create_domain_reputation(self, df):
        """
        Create DomainReputation feature
        Based on domain age and DNS records
        """
        logger.info("Creating DomainReputation feature...")
        
        df['DomainReputation'] = (
            df['age_of_domain'] + 
            df['DNSRecord'] + 
            df['web_traffic']
        ) / 3.0
        
        logger.info(f"DomainReputation - Range: [{df['DomainReputation'].min():.2f}, {df['DomainReputation'].max():.2f}]")
        return df
    
    def create_security_features_count(self, df):
        """
        Count positive security indicators
        """
        logger.info("Creating SecurityFeaturesCount feature...")
        
        security_features = [
            'SSLfinal_State', 'Domain_registeration_length', 
            'HTTPS_token', 'age_of_domain', 'DNSRecord'
        ]
        
        df['SecurityFeaturesCount'] = df[security_features].apply(
            lambda x: (x == 1).sum(), axis=1
        )
        
        logger.info(f"SecurityFeaturesCount - Range: [{df['SecurityFeaturesCount'].min()}, {df['SecurityFeaturesCount'].max()}]")
        return df
    
    def create_suspicious_features_count(self, df):
        """
        Count suspicious indicators
        """
        logger.info("Creating SuspiciousFeaturesCount feature...")
        
        suspicious_features = [
            'having_IP_Address', 'having_At_Symbol', 'Prefix_Suffix',
            'having_Sub_Domain', 'Request_URL', 'Abnormal_URL', 'Redirect'
        ]
        
        df['SuspiciousFeaturesCount'] = df[suspicious_features].apply(
            lambda x: (x == -1).sum(), axis=1
        )
        
        logger.info(f"SuspiciousFeaturesCount - Range: [{df['SuspiciousFeaturesCount'].min()}, {df['SuspiciousFeaturesCount'].max()}]")
        return df
    
    def create_ssl_anchor_interaction(self, df):
        """
        Create SSLAnchorInteraction feature
        Interaction between SSL and URL anchor characteristics
        """
        logger.info("Creating SSLAnchorInteraction feature...")
        
        df['SSLAnchorInteraction'] = df['SSLfinal_State'] * df['URL_of_Anchor']
        
        logger.info(f"SSLAnchorInteraction - Unique values: {df['SSLAnchorInteraction'].nunique()}")
        return df
    
    def verify_features(self, df):
        """Verify all engineered features were created"""
        logger.info("Verifying engineered features...")
        
        expected_features = self.config['features']['engineered_features']
        
        created = []
        missing = []
        
        for feature in expected_features:
            if feature in df.columns:
                created.append(feature)
            else:
                missing.append(feature)
        
        logger.info(f"Created features ({len(created)}): {created}")
        
        if missing:
            logger.warning(f"Missing features ({len(missing)}): {missing}")
        
        return len(missing) == 0
    
    def save_feature_data(self, df):
        """Save data with engineered features"""
        Path(self.feature_data_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.feature_data_path, index=False)
        logger.info(f"Feature data saved to {self.feature_data_path}")
    
    def save_feature_report(self, df_original, df_final):
        """Save feature engineering report"""
        report_path = Path("data/validation") / "feature_report.txt"
    
        with open(report_path, 'w', encoding='utf-8') as f:  # ← Added encoding='utf-8'
            f.write("="*60 + "\n")
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("="*60 + "\n\n")
        
            f.write(f"Original shape: {df_original.shape}\n")
            f.write(f"Final shape: {df_final.shape}\n")
            f.write(f"Features added: {df_final.shape[1] - df_original.shape[1]}\n\n")
        
            f.write("Dropped Features:\n")
            dropped_features = self.config['features']['drop_features']
            for feature in dropped_features:
                if feature in df_original.columns:
                    f.write(f"  - {feature}\n")
            f.write("\n")
        
            f.write("Engineered Features:\n")
            new_features = [col for col in df_final.columns if col not in df_original.columns]
            for feature in new_features:
                f.write(f"  [OK] {feature}\n")  # ← Changed ✓ to [OK]
            f.write("\n")
        
            f.write("Feature Statistics:\n")
            f.write(str(df_final.describe()) + "\n\n")
        
            f.write("="*60 + "\n")
    
        logger.info(f"Feature report saved to {report_path}")
    
    def run(self, df):
        """Run complete feature engineering pipeline"""
        logger.info("="*50)
        logger.info("STARTING FEATURE ENGINEERING PIPELINE")
        logger.info("="*50)
        
        df_original = df.copy()
        
        # Drop weak features
        df = self.drop_weak_features(df)
        
        # Create engineered features
        df = self.create_ssl_domain_trust(df)
        df = self.create_url_suspicion_score(df)
        df = self.create_content_credibility(df)
        df = self.create_domain_reputation(df)
        df = self.create_security_features_count(df)
        df = self.create_suspicious_features_count(df)
        df = self.create_ssl_anchor_interaction(df)
        
        # Verify
        all_created = self.verify_features(df)
        
        if all_created:
            logger.info("✓ All features created successfully!")
        else:
            logger.warning("✗ Some features were not created")
        
        # Save
        self.save_feature_data(df)
        self.save_feature_report(df_original, df)
        
        logger.info(f"Feature engineering completed! Final shape: {df.shape}")
        return df

if __name__ == "__main__":
    # Load preprocessed data
    preprocessed_data_path = "data/processed/cleaned_data.csv"
    df = pd.read_csv(preprocessed_data_path)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_features = engineer.run(df)
    
    print(f"\nFinal feature set: {df_features.shape}")
    print(f"New features: {[col for col in df_features.columns if col not in df.columns]}")
