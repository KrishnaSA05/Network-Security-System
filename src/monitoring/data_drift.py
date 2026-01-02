"""
Data Drift Detection Module
Detects distribution shifts between reference and new data
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/data_drift.log")

class DataDriftDetector:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize drift detector with configuration"""
        self.config = load_config(config_path)
        self.drift_config = self.config['data_drift']
        self.enabled = self.drift_config['enable']
        self.method = self.drift_config['method']
        self.threshold = self.drift_config['threshold']
        self.reference_path = self.drift_config['reference_data_path']
        self.target_column = self.config['data']['target_column']
        
        if self.enabled:
            logger.info(f"Drift detection enabled. Method: {self.method}, Threshold: {self.threshold}")
        else:
            logger.info("Drift detection disabled")
    
    def load_reference_data(self):
        """Load reference (baseline) data"""
        try:
            logger.info(f"Loading reference data from: {self.reference_path}")
            df_ref = pd.read_csv(self.reference_path)
            
            # Remove target column for drift detection
            if self.target_column in df_ref.columns:
                df_ref = df_ref.drop(columns=[self.target_column])
            
            logger.info(f"Reference data loaded. Shape: {df_ref.shape}")
            return df_ref
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise
    
    def ks_test(self, reference_data, current_data, feature):
        """
        Kolmogorov-Smirnov test for distribution comparison
        Returns: (statistic, p_value, is_drift)
        """
        ref_feature = reference_data[feature].dropna()
        cur_feature = current_data[feature].dropna()
        
        statistic, p_value = stats.ks_2samp(ref_feature, cur_feature)
        is_drift = p_value < self.threshold
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'is_drift': bool(is_drift),
            'method': 'ks_test'
        }
    
    def psi(self, reference_data, current_data, feature, bins=10):
        """
        Population Stability Index (PSI)
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change
        PSI > 0.25: Significant change
        """
        ref_feature = reference_data[feature].dropna()
        cur_feature = current_data[feature].dropna()
        
        # Create bins based on reference data
        breakpoints = np.percentile(ref_feature, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        
        # Calculate distribution for each dataset
        ref_dist = np.histogram(ref_feature, bins=breakpoints)[0] / len(ref_feature)
        cur_dist = np.histogram(cur_feature, bins=breakpoints)[0] / len(cur_feature)
        
        # Avoid division by zero
        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        cur_dist = np.where(cur_dist == 0, 0.0001, cur_dist)
        
        # Calculate PSI
        psi_value = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        
        is_drift = psi_value > 0.1  # Moderate or significant change
        
        return {
            'psi_value': float(psi_value),
            'is_drift': bool(is_drift),
            'severity': 'low' if psi_value < 0.1 else 'medium' if psi_value < 0.25 else 'high',
            'method': 'psi'
        }
    
    def wasserstein_distance(self, reference_data, current_data, feature):
        """
        Wasserstein distance (Earth Mover's Distance)
        Measures the distance between two distributions
        """
        ref_feature = reference_data[feature].dropna()
        cur_feature = current_data[feature].dropna()
        
        distance = stats.wasserstein_distance(ref_feature, cur_feature)
        
        # Normalize by feature range
        feature_range = ref_feature.max() - ref_feature.min()
        if feature_range > 0:
            normalized_distance = distance / feature_range
        else:
            normalized_distance = 0
        
        is_drift = normalized_distance > self.threshold
        
        return {
            'distance': float(distance),
            'normalized_distance': float(normalized_distance),
            'is_drift': bool(is_drift),
            'method': 'wasserstein'
        }
    
    def detect_drift_for_feature(self, reference_data, current_data, feature):
        """Detect drift for a single feature"""
        if self.method == "ks_test":
            return self.ks_test(reference_data, current_data, feature)
        elif self.method == "psi":
            return self.psi(reference_data, current_data, feature)
        elif self.method == "wasserstein":
            return self.wasserstein_distance(reference_data, current_data, feature)
        else:
            raise ValueError(f"Unsupported drift detection method: {self.method}")
    
    def detect_drift(self, current_data):
        """Detect drift across all features"""
        if not self.enabled:
            logger.info("Drift detection is disabled")
            return None
        
        logger.info("="*50)
        logger.info("STARTING DATA DRIFT DETECTION")
        logger.info("="*50)
        
        # Load reference data
        reference_data = self.load_reference_data()
        
        # Remove target column from current data
        current_features = current_data.drop(columns=[self.target_column], errors='ignore')
        
        # Get features to monitor
        features_to_monitor = self.drift_config.get('features_to_monitor', [])
        if not features_to_monitor:
            features_to_monitor = current_features.columns.tolist()
        
        logger.info(f"Monitoring {len(features_to_monitor)} features for drift")
        
        # Detect drift for each feature
        drift_results = {}
        drifted_features = []
        
        for feature in features_to_monitor:
            if feature not in reference_data.columns or feature not in current_features.columns:
                logger.warning(f"Feature '{feature}' not found in both datasets. Skipping.")
                continue
            
            try:
                result = self.detect_drift_for_feature(reference_data, current_features, feature)
                drift_results[feature] = result
                
                if result['is_drift']:
                    drifted_features.append(feature)
                    logger.warning(f"⚠ DRIFT DETECTED in '{feature}': {result}")
                else:
                    logger.info(f"✓ No drift in '{feature}'")
                    
            except Exception as e:
                logger.error(f"Error detecting drift for '{feature}': {str(e)}")
        
        # Summary
        drift_summary = {
            'timestamp': datetime.now().isoformat(),
            'method': self.method,
            'threshold': self.threshold,
            'total_features': len(features_to_monitor),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'drift_detected': len(drifted_features) > 0,
            'feature_results': drift_results
        }
        
        logger.info("="*50)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total features monitored: {len(features_to_monitor)}")
        logger.info(f"Features with drift: {len(drifted_features)}")
        
        if drifted_features:
            logger.warning(f"⚠ Drifted features: {drifted_features}")
        else:
            logger.info("✓ No drift detected")
        
        # Save report
        if self.drift_config['save_report']:
            self.save_drift_report(drift_summary)
        
        # Alert if drift detected
        if drift_summary['drift_detected'] and self.drift_config['alert_on_drift']:
            logger.warning("⚠⚠⚠ DRIFT ALERT: Data distribution has changed! ⚠⚠⚠")
        
        return drift_summary
    
    def save_drift_report(self, drift_summary):
        """Save drift detection report"""
        report_path = self.drift_config['report_path']
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(drift_summary, f, indent=2)
        
        logger.info(f"Drift report saved to: {report_path}")
        
        # Also save human-readable version
        txt_report_path = report_path.replace('.json', '.txt')
        with open(txt_report_path, 'w') as f:
            f.write("="*50 + "\n")
            f.write("DATA DRIFT DETECTION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {drift_summary['timestamp']}\n")
            f.write(f"Method: {drift_summary['method']}\n")
            f.write(f"Threshold: {drift_summary['threshold']}\n\n")
            f.write(f"Total features monitored: {drift_summary['total_features']}\n")
            f.write(f"Features with drift: {drift_summary['drifted_features_count']}\n\n")
            
            if drift_summary['drifted_features']:
                f.write("Drifted Features:\n")
                for feature in drift_summary['drifted_features']:
                    result = drift_summary['feature_results'][feature]
                    f.write(f"  - {feature}: {result}\n")
            else:
                f.write("✓ No drift detected\n")
            
            f.write("\n" + "="*50 + "\n")
        
        logger.info(f"Human-readable report saved to: {txt_report_path}")

if __name__ == "__main__":
    # Load current data
    current_data_path = "data/features/engineered_features.csv"
    current_data = pd.read_csv(current_data_path)
    
    # Detect drift
    detector = DataDriftDetector()
    drift_results = detector.detect_drift(current_data)
    
    if drift_results:
        print(f"\n{'⚠ DRIFT DETECTED' if drift_results['drift_detected'] else '✓ NO DRIFT'}")
        print(f"Drifted features: {drift_results['drifted_features_count']}/{drift_results['total_features']}")
