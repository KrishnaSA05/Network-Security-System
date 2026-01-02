"""
MLflow Helper Module
Utilities for MLflow experiment tracking and model registry
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/mlflow.log")

class MLflowTracker:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize MLflow tracker"""
        self.config = load_config(config_path)
        self.mlflow_config = self.config['mlflow']
        self.enabled = self.mlflow_config['enable']
        
        if not self.enabled:
            logger.info("MLflow tracking is disabled")
            return
        
        # Set tracking URI
        tracking_uri = self.mlflow_config['tracking_uri']
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Set experiment
        experiment_name = self.mlflow_config['experiment_name']
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment: {experiment_name}")
        
        # Enable autologging if configured
        if self.mlflow_config['autolog']['enable']:
            self._enable_autologging()
        
        self.active_run = None
    
    def _enable_autologging(self):
        """Enable MLflow autologging for supported frameworks"""
        logger.info("Enabling MLflow autologging...")
        
        try:
            mlflow.sklearn.autolog(
                log_models=self.mlflow_config['autolog']['log_models'],
                log_input_examples=self.mlflow_config['autolog']['log_input_examples'],
                log_model_signatures=self.mlflow_config['autolog']['log_model_signatures']
            )
            
            mlflow.xgboost.autolog(
                log_models=self.mlflow_config['autolog']['log_models'],
                log_input_examples=self.mlflow_config['autolog']['log_input_examples'],
                log_model_signatures=self.mlflow_config['autolog']['log_model_signatures']
            )
            
            mlflow.lightgbm.autolog(
                log_models=self.mlflow_config['autolog']['log_models'],
                log_input_examples=self.mlflow_config['autolog']['log_input_examples'],
                log_model_signatures=self.mlflow_config['autolog']['log_model_signatures']
            )
            
            logger.info("✓ Autologging enabled for sklearn, xgboost, lightgbm")
        except Exception as e:
            logger.warning(f"Autologging setup warning: {str(e)}")
    
    def start_run(self, run_name=None, tags=None):
        """Start a new MLflow run"""
        if not self.enabled:
            return None
        
        # Merge default tags with custom tags
        all_tags = self.mlflow_config.get('default_tags', {}).copy()
        if tags:
            all_tags.update(tags)
        
        self.active_run = mlflow.start_run(run_name=run_name, tags=all_tags)
        logger.info(f"Started MLflow run: {self.active_run.info.run_id}")
        
        return self.active_run
    
    def end_run(self):
        """End the current MLflow run"""
        if not self.enabled or self.active_run is None:
            return
        
        mlflow.end_run()
        logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
        self.active_run = None
    
    def log_params(self, params):
        """Log parameters to MLflow"""
        if not self.enabled or not self.mlflow_config['log_params']:
            return
        
        try:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters")
        except Exception as e:
            logger.warning(f"Error logging params: {str(e)}")
    
    def log_param(self, key, value):
        """Log a single parameter"""
        if not self.enabled or not self.mlflow_config['log_params']:
            return
        
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Error logging param {key}: {str(e)}")
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        if not self.enabled or not self.mlflow_config['log_metrics']:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged {len(metrics)} metrics")
        except Exception as e:
            logger.warning(f"Error logging metrics: {str(e)}")
    
    def log_metric(self, key, value, step=None):
        """Log a single metric"""
        if not self.enabled or not self.mlflow_config['log_metrics']:
            return
        
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Error logging metric {key}: {str(e)}")
    
    def log_model(self, model, artifact_path, model_type="sklearn", 
                  signature=None, input_example=None):
        """Log model to MLflow"""
        if not self.enabled:
            return
        
        try:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            
            logger.info(f"Logged {model_type} model to {artifact_path}")
        except Exception as e:
            logger.warning(f"Error logging model: {str(e)}")
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact (file) to MLflow"""
        if not self.enabled or not self.mlflow_config['log_artifacts']:
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Error logging artifact: {str(e)}")
    
    def log_confusion_matrix(self, cm, labels=None):
        """Log confusion matrix as artifact"""
        if not self.enabled or not self.mlflow_config['log_confusion_matrix']:
            return
        
        try:
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels or ['Phishing', 'Legitimate'],
                       yticklabels=labels or ['Phishing', 'Legitimate'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save temporarily
            temp_path = "temp_confusion_matrix.png"
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to MLflow
            mlflow.log_artifact(temp_path)
            
            # Clean up
            import os
            os.remove(temp_path)
            
            logger.info("Logged confusion matrix")
        except Exception as e:
            logger.warning(f"Error logging confusion matrix: {str(e)}")
    
    def log_feature_importance(self, model, feature_names, top_n=20):
        """Log feature importance plot"""
        if not self.enabled or not self.mlflow_config['log_feature_importance']:
            return
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create DataFrame
                import pandas as pd
                feat_imp = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(top_n)
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.barh(feat_imp['feature'], feat_imp['importance'])
                plt.xlabel('Importance')
                plt.title(f'Top {top_n} Feature Importances')
                plt.gca().invert_yaxis()
                
                # Save temporarily
                temp_path = "temp_feature_importance.png"
                plt.savefig(temp_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Log to MLflow
                mlflow.log_artifact(temp_path)
                
                # Clean up
                import os
                os.remove(temp_path)
                
                logger.info("Logged feature importance")
            else:
                logger.info("Model does not have feature_importances_ attribute")
        except Exception as e:
            logger.warning(f"Error logging feature importance: {str(e)}")
    
    def register_model(self, model_name, run_id=None):
        """Register model to MLflow Model Registry"""
        if not self.enabled or not self.mlflow_config['register_best_model']:
            return None
        
        try:
            if run_id is None and self.active_run:
                run_id = self.active_run.info.run_id
            
            model_uri = f"runs:/{run_id}/model"
            
            registered_model = mlflow.register_model(model_uri, model_name)
            
            logger.info(f"Registered model: {model_name} (version {registered_model.version})")
            return registered_model
        except Exception as e:
            logger.warning(f"Error registering model: {str(e)}")
            return None
    
    def set_tags(self, tags):
        """Set tags for the current run"""
        if not self.enabled:
            return
        
        try:
            mlflow.set_tags(tags)
            logger.info(f"Set {len(tags)} tags")
        except Exception as e:
            logger.warning(f"Error setting tags: {str(e)}")
    
    def get_run_id(self):
        """Get current run ID"""
        if self.active_run:
            return self.active_run.info.run_id
        return None

if __name__ == "__main__":
    # Test MLflow setup
    tracker = MLflowTracker()
    
    if tracker.enabled:
        tracker.start_run(run_name="test_run")
        tracker.log_params({"test_param": 1})
        tracker.log_metrics({"test_metric": 0.95})
        tracker.end_run()
        
        print("MLflow test successful!")
        print(f"Check MLflow UI: mlflow ui --backend-store-uri {tracker.mlflow_config['tracking_uri']}")
