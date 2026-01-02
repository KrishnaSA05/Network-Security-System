"""
Model Training Module with MLflow Integration
Trains multiple ML models with experiment tracking
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config
from utils.mlflow_helper import MLflowTracker

logger = setup_logger(__name__, "logs/training.log")

class ModelTrainer:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize model trainer with configuration"""
        self.config = load_config(config_path)
        self.target_column = self.config['data']['target_column']
        self.test_size = self.config['data']['train_test_split']
        self.random_state = self.config['data']['random_state']
        self.models_config = self.config['models']
        self.tuning_config = self.config['hyperparameter_tuning']
        self.save_path = self.config['training']['save_model_path']
        
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.trained_models = {}
        self.training_history = {}
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker(config_path)
        
    def load_data(self, data_path=None):
        """Load feature-engineered data"""
        if data_path is None:
            data_path = self.config['data']['feature_data_path']
        
        logger.info(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded. Shape: {df.shape}")
        
        return df
    
    def split_data(self, df):
        """Split data into train and test sets"""
        logger.info("Splitting data into train/test sets...")
        
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        logger.info(f"Test class distribution:\n{y_test.value_counts()}")
        
        # Log dataset info to MLflow
        self.mlflow_tracker.log_params({
            'dataset_total_samples': len(df),
            'dataset_features': X_train.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'test_size_ratio': self.test_size
        })
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models with base parameters"""
        logger.info("Initializing models...")
        
        self.models['random_forest'] = RandomForestClassifier(
            **self.models_config['random_forest']
        )
        
        xgb_params = self.models_config['xgboost'].copy()
        self.models['xgboost'] = XGBClassifier(
            **xgb_params,
            eval_metric='logloss'
        )
        
        lgbm_params = self.models_config['lightgbm'].copy()
        self.models['lightgbm'] = LGBMClassifier(
            **lgbm_params,
            verbose=-1
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            **self.models_config['logistic_regression']
        )
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_model(self, model_name, model, X_train, y_train, X_test, y_test):
        """Train a single model with MLflow tracking"""
        logger.info(f"Training {model_name}...")
        
        # Start MLflow run
        run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_tracker.start_run(
            run_name=run_name,
            tags={'model_type': model_name}
        )
        
        start_time = datetime.now()
        
        try:
            # Log model parameters
            params = model.get_params()
            self.mlflow_tracker.log_params(params)
            
            # Train model
            model.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Log training time
            self.mlflow_tracker.log_metric('training_time_seconds', training_time)
            
            # Quick evaluation on train and test
            from sklearn.metrics import accuracy_score
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            self.mlflow_tracker.log_metrics({
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            })
            
            logger.info(f"✓ {model_name} trained successfully in {training_time:.2f}s")
            logger.info(f"  Train Accuracy: {train_acc:.4f}")
            logger.info(f"  Test Accuracy: {test_acc:.4f}")
            
            # Log feature importance
            if hasattr(model, 'feature_importances_'):
                self.mlflow_tracker.log_feature_importance(
                    model, 
                    X_train.columns.tolist()
                )
            
            # Store training info
            self.training_history[model_name] = {
                'training_time': training_time,
                'timestamp': end_time.isoformat(),
                'params': params,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'run_id': self.mlflow_tracker.get_run_id()
            }
            
            # End MLflow run
            self.mlflow_tracker.end_run()
            
            return model, True
            
        except Exception as e:
            logger.error(f"✗ Error training {model_name}: {str(e)}")
            self.mlflow_tracker.end_run()
            return None, False
    
    def tune_hyperparameters(self, model_name, base_model, X_train, y_train):
        """Perform hyperparameter tuning with MLflow tracking"""
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        # Start nested run for tuning
        run_name = f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_tracker.start_run(
            run_name=run_name,
            tags={'model_type': model_name, 'phase': 'hyperparameter_tuning'}
        )
        
        param_grid = self.tuning_config['param_grids'][model_name]
        cv_folds = self.tuning_config['cv_folds']
        scoring = self.tuning_config['scoring']
        n_jobs = self.tuning_config['n_jobs']
        
        # Log tuning configuration
        self.mlflow_tracker.log_params({
            'tuning_method': self.tuning_config['method'],
            'cv_folds': cv_folds,
            'scoring': scoring
        })
        
        start_time = datetime.now()
        
        try:
            if self.tuning_config['method'] == 'grid':
                logger.info(f"Using GridSearchCV with {cv_folds}-fold CV")
                search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=self.tuning_config['verbose']
                )
            else:
                logger.info(f"Using RandomizedSearchCV with {self.tuning_config['n_iter']} iterations")
                self.mlflow_tracker.log_param('n_iter', self.tuning_config['n_iter'])
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=self.tuning_config['n_iter'],
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                    verbose=self.tuning_config['verbose']
                )
            
            search.fit(X_train, y_train)
            
            end_time = datetime.now()
            tuning_time = (end_time - start_time).total_seconds()
            
            # Log best parameters and score
            self.mlflow_tracker.log_params({
                f'best_{k}': v for k, v in search.best_params_.items()
            })
            self.mlflow_tracker.log_metrics({
                'best_cv_score': search.best_score_,
                'tuning_time_seconds': tuning_time
            })
            
            logger.info(f"✓ Tuning completed in {tuning_time:.2f}s")
            logger.info(f"Best score: {search.best_score_:.4f}")
            logger.info(f"Best params: {search.best_params_}")
            
            # Store tuning info
            self.training_history[model_name].update({
                'tuning_time': tuning_time,
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'tuned': True,
                'tuning_run_id': self.mlflow_tracker.get_run_id()
            })
            
            # End tuning run
            self.mlflow_tracker.end_run()
            
            return search.best_estimator_, True
            
        except Exception as e:
            logger.error(f"✗ Error tuning {model_name}: {str(e)}")
            self.mlflow_tracker.end_run()
            return base_model, False
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models with MLflow tracking"""
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        tuning_enabled = self.tuning_config['enable']
        
        for model_name, model in self.models.items():
            logger.info(f"\n--- Training {model_name.upper()} ---")
            
            # Train base model
            trained_model, success = self.train_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            
            if not success:
                continue
            
            # Hyperparameter tuning (if enabled)
            if tuning_enabled:
                tuned_model, tuning_success = self.tune_hyperparameters(
                    model_name, trained_model, X_train, y_train
                )
                if tuning_success:
                    trained_model = tuned_model
            else:
                self.training_history[model_name]['tuned'] = False
            
            # Store trained model
            self.trained_models[model_name] = trained_model
        
        logger.info(f"\n✓ Successfully trained {len(self.trained_models)} models")
    
    def save_model(self, model_name, model):
        """Save a trained model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
        filepath = Path(self.save_path) / filename
        
        joblib.dump(model, filepath)
        logger.info(f"Model saved: {filepath}")
        
        # Also save latest version
        latest_path = Path(self.save_path) / f"{model_name}_latest.pkl"
        joblib.dump(model, latest_path)
        
        return str(filepath)
    
    def save_all_models(self):
        """Save all trained models"""
        logger.info("Saving all trained models...")
        
        saved_paths = {}
        for model_name, model in self.trained_models.items():
            filepath = self.save_model(model_name, model)
            saved_paths[model_name] = filepath
        
        logger.info(f"✓ Saved {len(saved_paths)} models")
        return saved_paths
    
    def save_training_history(self):
        """Save training history"""
        history_path = Path(self.save_path) / "training_history.json"
        
        import json
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved: {history_path}")
    
    def run(self, data_path=None):
        """Run complete training pipeline"""
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING PIPELINE WITH MLFLOW")
        logger.info("="*60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Initialize models
        self.initialize_models()
        
        # Train all models
        self.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_all_models()
        
        # Save history
        self.save_training_history()
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED!")
        logger.info(f"✓ View experiments: mlflow ui --backend-store-uri {self.mlflow_tracker.mlflow_config['tracking_uri']}")
        logger.info("="*60)
        
        return self.trained_models, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    trainer = ModelTrainer()
    trained_models, X_train, X_test, y_train, y_test = trainer.run()
    
    print(f"\n Training completed!")
    print(f"Trained models: {list(trained_models.keys())}")
    print(f"Models saved to: {trainer.save_path}")
    print(f"\n🔍 View MLflow UI:")
    print(f"   mlflow ui --backend-store-uri mlruns")
    print(f"   Then open: http://localhost:5000")
