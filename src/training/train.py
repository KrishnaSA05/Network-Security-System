"""
Model Training Module
Trains multiple ML models with optional hyperparameter tuning
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
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models with base parameters"""
        logger.info("Initializing models...")
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            **self.models_config['random_forest']
        )
        
        # XGBoost
        xgb_params = self.models_config['xgboost'].copy()
        self.models['xgboost'] = XGBClassifier(
            **xgb_params,
            eval_metric='logloss'
        )
        
        # LightGBM
        lgbm_params = self.models_config['lightgbm'].copy()
        self.models['lightgbm'] = LGBMClassifier(
            **lgbm_params,
            verbose=-1
        )
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            **self.models_config['logistic_regression']
        )
        
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
    
    def train_model(self, model_name, model, X_train, y_train):
        """Train a single model"""
        logger.info(f"Training {model_name}...")
        
        start_time = datetime.now()
        
        try:
            model.fit(X_train, y_train)
            
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            logger.info(f"✓ {model_name} trained successfully in {training_time:.2f}s")
            
            # Store training info
            self.training_history[model_name] = {
                'training_time': training_time,
                'timestamp': end_time.isoformat(),
                'params': model.get_params()
            }
            
            return model, True
            
        except Exception as e:
            logger.error(f"✗ Error training {model_name}: {str(e)}")
            return None, False
    
    def tune_hyperparameters(self, model_name, base_model, X_train, y_train):
        """Perform hyperparameter tuning"""
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        param_grid = self.tuning_config['param_grids'][model_name]
        cv_folds = self.tuning_config['cv_folds']
        scoring = self.tuning_config['scoring']
        n_jobs = self.tuning_config['n_jobs']
        
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
            else:  # randomized
                logger.info(f"Using RandomizedSearchCV with {self.tuning_config['n_iter']} iterations")
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
            
            logger.info(f"✓ Tuning completed in {tuning_time:.2f}s")
            logger.info(f"Best score: {search.best_score_:.4f}")
            logger.info(f"Best params: {search.best_params_}")
            
            # Store tuning info
            self.training_history[model_name].update({
                'tuning_time': tuning_time,
                'best_score': search.best_score_,
                'best_params': search.best_params_,
                'tuned': True
            })
            
            return search.best_estimator_, True
            
        except Exception as e:
            logger.error(f"✗ Error tuning {model_name}: {str(e)}")
            return base_model, False
    
    def train_all_models(self, X_train, y_train):
        """Train all models"""
        logger.info("="*60)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*60)
        
        tuning_enabled = self.tuning_config['enable']
        
        for model_name, model in self.models.items():
            logger.info(f"\n--- Training {model_name.upper()} ---")
            
            # Train base model
            trained_model, success = self.train_model(model_name, model, X_train, y_train)
            
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
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*60)
        
        # Load data
        df = self.load_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # Initialize models
        self.initialize_models()
        
        # Train all models
        self.train_all_models(X_train, y_train)
        
        # Save models
        self.save_all_models()
        
        # Save history
        self.save_training_history()
        
        logger.info("="*60)
        logger.info("TRAINING PIPELINE COMPLETED!")
        logger.info("="*60)
        
        return self.trained_models, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    trainer = ModelTrainer()
    trained_models, X_train, X_test, y_train, y_test = trainer.run()
    
    print(f"\n✅ Training completed!")
    print(f"Trained models: {list(trained_models.keys())}")
    print(f"Models saved to: {trainer.save_path}")
