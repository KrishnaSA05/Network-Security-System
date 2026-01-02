"""
Standalone Hyperparameter Tuning Module
Can be used independently for tuning specific models
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/hyperparameter_tuning.log")

class HyperparameterTuner:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize hyperparameter tuner"""
        self.config = load_config(config_path)
        self.tuning_config = self.config['hyperparameter_tuning']
        self.random_state = self.config['data']['random_state']
        
    def tune_model(self, model_name, base_model, X_train, y_train):
        """Tune hyperparameters for a specific model"""
        logger.info("="*60)
        logger.info(f"TUNING {model_name.upper()}")
        logger.info("="*60)
        
        param_grid = self.tuning_config['param_grids'][model_name]
        method = self.tuning_config['method']
        cv_folds = self.tuning_config['cv_folds']
        scoring = self.tuning_config['scoring']
        n_jobs = self.tuning_config['n_jobs']
        
        logger.info(f"Method: {method}")
        logger.info(f"CV Folds: {cv_folds}")
        logger.info(f"Scoring: {scoring}")
        logger.info(f"Parameter grid size: {len(param_grid)}")
        
        if method == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=self.tuning_config['verbose'],
                return_train_score=True
            )
        else:  # randomized
            n_iter = self.tuning_config['n_iter']
            logger.info(f"Random iterations: {n_iter}")
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=n_jobs,
                random_state=self.random_state,
                verbose=self.tuning_config['verbose'],
                return_train_score=True
            )
        
        logger.info("Starting hyperparameter search...")
        search.fit(X_train, y_train)
        
        # Results
        logger.info("\n" + "="*60)
        logger.info("TUNING RESULTS")
        logger.info("="*60)
        logger.info(f"Best CV Score: {search.best_score_:.4f}")
        logger.info(f"Best Parameters:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        # Show top 5 parameter combinations
        results_df = pd.DataFrame(search.cv_results_)
        top_5 = results_df.nsmallest(5, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]
        
        logger.info("\nTop 5 Parameter Combinations:")
        logger.info("\n" + top_5.to_string(index=False))
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def compare_tuned_vs_default(self, model_name, default_model, tuned_model, X_train, y_train):
        """Compare tuned model vs default model"""
        logger.info(f"\nComparing tuned vs default {model_name}...")
        
        cv_folds = self.tuning_config['cv_folds']
        scoring = self.tuning_config['scoring']
        
        # Cross-validation scores
        default_scores = cross_val_score(default_model, X_train, y_train, 
                                        cv=cv_folds, scoring=scoring, n_jobs=-1)
        tuned_scores = cross_val_score(tuned_model, X_train, y_train, 
                                      cv=cv_folds, scoring=scoring, n_jobs=-1)
        
        logger.info(f"Default Model - Mean {scoring}: {default_scores.mean():.4f} (+/- {default_scores.std():.4f})")
        logger.info(f"Tuned Model - Mean {scoring}: {tuned_scores.mean():.4f} (+/- {tuned_scores.std():.4f})")
        logger.info(f"Improvement: {(tuned_scores.mean() - default_scores.mean()):.4f}")
        
        return {
            'default_mean': default_scores.mean(),
            'default_std': default_scores.std(),
            'tuned_mean': tuned_scores.mean(),
            'tuned_std': tuned_scores.std(),
            'improvement': tuned_scores.mean() - default_scores.mean()
        }

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    
    # Load data
    df = pd.read_csv("data/features/engineered_features.csv")
    X = df.drop(columns=['Result'])
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize tuner
    tuner = HyperparameterTuner()
    
    # Load config
    config = load_config()
    base_params = config['models']['random_forest']
    
    # Create base model
    base_model = RandomForestClassifier(**base_params)
    
    # Tune
    best_model, best_params, best_score = tuner.tune_model('random_forest', base_model, X_train, y_train)
    
    # Compare
    comparison = tuner.compare_tuned_vs_default('random_forest', base_model, best_model, X_train, y_train)
    
    print("\n✅ Hyperparameter tuning completed!")
    print(f"Best score: {best_score:.4f}")
    print(f"Improvement: {comparison['improvement']:.4f}")
