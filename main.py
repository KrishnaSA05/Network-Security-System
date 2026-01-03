"""
Network Security System - Main Pipeline Orchestrator
Runs the complete ML pipeline from data ingestion to model evaluation
"""
import sys
from pathlib import Path
from datetime import datetime

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Now import from src modules
from utils.logger import setup_logger, load_config
from data_pipeline.ingest import DataIngestor
from data_pipeline.preprocess import DataPreprocessor
from data_pipeline.validate import DataValidator
from features.build_features import FeatureEngineer
from monitoring.data_drift import DataDriftDetector
from training.train import ModelTrainer
from training.evaluate import ModelEvaluator

# Initialize logger
logger = setup_logger(__name__, "logs/pipeline.log")


def run_complete_pipeline():
    """Execute the complete ML pipeline"""
    try:
        logger.info("="*60)
        logger.info("STARTING COMPLETE ML PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # STEP 1: Data Ingestion
        logger.info("\n[STEP 1/7] Data Ingestion")
        logger.info("-" * 60)
        ingestor = DataIngestor()
        df = ingestor.run()
        logger.info(f"✓ Data ingestion completed. Shape: {df.shape}")
        
        # STEP 2: Data Preprocessing
        logger.info("\n[STEP 2/7] Data Preprocessing")
        logger.info("-" * 60)
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.run(df)
        logger.info(f"✓ Preprocessing completed. Shape: {df_clean.shape}")
        
        # STEP 3: Feature Engineering
        logger.info("\n[STEP 3/7] Feature Engineering")
        logger.info("-" * 60)
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.run(df_clean)
        logger.info(f"✓ Feature engineering completed. Shape: {df_features.shape}")
        
        # STEP 4: Data Drift Detection (optional)
        config = load_config("src/config/config.yaml")
        if config['data_drift']['enable']:
            logger.info("\n[STEP 4/7] Data Drift Detection")
            logger.info("-" * 60)
            try:
                drift_detector = DataDriftDetector()
                
                # Load reference and current data
                import pandas as pd
                reference_df = pd.read_csv(config['data_drift']['reference_data_path'])
                current_df = pd.read_csv(config['data']['feature_data_path'])
                
                # Detect drift
                drift_report = drift_detector.detect_drift(reference_df, current_df)
                
                # Save report
                drift_detector.save_report(drift_report)
                
                logger.info("✓ Drift detection completed")
                
                if drift_report['drift_detected']:
                    logger.warning(f"⚠ Drift detected in {len(drift_report['drifted_features'])} features")
                else:
                    logger.info("✓ No significant drift detected")
                    
            except FileNotFoundError as e:
                logger.warning(f"Drift detection skipped: Reference data not found")
                logger.info("Hint: Reference data will be created after first successful run")
            except Exception as e:
                logger.error(f"Drift detection failed: {e}")
                logger.info("Continuing pipeline...")
        else:
            logger.info("\n[STEP 4/7] Data Drift Detection - SKIPPED (disabled in config)")
        
        # STEP 5: Data Validation
        logger.info("\n[STEP 5/7] Data Validation")
        logger.info("-" * 60)
        validator = DataValidator()
        validator.run(df_features)
        logger.info("✓ Data validation completed")
        
        # STEP 6: Model Training
        logger.info("\n[STEP 6/7] Model Training")
        logger.info("-" * 60)
        trainer = ModelTrainer()
        trained_models, X_train, X_test, y_train, y_test = trainer.run()
        logger.info(f"✓ Training completed. {len(trained_models)} models trained")
        
        # STEP 7: Model Evaluation
        logger.info("\n[STEP 7/7] Model Evaluation")
        logger.info("-" * 60)
        evaluator = ModelEvaluator()
        evaluator.run(trained_models, X_test, y_test)
        logger.info("✓ Evaluation completed")
        
        # Pipeline Summary
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"\nOutputs:")
        logger.info(f"  - Processed data: {config['data']['processed_data_path']}")
        logger.info(f"  - Feature data: {config['data']['feature_data_path']}")
        logger.info(f"  - Models: {config['training']['save_model_path']}")
        logger.info(f"  - Validation reports: data/validation/")
        logger.info(f"  - Logs: logs/")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. View MLflow UI: mlflow ui")
        logger.info(f"  2. Start API: uvicorn src.serving.app:app --reload")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print("✅ PIPELINE EXECUTION SUCCESSFUL!")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print(f"Models trained: {len(trained_models)}")
        print(f"\n📊 View results:")
        print(f"   - MLflow UI: mlflow ui")
        print(f"   - Logs: logs/pipeline.log")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.exception("Full traceback:")
        
        print("\n" + "="*60)
        print("❌ PIPELINE FAILED!")
        print("="*60)
        print(f"Error: {str(e)}")
        print(f"Check logs: logs/pipeline.log")
        print("="*60)
        
        return False


if __name__ == "__main__":
    print("="*60)
    print("NETWORK SECURITY SYSTEM - ML PIPELINE")
    print("="*60)
    print("Starting pipeline execution...")
    print("")
    
    success = run_complete_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        exit(0)
    else:
        print("\n⚠️  Pipeline failed. Check logs for details.")
        exit(1)
