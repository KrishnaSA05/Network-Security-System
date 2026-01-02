"""
Model Evaluation Module
Evaluates trained models and generates comprehensive reports
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config
from utils.mlflow_helper import MLflowTracker

logger = setup_logger(__name__, "logs/evaluation.log")

class ModelEvaluator:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize model evaluator with configuration"""
        self.config = load_config(config_path)
        self.eval_config = self.config['evaluation']
        self.plots_dir = self.eval_config['plots_dir']
        self.mlflow_tracker = MLflowTracker(config_path)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Start MLflow run for evaluation
        run_name = f"{model_name}_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_tracker.start_run(
            run_name=run_name,
            tags={'model_type': model_name, 'phase': 'evaluation'}
        )
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1_score': f1_score(y_test, y_pred, average='binary'),
            }
            
            # ROC AUC (if probabilities available)
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics to MLflow
            self.mlflow_tracker.log_metrics(metrics)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            # Log confusion matrix to MLflow
            self.mlflow_tracker.log_confusion_matrix(cm)
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Store results
            self.results[model_name] = {
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Log results
            logger.info(f"✓ {model_name} Results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
            self.mlflow_tracker.end_run()

            return metrics, cm, class_report
            
        except Exception as e:
            logger.error(f"✗ Error evaluating {model_name}: {str(e)}")
            self.mlflow_tracker.end_run()
            return None, None, None
    
    def evaluate_all_models(self, trained_models, X_test, y_test):
        """Evaluate all trained models"""
        logger.info("="*60)
        logger.info("EVALUATING ALL MODELS")
        logger.info("="*60)
        
        for model_name, model in trained_models.items():
            logger.info(f"\n--- Evaluating {model_name.upper()} ---")
            self.evaluate_model(model_name, model, X_test, y_test)
        
        logger.info(f"\n✓ Evaluated {len(self.results)} models")
    
    def plot_confusion_matrix(self, model_name, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Phishing', 'Legitimate'],
                    yticklabels=['Phishing', 'Legitimate'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        filepath = Path(self.plots_dir) / f"{model_name}_confusion_matrix.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved: {filepath}")
    
    def plot_metrics_comparison(self):
        """Plot metrics comparison across all models"""
        metrics_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_df.plot(kind='bar', ax=ax)
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        filepath = Path(self.plots_dir) / "metrics_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics comparison saved: {filepath}")
        
        return metrics_df
    
    def generate_comparison_table(self):
        """Generate comparison table of all models"""
        comparison = []
        
        for model_name, result in self.results.items():
            row = {'Model': model_name}
            row.update(result['metrics'])
            comparison.append(row)
        
        df = pd.DataFrame(comparison).sort_values('accuracy', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("="*60)
        logger.info("\n" + df.to_string(index=False))
        
        # Save to CSV
        csv_path = Path(self.plots_dir) / "model_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"\nComparison table saved: {csv_path}")
        
        return df
    
    def get_best_model(self, metric='accuracy'):
        """Get the best performing model"""
        best_model_name = max(
            self.results.keys(),
            key=lambda k: self.results[k]['metrics'].get(metric, 0)
        )
        best_score = self.results[best_model_name]['metrics'][metric]
        
        logger.info(f"\n🏆 Best model by {metric}: {best_model_name} ({best_score:.4f})")
        
        return best_model_name, best_score
    
    def save_evaluation_report(self):
        """Save comprehensive evaluation report"""
        report_path = Path(self.plots_dir) / "evaluation_report.json"
        
        # Prepare report data (convert numpy types to native Python)
        report = {}
        for model_name, result in self.results.items():
            report[model_name] = {
                'metrics': result['metrics'],
                'confusion_matrix': result['confusion_matrix'],
                'classification_report': result['classification_report']
            }
        
        # Add summary
        best_model, best_score = self.get_best_model('accuracy')
        report['summary'] = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.results),
            'best_model': best_model,
            'best_accuracy': best_score
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved: {report_path}")
        
        # Human-readable version
        txt_path = Path(self.plots_dir) / "evaluation_report.txt"
        with open(txt_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {report['summary']['timestamp']}\n")
            f.write(f"Total Models: {report['summary']['total_models']}\n")
            f.write(f"Best Model: {report['summary']['best_model']}\n")
            f.write(f"Best Accuracy: {report['summary']['best_accuracy']:.4f}\n\n")
            
            for model_name, data in report.items():
                if model_name == 'summary':
                    continue
                
                f.write("-"*60 + "\n")
                f.write(f"{model_name.upper()}\n")
                f.write("-"*60 + "\n")
                
                f.write("Metrics:\n")
                for metric, value in data['metrics'].items():
                    f.write(f"  {metric}: {value:.4f}\n")
                
                f.write("\nConfusion Matrix:\n")
                cm = np.array(data['confusion_matrix'])
                f.write(f"  [[{cm[0][0]}, {cm[0][1]}],\n")
                f.write(f"   [{cm[1][0]}, {cm[1][1]}]]\n\n")
        
        logger.info(f"Human-readable report saved: {txt_path}")
    
    def plot_all_visualizations(self):
        """Generate all visualization plots"""
        if not self.eval_config.get('save_plots', True):
            logger.info("Plot saving is disabled")
            return
        
        logger.info("Generating visualization plots...")
        
        # Confusion matrices for each model
        for model_name, result in self.results.items():
            cm = np.array(result['confusion_matrix'])
            self.plot_confusion_matrix(model_name, cm)
        
        # Metrics comparison
        self.plot_metrics_comparison()
        
        logger.info("✓ All plots generated")
    
    def run(self, trained_models, X_test, y_test):
        """Run complete evaluation pipeline"""
        logger.info("="*60)
        logger.info("STARTING MODEL EVALUATION PIPELINE")
        logger.info("="*60)
        
        # Evaluate all models
        self.evaluate_all_models(trained_models, X_test, y_test)
        
        # Generate comparison table
        comparison_df = self.generate_comparison_table()
        
        # Get best model
        best_model, best_score = self.get_best_model('accuracy')
        
        # Generate plots
        self.plot_all_visualizations()
        
        # Save reports
        self.save_evaluation_report()
        
        logger.info("="*60)
        logger.info("EVALUATION PIPELINE COMPLETED!")
        logger.info("="*60)
        
        return self.results, comparison_df

if __name__ == "__main__":
    # This should be run after training
    import joblib
    from pathlib import Path
    
    # Load test data
    df = pd.read_csv("data/features/engineered_features.csv")
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=['Result'])
    y = df['Result']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Load trained models
    model_dir = Path("src/models/saved_models")
    trained_models = {}
    
    for model_file in model_dir.glob("*_latest.pkl"):
        model_name = model_file.stem.replace("_latest", "")
        trained_models[model_name] = joblib.load(model_file)
    
    # Evaluate
    evaluator = ModelEvaluator()
    results, comparison = evaluator.run(trained_models, X_test, y_test)
    
    print("\n✅ Evaluation completed!")
    print(f"\nBest model: {evaluator.get_best_model()[0]}")
