"""
Model Registry Module
Manages model versioning and metadata
"""
import json
import joblib
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, load_config

logger = setup_logger(__name__, "logs/model_registry.log")

class ModelRegistry:
    def __init__(self, config_path="src/config/config.yaml"):
        """Initialize model registry"""
        self.config = load_config(config_path)
        self.registry_path = Path(self.config['training']['save_model_path']) / "registry.json"
        self.models_dir = Path(self.config['training']['save_model_path'])
        
        # Load or create registry
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {}
            self._save_registry()
    
    def register_model(self, model_name, model, metrics, params=None, tags=None):
        """Register a new model version"""
        logger.info(f"Registering model: {model_name}")
        
        # Generate version
        if model_name not in self.registry:
            self.registry[model_name] = {'versions': []}
            version = 1
        else:
            version = len(self.registry[model_name]['versions']) + 1
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_v{version}_{timestamp}.pkl"
        filepath = self.models_dir / filename
        
        joblib.dump(model, filepath)
        
        # Register metadata
        model_info = {
            'version': version,
            'filename': filename,
            'filepath': str(filepath),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'params': params or {},
            'tags': tags or []
        }
        
        self.registry[model_name]['versions'].append(model_info)
        
        # Update best model
        if 'best_version' not in self.registry[model_name]:
            self.registry[model_name]['best_version'] = version
        else:
            current_best = self.get_best_model_info(model_name)
            if metrics.get('accuracy', 0) > current_best['metrics'].get('accuracy', 0):
                self.registry[model_name]['best_version'] = version
                logger.info(f"✓ New best version for {model_name}: v{version}")
        
        self._save_registry()
        
        logger.info(f"✓ Registered {model_name} v{version}")
        return version
    
    def get_model(self, model_name, version=None):
        """Load a specific model version"""
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        if version is None:
            # Get best version
            version = self.registry[model_name]['best_version']
            logger.info(f"Loading best version of {model_name}: v{version}")
        
        # Find model info
        model_info = None
        for v in self.registry[model_name]['versions']:
            if v['version'] == version:
                model_info = v
                break
        
        if model_info is None:
            raise ValueError(f"Version {version} not found for model '{model_name}'")
        
        # Load model
        filepath = Path(model_info['filepath'])
        model = joblib.load(filepath)
        
        logger.info(f"✓ Loaded {model_name} v{version}")
        return model, model_info
    
    def get_best_model_info(self, model_name):
        """Get metadata of the best model version"""
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        best_version = self.registry[model_name]['best_version']
        
        for v in self.registry[model_name]['versions']:
            if v['version'] == best_version:
                return v
        
        return None
    
    def list_models(self):
        """List all registered models"""
        logger.info("Registered Models:")
        
        for model_name, data in self.registry.items():
            versions = len(data['versions'])
            best_version = data['best_version']
            logger.info(f"  {model_name}: {versions} version(s), best: v{best_version}")
        
        return list(self.registry.keys())
    
    def get_model_history(self, model_name):
        """Get version history of a model"""
        if model_name not in self.registry:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        versions = self.registry[model_name]['versions']
        
        logger.info(f"\nVersion History for {model_name}:")
        for v in versions:
            logger.info(f"  v{v['version']} - {v['timestamp']} - Accuracy: {v['metrics'].get('accuracy', 'N/A'):.4f}")
        
        return versions
    
    def _save_registry(self):
        """Save registry to file"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

if __name__ == "__main__":
    registry = ModelRegistry()
    registry.list_models()
