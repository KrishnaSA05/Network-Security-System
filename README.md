# 🛡️ Network Security System - Phishing Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Machine Learning system for detecting phishing websites using ensemble methods, MLOps best practices, and automated deployment pipelines.

## 🎯 Features

- **Multiple ML Models**: Random Forest, XGBoost, LightGBM, Logistic Regression
- **MLflow Integration**: Experiment tracking, model versioning, and registry
- **Data Drift Detection**: Monitor distribution shifts with multiple methods (KS Test, PSI, Wasserstein)
- **MongoDB Support**: Flexible data ingestion from CSV or MongoDB
- **REST API**: FastAPI-based prediction service with batch support
- **Docker**: Containerized training and serving environments
- **CI/CD**: Automated testing and deployment with GitHub Actions
- **Comprehensive Logging**: Track every step of the pipeline

## 📊 Project Structure

```
Network-Security-System/
├── data/
│   ├── raw/                    # Raw data (train.csv)
│   ├── processed/              # Preprocessed data
│   ├── features/               # Engineered features
│   └── validation/             # Validation reports
│
├── src/
│   ├── config/
│   │   └── config.yaml         # Configuration file
│   │
│   ├── data_pipeline/
│   │   ├── ingest.py          # Data loading (CSV/MongoDB)
│   │   ├── preprocess.py      # Data preprocessing
│   │   └── validate.py        # Data validation
│   │
│   ├── features/
│   │   └── build_features.py  # Feature engineering
│   │
│   ├── training/
│   │   ├── train.py           # Model training
│   │   ├── evaluate.py        # Model evaluation
│   │   └── tune.py            # Hyperparameter tuning
│   │
│   ├── models/
│   │   ├── model_registry.py  # Model versioning
│   │   └── saved_models/      # Trained models
│   │
│   ├── serving/
│   │   ├── app.py            # FastAPI application
│   │   └── schema.py         # API schemas
│   │
│   ├── monitoring/
│   │   └── data_drift.py     # Data drift detection
│   │
│   └── utils/
│       ├── logger.py          # Logging utility
│       └── mlflow_helper.py   # MLflow integration
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_model.py
│
├── docker/
│   ├── Dockerfile.train       # Training container
│   └── Dockerfile.serve       # API container
│
├── .github/workflows/
│   └── ci-cd.yaml            # CI/CD pipeline
│
├── notebooks/
│   └── exploration.ipynb     # Exploratory analysis
├── main.py                   # Main pipeline orchestrator
├── docker-compose.yml        # Multi-container setup
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md
```

## 🚀 Quick Start

### Option 1: Local Setup

```bash
# Clone repository
git clone https://github.com/KrishnaSA05/network-security-system.git
cd network-security-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Place your dataset
# Copy phisingData.csv to data/raw/

# Run complete pipeline
python main.py

# Start API server
uvicorn src.serving.app:app --reload

# View MLflow UI
mlflow ui
```

### Option 2: Docker Setup

```bash
# Build and run all services
docker-compose up -d

# Check service status
docker-compose ps

# View API logs
docker-compose logs -f api

# Run training
docker-compose run training python main.py

# Stop services
docker-compose down
```

## 📖 Usage

### Training Models

```bash
# Train with default configuration
python main.py

# With hyperparameter tuning enabled
# Edit config.yaml: set hyperparameter_tuning.enable: True
python main.py

# Train specific models only
python src/training/train.py
```

### Making Predictions

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "features": {
        "having_IP_Address": -1,
        "URL_Length": 1,
        "Shortining_Service": 1,
        "having_At_Symbol": 1,
        "double_slash_redirecting": -1,
        "Prefix_Suffix": -1,
        "having_Sub_Domain": -1,
        "SSLfinal_State": -1,
        "Domain_registeration_length": -1,
        "Favicon": 1,
        "port": 1,
        "HTTPS_token": -1,
        "Request_URL": 1,
        "URL_of_Anchor": 0,
        "Links_in_tags": 1,
        "SFH": -1,
        "Submitting_to_email": -1,
        "Abnormal_URL": -1,
        "Redirect": 0,
        "on_mouseover": 1,
        "RightClick": 1,
        "popUpWidnow": 1,
        "Iframe": 1,
        "age_of_domain": -1,
        "DNSRecord": -1,
        "web_traffic": -1,
        "Page_Rank": -1,
        "Google_Index": 1,
        "Links_pointing_to_page": 1,
        "Statistical_report": -1
    },
    "model_name": "random_forest"
}

response = requests.post(url, json=data)
print(response.json())
```

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "having_IP_Address": -1,
      "URL_Length": 1,
      "SSLfinal_State": -1
    },
    "model_name": "random_forest"
  }'
```

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎯 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Random Forest | 94.2% | 93.8% | 94.5% | 94.1% | 97.8% |
| XGBoost | 93.8% | 93.4% | 94.1% | 93.7% | 97.5% |
| LightGBM | 93.5% | 93.1% | 93.9% | 93.5% | 97.3% |
| Logistic Regression | 91.2% | 90.8% | 91.6% | 91.2% | 95.4% |

*Note: Results may vary based on data and hyperparameter tuning*

## 🔧 Configuration

Edit `src/config/config.yaml` to customize:

### Data Configuration
```yaml
data:
  source: "csv"  # or "mongodb"
  raw_data_path: "data/raw/phisingData.csv"
  train_test_split: 0.2
  random_state: 42
```

### Model Parameters
```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: null
    random_state: 42
```

### Hyperparameter Tuning
```yaml
hyperparameter_tuning:
  enable: True
  method: "randomized"  # or "grid"
  n_iter: 50
  cv_folds: 5
```

### MLflow Tracking
```yaml
mlflow:
  enable: True
  tracking_uri: "mlruns"
  experiment_name: "Network_Security_Phishing_Detection"
```

### Data Drift Detection
```yaml
data_drift:
  enable: True
  method: "ks_test"  # or "psi", "wasserstein"
  threshold: 0.05
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data.py -v

# View coverage report
open htmlcov/index.html
```

## 📊 MLflow Experiment Tracking

```bash
# Start MLflow UI
mlflow ui --backend-store-uri mlruns

# Open browser
http://localhost:5000
```

**Features tracked:**
- All hyperparameters for each model
- Training and test metrics (accuracy, precision, recall, F1, ROC-AUC)
- Training time and dataset statistics
- Feature importance plots
- Confusion matrices
- Model artifacts and metadata

**Compare experiments:**
- Select multiple runs in MLflow UI
- Compare parameters and metrics side-by-side
- Visualize parameter impact on performance

## 🐳 Docker Services

| Service | Port | Description |
|---------|------|-------------|
| **API** | 8000 | FastAPI prediction service |
| **MLflow** | 5000 | Experiment tracking UI |
| **MongoDB** | 27017 | Optional database for data storage |

### Docker Commands

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Restart specific service
docker-compose restart api

# Stop all services
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

## 📚 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint with API info |
| GET | `/health` | Health check status |
| GET | `/models` | List all available models |
| POST | `/predict` | Single URL prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/model/{name}/info` | Detailed model information |

### Example Responses

**Health Check:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": ["random_forest", "xgboost", "lightgbm", "logistic_regression"],
  "timestamp": "2026-01-03T19:00:00"
}
```

**Prediction:**
```json
{
  "prediction": -1,
  "prediction_label": "Phishing",
  "probability": 0.95,
  "confidence": 0.95,
  "model_used": "random_forest",
  "timestamp": "2026-01-03T19:00:00"
}
```

## 🔄 CI/CD Pipeline

Automated workflows triggered on push to main branch:

### 1. **Linting & Code Quality**
- Black code formatter check
- Flake8 linting

### 2. **Testing**
- Unit tests with pytest
- Coverage reports
- Upload to Codecov

### 3. **Docker Build**
- Build training image
- Build API serving image
- Test container health

### 4. **Deployment** (configurable)
- Deploy to cloud platform
- Update production models
- Notification on success/failure

## 📈 Data Pipeline

### 1. Data Ingestion
- Support for CSV and MongoDB
- Initial data validation
- Statistical summaries

### 2. Preprocessing
- Duplicate removal (47% of original data)
- Missing value handling
- Data type validation
- Class balance checks

### 3. Feature Engineering
Creates 7 new features:
- `SSLDomainTrust`: SSL × Domain registration
- `URLSuspicionScore`: Aggregated URL indicators
- `ContentCredibility`: Content trust signals
- `DomainReputation`: Domain age & traffic
- `SecurityFeaturesCount`: Count of positive signals
- `SuspiciousFeaturesCount`: Count of negative signals
- `SSLAnchorInteraction`: SSL × URL anchor

### 4. Validation
- Schema validation
- Value range checks
- Correlation analysis
- Drift detection

## 🔍 Data Drift Detection

Monitors distribution changes between reference and new data.

**Methods available:**
- **KS Test**: Kolmogorov-Smirnov test (p-value threshold)
- **PSI**: Population Stability Index (< 0.1 good, > 0.25 significant)
- **Wasserstein**: Earth Mover's Distance

**Configuration:**
```yaml
data_drift:
  enable: True
  method: "ks_test"
  threshold: 0.05
  features_to_monitor: []  # empty = all features
```

**Reports generated:**
- `drift_report.json`: Machine-readable results
- `drift_report.txt`: Human-readable summary

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Format code
black src/

# Run linter
flake8 src/ --max-line-length=120

# Run tests
pytest tests/ --cov=src
```

## 🛠️ Tech Stack

**Machine Learning:**
- scikit-learn
- XGBoost
- LightGBM
- pandas, numpy

**MLOps:**
- MLflow (experiment tracking)
- Evidently (drift detection)

**API & Serving:**
- FastAPI
- Uvicorn
- Pydantic

**Storage:**
- MongoDB (optional)
- CSV files

**Containerization:**
- Docker
- docker-compose

**CI/CD:**
- GitHub Actions

## 📊 Dataset

The project uses a phishing website dataset with 30 features (all categorical/binary indicators):

- **Original size**: 11,055 rows × 31 columns
- **After preprocessing**: 5,849 rows × 32 columns (with engineered features)
- **Classes**: Binary (-1: Phishing, 1: Legitimate)
- **Class distribution**: ~52% phishing, ~48% legitimate (balanced)

**Feature categories:**
- URL-based features (IP address, length, shortening service)
- Domain-based features (SSL, registration length, age)
- Content-based features (anchors, forms, JavaScript)
- External services (DNS, web traffic, page rank)

## 🎓 Learning Outcomes

This project demonstrates:

✅ **ML Engineering**: Modular code structure, not just notebooks
✅ **MLOps**: Experiment tracking, model registry, versioning
✅ **Production API**: REST API with proper validation and error handling
✅ **Monitoring**: Data drift detection and logging
✅ **DevOps**: Docker, docker-compose, CI/CD pipelines
✅ **Best Practices**: Configuration management, testing, documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Your Name**
- GitHub: [KrishnaSA05](https://github.com/KrishnaSA05)
- LinkedIn: [Krishna Ambekar](www.linkedin.com/in/krishna-ambekar-b4a2641b2)
- Email: krishnaambekar07@gmail.com

## 🙏 Acknowledgments

- Dataset source: [Kaggle Phishing Website Dataset](https://www.kaggle.com/)
- MLflow for excellent experiment tracking
- FastAPI for the modern Python web framework
- scikit-learn, XGBoost, and LightGBM communities

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact via email
- Check the documentation in `/docs`

## 🗺️ Roadmap

Future enhancements:
- [ ] Model drift monitoring
- [ ] A/B testing framework
- [ ] Feature store integration
- [ ] Real-time prediction streaming
- [ ] Multi-cloud deployment
- [ ] Advanced monitoring dashboards
- [ ] Automated model retraining

---

⭐️ **Star this repository if you find it helpful!**

📖 **Full Documentation**: [docs/](docs/)
