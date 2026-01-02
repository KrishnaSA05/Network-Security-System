"""
Setup script for Network Security System
"""
from setuptools import setup, find_packages

setup(
    name="network-security-system",
    version="1.0.0",
    description="ML-based Network Security and Phishing Detection System",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1.4",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
        "xgboost>=2.0.3",
        "lightgbm>=4.1.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "pydantic>=2.5.3",
        "mlflow>=2.9.2",
        "pymongo>=4.6.1",
        "pyyaml>=6.0.1",
        "joblib>=1.3.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-models=main:run_complete_pipeline",
            "serve-api=src.serving.app:main",
        ]
    },
)
