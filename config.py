"""
Configuration settings for LSTM Stock Price Prediction
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Model paths
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Data configuration
STOCK_TICKER = "AAPL"
HISTORICAL_YEARS = 5
LOOKBACK_DAYS = 180
PREDICT_DAYS = 30
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model configuration
LSTM_UNITS_LAYER_1 = 50
LSTM_UNITS_LAYER_2 = 50
LSTM_UNITS_LAYER_3 = 25
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1
API_LOG_LEVEL = "info"

# Frontend configuration
STREAMLIT_PORT = 8501
STREAMLIT_THEME = "light"
API_BASE_URL = f"http://localhost:{API_PORT}"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
