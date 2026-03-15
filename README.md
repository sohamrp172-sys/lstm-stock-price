# LSTM Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)

Production-ready deep learning system for stock price forecasting using LSTM neural networks with FastAPI backend and Streamlit frontend.

## 📋 Project Structure

```
lstm-stock-price-prediction/
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                # Documentation
│
├── src/
│   ├── data/                # Data loading and preparation
│   │   └── data_loader.py   #   StockDataLoader class
│   ├── model/               # LSTM model architecture
│   │   └── lstm.py          #   LSTMModel class
│   ├── training/            # Training pipeline
│   │   └── train.py         #   TrainingPipeline class
│   ├── backend/             # FastAPI REST API
│   │   └── api.py           #   API endpoints
│   └── frontend/            # Streamlit web UI
│       └── app.py           #   Dashboard
│
├── scripts/                 # Entry points
│   ├── train.py            # python scripts/train.py
│   ├── run_backend.py      # python scripts/run_backend.py
│   └── run_frontend.py     # python scripts/run_frontend.py
│
├── models/                  # Trained models (artifacts)
│   ├── lstm_model.keras    # Pre-trained weights
│   ├── scaler.pkl          # Data normalizer
│   └── model_metadata.json # Config & metrics
│
├── data/                    # Data directory (empty)
├── tests/                   # Test suite
│   └── test_deployment.py  # Integration tests
│
└── stock_env/              # Virtual environment
```

---

## ✨ Features

✅ **3-Layer LSTM Network** - Deep learning for time series
✅ **FastAPI Backend** - REST API with auto-documentation  
✅ **Streamlit Dashboard** - Interactive web UI
✅ **Data Caching** - Optimized performance
✅ **Modular Architecture** - Easy to extend
✅ **Production Ready** - Error handling, logging
✅ **Deployment Tests** - Verify setup

---

## 🚀 Quick Start

### 1. Setup (First Time Only)

```bash
# Activate virtual environment
.\stock_env\Scripts\Activate.ps1  # Windows
source stock_env/bin/activate      # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Model (Optional)

```bash
python scripts/train.py
```

Expected output:
```
Test MAPE: 7.65%
Test RMSE: $0.0624
Training time: ~5-10 minutes
```

### 3. Start Backend API

```bash
python scripts/run_backend.py
```

Access at: `http://localhost:8000/docs`

### 4. Start Frontend (New Terminal)

```bash
python scripts/run_frontend.py
```

Access at: `http://localhost:8501`

### 5. Test Everything

```bash
python tests/test_deployment.py
```

---

## 🏗️ Architecture

### `src/data/data_loader.py`
Handles data fetching and preparation:
- Downloads historical data from Yahoo Finance
- Normalizes prices using MinMaxScaler
- Creates LSTM sequences (lookback + target)
- Splits into train/val/test sets

### `src/model/lstm.py`
Defines LSTM architecture:
- Layer 1: 50 LSTM units + 0.2 dropout
- Layer 2: 50 LSTM units + 0.2 dropout
- Layer 3: 25 LSTM units + 0.2 dropout
- Output: Dense(1) for price prediction

### `src/training/train.py`
Orchestrates training pipeline:
- Loads and prepares data
- Builds and trains model
- Saves artifacts (model, scaler, metadata)

### `src/backend/api.py`
REST API endpoints:
- `GET /` - Status & metadata
- `GET /health` - Health check
- `POST /predict` - Next 30-day forecast
- `POST /historical` - Historical + predictions
- `GET /current/{ticker}` - Stock price

### `src/frontend/app.py`
Streamlit dashboard with:
- Real-time stock data
- Price forecasts
- Historical analysis
- Performance metrics

---

## 📊 Performance

Tested on AAPL (5 years of data):

| Metric | Value |
|--------|-------|
| Test MAPE | 7.65% |
| Test RMSE | $0.0624 |
| Test MAE | $0.0378 |
| Data Points | 1,258 |
| Model Size | ~2MB |

---

## 🔧 Configuration

Edit `config.py`:

```python
# Model hyperparameters
LSTM_UNITS_LAYER_1 = 50
DROPOUT_RATE = 0.2
EPOCHS = 50
BATCH_SIZE = 32

# Data
STOCK_TICKER = "AAPL"
HISTORICAL_YEARS = 5
LOOKBACK_DAYS = 180

# API
API_PORT = 8000
API_HOST = "0.0.0.0"
```

---

## 📦 Dependencies

Core packages:
- `tensorflow` - Deep learning
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Stock data
- `fastapi` - REST API
- `streamlit` - Web UI
- `plotly` - Interactive charts
- `scikit-learn` - ML utilities

See `requirements.txt` for full list.

---

## 🧪 Testing

### Run Tests

```bash
python tests/test_deployment.py
```

Expected output:
```
✓ FastAPI: PASSED
✓ Streamlit: PASSED  
✓ Current Price: PASSED
✓ Prediction: PASSED
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Get current price
curl http://localhost:8000/current/AAPL

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days_ahead": 30}'
```

---

## 📚 API Endpoints

### Get API Metadata
```http
GET /metadata
```

### Make Prediction
```http
POST /predict
{
  "ticker": "AAPL",
  "days_ahead": 30
}
```

### Get Historical Data
```http
POST /historical
{
  "ticker": "AAPL",
  "days": 180
}
```

### Get Current Price
```http
GET /current/AAPL
```

---

## 🧑‍💻 Customization

### Change Stock Symbol
Edit `config.py`:
```python
STOCK_TICKER = "GOOGL"  # MSFT, TSLA, AMZN, etc.
```

### Adjust Model Size
Edit `config.py`:
```python
LSTM_UNITS_LAYER_1 = 64   # Increase for complexity
EPOCHS = 100               # More training iterations
BATCH_SIZE = 16            # Smaller batches
```

### Modify Data Period
Edit `config.py`:
```python
HISTORICAL_YEARS = 10      # More historical data
LOOKBACK_DAYS = 365        # Longer memory window
PREDICT_DAYS = 60          # Longer predictions
```

---

## ⚠️ Important Notes

**This is educational software:**
- ❌ NOT financial advice
- ❌ NOT investment recommendations
- ❌ Past performance ≠ future results
- ✅ For learning purposes only
- ✅ Always research before investing

**Model considerations:**
- Trained on historical data only
- Does not account for market shocks
- Predictions are estimates not guarantees
- Real-world performance may differ significantly

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| API won't start | Port 8000 in use → `netstat -an \| grep 8000` |
| Frontend can't connect | Verify `API_BASE_URL` in config |
| Model not found | Run `python scripts/train.py` first |
| 404 errors | Ensure backend is running on port 8000 |
| Slow predictions | Model loading or network latency |

---

## 📦 Requirements Versions

See `requirements.txt` for complete list:
- TensorFlow 2.13+
- FastAPI 0.100+
- Streamlit 1.40+
- Pandas 2.0+
- Numpy 1.24+
- Scikit-learn 1.3+

---

## 📄 License

MIT License - See [LICENSE](LICENSE)

---

## 🤝 Contributing

Improvements and contributions welcome!

Areas for enhancement:
- Alternative architectures (GRU, Transformer)
- Ensemble methods
- More stock exchanges
- Real-time data streaming
- Docker support
- Mobile app

---

**Built with TensorFlow, FastAPI, and Streamlit** ❤️
