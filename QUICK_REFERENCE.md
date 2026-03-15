# Quick Reference Guide

## Getting Started

### 1️⃣ Activate Virtual Environment
```bash
# Windows
.\stock_env\Scripts\Activate.ps1

# Linux/Mac
source stock_env/bin/activate
```

### 2️⃣ Train Model (Optional)
```bash
python scripts/train.py
```

### 3️⃣ Start Backend API
**Terminal 1:**
```bash
python scripts/run_backend.py
```
🌐 API runs on: `http://localhost:8000`
📖 API docs at: `http://localhost:8000/docs`

### 4️⃣ Start Frontend
**Terminal 2:**
```bash
python scripts/run_frontend.py
```
🎨 Dashboard at: `http://localhost:8501`

### 5️⃣ Test Deployment
```bash
python tests/test_deployment.py
```

---

## File Locations

| What | Where |
|------|-------|
| **Model** | `models/lstm_model.keras` |
| **Scaler** | `models/scaler.pkl` |
| **Config** | `models/model_metadata.json` |
| **Settings** | `config.py` |
| **Data Loader** | `src/data/data_loader.py` |
| **LSTM Model** | `src/model/lstm.py` |
| **Training** | `src/training/train.py` |
| **API** | `src/backend/api.py` |
| **Dashboard** | `src/frontend/app.py` |

---

## Configuration (Edit: `config.py`)

### Change Stock Ticker
```python
STOCK_TICKER = "GOOGL"  # AAPL, MSFT, TSLA, etc.
```

### Adjust Model Size
```python
LSTM_UNITS_LAYER_1 = 64      # Increase complexity
EPOCHS = 100                  # More training
BATCH_SIZE = 16               # Smaller batches
```

### Change Prediction Window
```python
LOOKBACK_DAYS = 365           # Longer memory
PREDICT_DAYS = 60             # Longer forecast
HISTORICAL_YEARS = 10         # More data
```

### Change Ports
```python
API_PORT = 8000      # FastAPI port
STREAMLIT_PORT = 8501  # Streamlit port
```

---

## API Usage Examples

### Python
```python
import requests

# Predict AAPL for 30 days
response = requests.post(
    'http://localhost:8000/predict',
    json={'ticker': 'AAPL', 'days_ahead': 30}
)
print(response.json())
```

### cURL
```bash
# Current price
curl http://localhost:8000/current/AAPL

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days_ahead": 30}'

# Historical data
curl -X POST http://localhost:8000/historical \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "days": 180}'
```

---

## Common Tasks

### Retrain Model with New Data
```bash
# Update STOCK_TICKER in config.py if needed
python scripts/train.py
```

### Clear Cache
```bash
# Delete __pycache__ directories
find . -type d -name __pycache__ -delete

# Python will regenerate on next run
```

### Check Model Performance
```bash
# Look at models/model_metadata.json
cat models/model_metadata.json
```

### View Training Logs
```bash
# Logs printed to console during training
python scripts/train.py  # See console output
```

---

## Troubleshooting

### API won't start
```bash
# Port 8000 might be in use
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Frontend can't connect to API
```bash
# Ensure backend is running
# Check API_BASE_URL in config.py
# Verify firewall allows localhost:8000
```

### Model not found
```bash
# Must train first or copy models/
python scripts/train.py
```

### Slow predictions
```bash
# First request loads model (~3s)
# Subsequent requests faster (~0.5s)
# Can add caching for production
```

---

## Project Structure

```
lstm-stock-price-prediction/
├── config.py              # ⭐ EDIT THIS for settings
├── requirements.txt       # Dependencies
├── README.md              # Full documentation
│
├── src/
│   ├── data/data_loader.py        # Data handling
│   ├── model/lstm.py              # Model architecture  
│   ├── training/train.py          # Training pipeline
│   ├── backend/api.py             # FastAPI
│   └── frontend/app.py            # Streamlit
│
├── scripts/
│   ├── train.py           # python scripts/train.py
│   ├── run_backend.py     # python scripts/run_backend.py
│   └── run_frontend.py    # python scripts/run_frontend.py
│
├── models/                # Trained artifacts
│   ├── lstm_model.keras   # Model weights
│   ├── scaler.pkl         # Data normalizer
│   └── model_metadata.json # Config & metrics
│
├── tests/
│   └── test_deployment.py # Integration tests
│
└── stock_env/            # Virtual environment
```

---

## Key Metrics

Model trained on 5 years of AAPL data:

| Metric | Value |
|--------|-------|
| Test RMSE | $0.0624 |
| Test MAPE | 7.65% |
| Test MAE | $0.0378 |
| Data Points | 1,258 |
| Lookback | 180 days |

---

## Performance Tips

✅ Model loading is cached (first request ~3s, rest <1s)
✅ Stock data is cached for 1 hour
✅ Current prices cached for 5 minutes
✅ Predictions computed on demand

---

## Dependencies

**Core:**
- tensorflow, keras
- fastapi, uvicorn
- streamlit
- pandas, numpy
- scikit-learn
- yfinance

**Full list:** See `requirements.txt`

---

## Resources

📖 **Documentation:** See [README.md](README.md)
🏗️ **Architecture:** See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
🔄 **Migration:** See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)

---

## Support

If something isn't working:

1. Check API is running: `curl http://localhost:8000/health`
2. Check ports not in use: `netstat -an`
3. Verify model exists: `ls models/`
4. Run tests: `python tests/test_deployment.py`
5. Check logs in console

---

**Happy predicting! 🚀**
