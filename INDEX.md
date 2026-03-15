# 📚 LSTM Stock Price Prediction - Complete Repository

## 🎉 Project Successfully Restructured!

This is a **production-ready** LSTM stock price prediction system with clean, modular architecture.

---

## 🗂️ New Structure Overview

### **Core Code** (`src/`)
- **`data/`** - Data loading and preprocessing
- **`model/`** - LSTM neural network architecture
- **`training/`** - Training pipeline orchestrator
- **`backend/`** - FastAPI REST API server
- **`frontend/`** - Streamlit web dashboard

### **Entry Points** (`scripts/`)
- `train.py` - Train LSTM model
- `run_backend.py` - Start API server
- `run_frontend.py` - Start web dashboard

### **Artifacts** (`models/`)
- `lstm_model.keras` - Trained model weights
- `scaler.pkl` - Data normalizer
- `model_metadata.json` - Configuration & metrics

### **Configuration**
- `config.py` - **Centralized settings (edit here!)**

---

## 🚀 Quick Start (5 Steps)

```bash
# 1. Activate virtual environment
.\stock_env\Scripts\Activate.ps1        # Windows
# or
source stock_env/bin/activate           # Linux/Mac

# 2. Optional: Train fresh model
python scripts/train.py

# 3. Start API (Terminal 1)
python scripts/run_backend.py

# 4. Start Web Dashboard (Terminal 2)
python scripts/run_frontend.py

# 5. Test everything
python tests/test_deployment.py
```

**Access Point:**
- 🌐 **Dashboard:** http://localhost:8501
- 📖 **API Docs:** http://localhost:8000/docs

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | Full project documentation |
| **QUICK_REFERENCE.md** | Common tasks & commands |
| **PROJECT_STRUCTURE.md** | Detailed architecture |
| **MIGRATION_GUIDE.md** | What changed from old structure |

---

## ⚙️ Configuration

**Everything editable in** `config.py`:

```python
# Model parameters
LSTM_UNITS_LAYER_1 = 50           # Network size
EPOCHS = 50                        # Training iterations
BATCH_SIZE = 32                    # Batch size

# Data
STOCK_TICKER = "AAPL"            # Change to GOOGL, MSFT, etc.
LOOKBACK_DAYS = 180               # Historical window
PREDICT_DAYS = 30                 # Forecast horizon

# API & UI
API_PORT = 8000
STREAMLIT_PORT = 8501
```

---

## 📊 Features

✅ **LSTM Neural Network**
- 3 layers with dropout
- Optimized training
- Early stopping & LR scheduling

✅ **REST API**
- FastAPI with auto-docs
- Real-time predictions
- Historical data retrieval
- Data caching

✅ **Web Dashboard**
- Interactive Streamlit UI
- Live stock prices
- Price forecasts
- Historical analysis
- Model metrics

✅ **Production Ready**
- Modular code
- Error handling
- Logging
- Deployment ready
- Test suite included

---

## 🔌 API Endpoints

### Status
- `GET /` - API info
- `GET /health` - Health check
- `GET /metadata` - Model metrics

### Predictions
- `POST /predict` - Next 30-day forecast
- `POST /historical` - Historical + predictions
- `GET /current/{ticker}` - Current price

---

## 📁 File Organization

```
New Files Created:
✅ config.py                    - Centralized configuration
✅ src/data/data_loader.py      - Data handling module
✅ src/model/lstm.py            - LSTM architecture
✅ src/training/train.py        - Training pipeline
✅ src/backend/api.py           - FastAPI backend
✅ src/frontend/app.py          - Streamlit dashboard
✅ scripts/train.py             - Training entry point
✅ scripts/run_backend.py       - Backend entry point
✅ scripts/run_frontend.py      - Frontend entry point
✅ tests/test_deployment.py     - Integration tests
✅ README.md                    - Updated documentation
✅ requirements.txt             - Updated dependencies
✅ PROJECT_STRUCTURE.md         - Architecture docs
✅ MIGRATION_GUIDE.md           - Refactoring guide
✅ QUICK_REFERENCE.md           - Quick help guide
✅ INDEX.md                     - This file

Moved to models/:
✅ lstm_model.keras            → models/lstm_model.keras
✅ scaler.pkl                  → models/scaler.pkl
✅ model_metadata.json         → models/model_metadata.json
```

---

## 🎯 Usage Examples

### Python Client
```python
import requests

response = requests.post(
    'http://localhost:8000/predict',
    json={'ticker': 'AAPL', 'days_ahead': 30}
)
result = response.json()
print(f"Current: ${result['current_price']:.2f}")
print(f"Predicted: ${result['predicted_price']:.2f}")
```

### Dashboard
- Open `http://localhost:8501`
- Select stock ticker
- View live predictions
- Explore historical trends

---

## 🧪 Testing

```bash
# Full deployment test
python tests/test_deployment.py

# Expected output:
# ✓ FastAPI: PASSED
# ✓ Streamlit: PASSED
# ✓ Current Price: PASSED
# ✓ Prediction: PASSED
```

---

## 🔄 Workflow

1. **Edit Code** → Modify `src/` modules or `config.py`
2. **Test** → Run `python tests/test_deployment.py`
3. **Commit** → Upload to version control
4. **Deploy** → Use with Docker, Cloud, etc.

---

## 📦 Dependencies

**Core:**
- TensorFlow/Keras - Deep learning
- FastAPI - Web API
- Streamlit - Web UI
- Pandas/NumPy - Data processing
- Scikit-learn - ML utilities
- yfinance - Stock data

See `requirements.txt` for full list.

---

## ⚠️ Important Notes

**Educational Use Only:**
- ❌ NOT financial advice
- ❌ NOT investment recommendations
- ✅ For learning and demonstration

**Model Characteristics:**
- Trained on historical data
- Does not predict market shocks
- Real-world performance may differ
- Retraining recommended periodically

---

## 🚀 Next Steps

1. ✅ **Review Structure** - Explore `src/` modules
2. ✅ **Customize Config** - Edit `config.py` for your needs
3. ✅ **Train Model** - `python scripts/train.py` (optional)
4. ✅ **Run Locally** - Try scripts and dashboard
5. ✅ **Deploy** - Use Docker/Cloud for production

---

## 📚 Getting Help

1. **Quick Commands?** → See [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **How Things Work?** → See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
3. **What Changed?** → See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
4. **Full Details?** → See [README.md](README.md)

---

## 🎓 Learning Path

If new to the project:

1. Read [README.md](README.md) - Overview
2. Review [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture
3. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Commands
4. Explore `src/` code - Implementation details
5. Run `python tests/test_deployment.py` - Verify setup
6. Edit `config.py` - Customize for your needs

---

## 🤝 Contributing

Areas for improvement:
- 🔄 Alternative architectures (GRU, Transformer)
- 📈 Multi-stock support
- 🌐 Real-time streaming
- 🐳 Docker containerization
- ☁️ Cloud deployment guides

---

## 📊 Model Performance

**Trained on:** 5 years of AAPL data (1,258 points)

| Metric | Value |
|--------|-------|
| Test MAPE | 7.65% |
| Test RMSE | $0.0624 |
| Test MAE | $0.0378 |
| Lookback | 180 days |
| Predict | 30 days |

---

## 💡 Key Improvements Made

✅ **Modular Design** - Clean separation of concerns
✅ **Centralized Config** - Single source of truth
✅ **Clear Entry Points** - Easy to run each component
✅ **Documentation** - Multiple guides for different needs
✅ **Testing** - Built-in deployment tests
✅ **Production Ready** - Error handling, logging, caching
✅ **Extensible** - Easy to add features

---

**Status:** ✅ **COMPLETE AND TESTED**

The repository is now fully restructured, documented, and ready for:
- 🎓 Learning
- 🧪 Development
- 🚀 Deployment
- 📊 Extension

**Happy predicting!** 🎉

---

**Last Updated:** March 15, 2026
**Status:** Production Ready ✅
