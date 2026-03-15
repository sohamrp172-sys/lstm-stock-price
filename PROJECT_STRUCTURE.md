# Project Structure Overview

## Directory Layout

```
lstm-stock-price-prediction/
│
├── 📄 config.py                    # CENTRALIZED CONFIGURATION
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Project documentation
├── 📄 MIGRATION_GUIDE.md           # Refactoring summary
├── 📄 LICENSE                      # MIT License
│
├── 🗂️ src/                          # SOURCE CODE
│   │
│   ├── 📁 data/                    # Data loading and preparation
│   │   ├── __init__.py
│   │   └── data_loader.py          # StockDataLoader class
│   │                               # - fetch_data()
│   │                               # - prepare_data()
│   │                               # - create_sequences()
│   │                               # - split_data()
│   │
│   ├── 📁 model/                   # LSTM model architecture
│   │   ├── __init__.py
│   │   └── lstm.py                 # LSTMModel class
│   │                               # - build()
│   │                               # - train()
│   │                               # - predict()
│   │                               # - save()
│   │                               # - load()
│   │
│   ├── 📁 training/                # Training pipeline
│   │   ├── __init__.py
│   │   └── train.py                # TrainingPipeline class
│   │                               # - load_data()
│   │                               # - build_and_train()
│   │                               # - save_artifacts()
│   │                               # - run()
│   │
│   ├── 📁 backend/                 # FastAPI REST API
│   │   ├── __init__.py
│   │   └── api.py                  # FastAPI endpoints
│   │                               # - GET /
│   │                               # - POST /predict
│   │                               # - POST /historical
│   │                               # - GET /current/{ticker}
│   │
│   └── 📁 frontend/                # Streamlit web dashboard
│       ├── __init__.py
│       └── app.py                  # Streamlit application
│                                   # - Dashboard
│                                   # - Real-time predictions
│                                   # - Historical charts
│
├── 🗂️ scripts/                      # ENTRY POINTS
│   ├── train.py                    # python scripts/train.py
│   ├── run_backend.py              # python scripts/run_backend.py
│   └── run_frontend.py             # python scripts/run_frontend.py
│
├── 🗂️ models/                       # TRAINED ARTIFACTS
│   ├── lstm_model.keras            # Trained LSTM weights
│   ├── scaler.pkl                  # MinMaxScaler for normalization
│   └── model_metadata.json         # Model config and metrics
│
├── 🗂️ data/                         # DATA DIRECTORY
│   └── .gitkeep                    # (Empty, for user data)
│
├── 🗂️ tests/                        # TEST SUITE
│   ├── __init__.py
│   └── test_deployment.py          # Integration tests
│
├── 🗂️ stock_env/                    # PYTHON VIRTUAL ENVIRONMENT
│   └── ...
│
├── 📄 .gitignore                   # Git ignore rules
└── 📄 DEPLOYMENT_NOTES.md          # Additional notes
```

---

## Module Relationships

```
┌─────────────────────────────────────────────────────────────┐
│                         config.py                            │
│              (Centralized Configuration)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┬──────────────┐
        │            │            │              │
        ▼            ▼            ▼              ▼
   ┌─────────┐  ┌────────┐  ┌────────┐  ┌──────────┐
   │   data/ │  │ model/ │  │training│  │ backend/ │
   │         │  │        │  │        │  │          │
   │Loader   │  │ LSTM   │  │Pipeline│  │ FastAPI  │
   └────┬────┘  └───┬────┘  └────┬───┘  │ API      │
        │           │            │      └────┬─────┘
        │           │            │           │
        └───────────┼────────────┴───────────┘
                    │
               ┌────┴────┐
               ▼         ▼
           ┌────────┬──────────┐
           │ models/│ frontend/│
           │        │          │
           │Weights│Streamlit │
           │Scaler │ Dashboard│
           └────────┴──────────┘
```

---

## Data Flow

### Training Flow
```
Config.py
   │
   ├─> src/data/data_loader.py
   │   └─> Fetch AAPL data (5 years)
   │   └─> Normalize & create sequences
   │
   ├─> src/model/lstm.py
   │   └─> Build 3-layer LSTM
   │   └─> Train on data splits
   │
   └─> src/training/train.py
       └─> Orchestrate pipeline
       └─> Save to models/
```

### Inference Flow
```
Request (HTTP)
   │
   └─> src/backend/api.py
       └─> Load from models/
       └─> Get stock data
       └─> Normalize
       └─> Predict
       └─> Denormalize
       └─> Return JSON

Response (JSON)
   │
   └─> src/frontend/app.py
       └─> Display charts
       └─> Show predictions
       └─> Interactive UI
```

---

## File Dependencies

```
src/frontend/app.py
├─ config.py (API_BASE_URL, ports)
└─ requests (HTTP client)

src/backend/api.py  
├─ config.py (model paths, API settings)
├─ tensorflow (load model)
├─ src/data/data_loader.py (get_stock_data)
└─ pickle (scaler)

src/training/train.py
├─ config.py (all settings)
├─ src/data/data_loader.py (StockDataLoader)
└─ src/model/lstm.py (LSTMModel)

scripts/train.py
└─ src/training/train.py (train function)

scripts/run_backend.py
└─ src/backend/api.py (run_server)

scripts/run_frontend.py
└─ src/frontend/app.py (streamlit run)
```

---

## Configuration Hierarchy

All settings centralized in `config.py`:

```python
# 1. Paths
PROJECT_ROOT
DATA_DIR
MODELS_DIR
LSTM_MODEL_PATH
SCALER_PATH
METADATA_PATH

# 2. Model Parameters
LOOKBACK_DAYS = 180
PREDICT_DAYS = 30
EPOCHS = 50
LSTM_UNITS_LAYER_1 = 50
DROPOUT_RATE = 0.2

# 3. Data Settings
STOCK_TICKER = "AAPL"
HISTORICAL_YEARS = 5
BATCH_SIZE = 32

# 4. API Settings
API_PORT = 8000
API_HOST = "0.0.0.0"

# 5. Frontend Settings
STREAMLIT_PORT = 8501
API_BASE_URL = "http://localhost:8000"
```

---

## Class Hierarchy

```
StockDataLoader (src/data/data_loader.py)
├─ __init__(ticker, years)
├─ fetch_data()
├─ prepare_data()
├─ create_sequences()
├─ split_data()
├─ get_scaler_params()
└─ save_scaler()

LSTMModel (src/model/lstm.py)
├─ __init__(lookback)
├─ build()
├─ train()
├─ predict()
├─ evaluate()
├─ save()
└─ load()

TrainingPipeline (src/training/train.py)
├─ __init__(ticker, years)
├─ load_data()
├─ build_and_train()
├─ calculate_train_metrics()
├─ save_artifacts()
└─ run()
```

---

## API Endpoints

```
FastAPI (port 8000)

GET /
├─ Status: online
├─ Model info
└─ Available endpoints

GET /health
└─ Health check

GET /metadata
└─ Model config & metrics

POST /predict
├─ Input: ticker, days_ahead
└─ Output: prediction response

POST /historical
├─ Input: ticker, days
└─ Output: historical data + predictions

GET /current/{ticker}
└─ Current stock price
```

---

## Frontend Pages

```
Streamlit (port 8501)

📊 Dashboard
├─ Current price
├─ High/Low
├─ Volume

🎯 Predictions
├─ Price forecast
├─ Confidence level
└─ Trend analysis

📈 Historical
├─ Price chart
├─ Model overlay
└─ Data table

ℹ️ About
└─ Project info
```

---

## Development Workflow

```
1. Modify code
   └─ Update src/ modules
   └─ Update config.py if needed

2. Test locally
   └─ python scripts/train.py (optional)
   └─ python scripts/run_backend.py
   └─ python scripts/run_frontend.py
   └─ python tests/test_deployment.py

3. Commit
   └─ git add .
   └─ git commit -m "message"
   └─ git push

4. Deploy
   └─ Docker or Cloud platform
```

---

## Key Design Principles

✅ **Single Responsibility** - Each module has one job
✅ **DRY (Don't Repeat Yourself)** - Centralized config
✅ **Modularity** - Easy to extend or replace
✅ **Clear Separation** - Data/Model/Training/API/UI
✅ **Testability** - Can test components independently
✅ **Maintainability** - Well-organized and documented

---

## Performance Characteristics

| Component | Startup | Query |
|-----------|---------|-------|
| Data Loader | ~2s (first), cached after | N/A |
| LSTM Model | ~3s load | ~0.5s predict |
| FastAPI | ~2s | ~1.5s (API round trip) |
| Streamlit | ~5s | N/A |
| Prediction End-to-End | N/A | ~2s |

---

**This modular structure is production-ready and easy to maintain!** 🚀
