"""
FastAPI Backend for LSTM Stock Price Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
import pickle
import json
import yfinance as yf
import logging
import sys
sys.path.insert(0, str(__file__).split('src')[0])

from config import (
    LSTM_MODEL_PATH, SCALER_PATH, METADATA_PATH,
    API_HOST, API_PORT, API_LOG_LEVEL
)
from src.data.data_loader import get_stock_data

# Setup logging
logging.basicConfig(level=API_LOG_LEVEL.upper())
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="LSTM Stock Price Prediction API",
    description="Deep Learning Model for Stock Price Forecasting",
    version="2.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
logger.info("Loading LSTM model and scaler...")
try:
    model = tf.keras.models.load_model(str(LSTM_MODEL_PATH))
    with open(str(SCALER_PATH), 'rb') as f:
        scaler = pickle.load(f)
    with open(str(METADATA_PATH), 'r') as f:
        metadata = json.load(f)
    logger.info("✓ Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"✗ Error loading model: {e}")
    raise

# Cache for stock data
_stock_data_cache = {}
_cache_timestamp = {}


# Request/Response models
class PredictionRequest(BaseModel):
    ticker: str = "AAPL"
    days_ahead: int = 30


class StockDataRequest(BaseModel):
    ticker: str = "AAPL"
    days: int = 180


class PredictionResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    prediction_confidence: float
    prediction_date: str
    lookback_days: int
    recent_trend: str


class StockDataResponse(BaseModel):
    ticker: str
    dates: List[str]
    prices: List[float]
    predictions: List[float]


# Utility functions
def denormalize_value(normalized_value: float) -> float:
    """Convert normalized value back to actual stock price"""
    return float(normalized_value) * (metadata['scaler_max'] - metadata['scaler_min']) + metadata['scaler_min']


def normalize_value(actual_value: float) -> float:
    """Convert actual stock price to normalized value"""
    return (float(actual_value) - metadata['scaler_min']) / (metadata['scaler_max'] - metadata['scaler_min'])


def get_cached_stock_data(ticker: str, days: int = 500) -> pd.DataFrame:
    """Get stock data with caching"""
    cache_key = f"{ticker}_{days}"
    
    if cache_key in _stock_data_cache:
        if datetime.now() - _cache_timestamp.get(cache_key, datetime.min) < timedelta(hours=1):
            return _stock_data_cache[cache_key]
    
    data = get_stock_data(ticker, days=days, progress=False)
    
    _stock_data_cache[cache_key] = data
    _cache_timestamp[cache_key] = datetime.now()
    
    return data


# API Endpoints

@app.get("/")
async def root():
    """API root with metadata"""
    return {
        "status": "online",
        "api": "LSTM Stock Price Prediction API",
        "version": "2.0.0",
        "model": {
            "type": "LSTM Neural Network",
            "layers": metadata['architecture']['layers'],
            "lookback_days": metadata['lookback_days'],
            "test_rmse": f"${metadata['test_rmse']:.6f}",
            "test_mape": f"{metadata['test_mape']*100:.2f}%"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metadata")
async def get_metadata():
    """Get model metadata"""
    return metadata


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Predict stock price"""
    try:
        ticker = request.ticker.upper()
        
        # Fetch data
        data = get_cached_stock_data(ticker, days=500)
        
        # Extract close prices
        if 'Close' in data.columns:
            prices = data['Close'].values
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                prices = data[numeric_cols[0]].values
            else:
                raise ValueError("No valid price data found")
        
        # Normalize
        normalized_prices = (prices - metadata['scaler_min']) / (metadata['scaler_max'] - metadata['scaler_min'])
        
        # Prepare sequence
        lookback = metadata['lookback_days']
        if len(normalized_prices) < lookback:
            raise ValueError(f"Need at least {lookback} data points")
        
        sequence = normalized_prices[-lookback:].reshape(1, lookback, 1)
        
        # Predict
        normalized_pred = float(model.predict(sequence, verbose=0)[0][0])
        predicted_actual_price = denormalize_value(normalized_pred)
        
        # Calculate trend
        recent_prices = prices[-30:]
        trend = "up" if float(recent_prices[-1]) > float(recent_prices[0]) else "down"
        
        # Confidence
        recent_volatility = float(np.std(recent_prices)) / float(np.mean(recent_prices))
        confidence = max(0.5, min(1.0, 1.0 - recent_volatility))
        
        current_price = float(prices[-1])
        
        return PredictionResponse(
            ticker=ticker,
            current_price=current_price,
            predicted_price=predicted_actual_price,
            prediction_confidence=round(confidence, 2),
            prediction_date=(datetime.now() + timedelta(days=request.days_ahead)).strftime("%Y-%m-%d"),
            lookback_days=metadata['lookback_days'],
            recent_trend=f"{trend} ({float(recent_prices[-1]):.2f} vs {float(recent_prices[0]):.2f})"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/historical")
async def get_historical(request: StockDataRequest):
    """Get historical data with predictions"""
    try:
        ticker = request.ticker.upper()
        
        # Limit to prevent timeout
        days = min(request.days, 365)
        
        # Fetch data
        data = get_cached_stock_data(ticker, days=int(days * 1.5))
        
        # Extract prices
        if 'Close' in data.columns:
            all_prices = data['Close'].values
        else:
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                all_prices = data[numeric_cols[0]].values
            else:
                raise ValueError("No valid price data found")
        
        prices = all_prices[-days:]
        dates = [d.strftime("%Y-%m-%d") for d in data.index[-days:]]
        
        # Normalize
        normalized_all = (all_prices - metadata['scaler_min']) / (metadata['scaler_max'] - metadata['scaler_min'])
        
        # Generate predictions
        predictions = []
        lookback = metadata['lookback_days']
        
        max_predictions = min(100, len(normalized_all) - lookback)
        start_idx = max(0, len(normalized_all) - lookback - max_predictions)
        
        for i in range(start_idx, len(normalized_all) - lookback):
            seq = normalized_all[i:i+lookback].reshape(1, lookback, 1)
            pred_norm = float(model.predict(seq, verbose=0)[0][0])
            pred_actual = denormalize_value(pred_norm)
            predictions.append(pred_actual)
        
        # Pad
        total_padding = len(prices) - len(predictions)
        predictions = [None] * total_padding + predictions
        
        return StockDataResponse(
            ticker=ticker,
            dates=dates,
            prices=[float(p) for p in prices],
            predictions=[float(p) if p is not None else None for p in predictions]
        )
        
    except Exception as e:
        logger.error(f"Historical data error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


@app.get("/current/{ticker}")
async def get_current_price(ticker: str):
    """Get current stock price"""
    try:
        ticker = ticker.upper()
        
        cache_key = f"current_{ticker}"
        if cache_key in _stock_data_cache:
            if datetime.now() - _cache_timestamp.get(cache_key, datetime.min) < timedelta(minutes=5):
                return _stock_data_cache[cache_key]
        
        data = yf.download(ticker, period='1d', progress=False)
        
        if data.empty:
            raise ValueError(f"No data for {ticker}")
        
        if isinstance(data.columns, pd.MultiIndex):
            data = data[ticker] if ticker in data.columns.get_level_values(0) else data
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        result = {
            "ticker": ticker,
            "current_price": float(data['Close'].iloc[-1]),
            "timestamp": datetime.now().isoformat(),
            "high": float(data['High'].iloc[-1]),
            "low": float(data['Low'].iloc[-1]),
            "volume": int(data['Volume'].iloc[-1]) if 'Volume' in data.columns else 0
        }
        
        _stock_data_cache[cache_key] = result
        _cache_timestamp[cache_key] = datetime.now()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


def run_server():
    """Run the FastAPI server"""
    import uvicorn
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level=API_LOG_LEVEL
    )


if __name__ == "__main__":
    run_server()
