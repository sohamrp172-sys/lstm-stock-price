"""
Data loading and preparation module for stock price prediction
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging
from typing import Tuple, Dict, Any
import sys
sys.path.insert(0, str(__file__).split('src')[0])
from config import DATA_DIR, HISTORICAL_YEARS, LOOKBACK_DAYS, TEST_SIZE, VALIDATION_SIZE

logger = logging.getLogger(__name__)


class StockDataLoader:
    """Load and preprocess stock data for LSTM training"""
    
    def __init__(self, ticker: str = "AAPL", years: int = HISTORICAL_YEARS):
        self.ticker = ticker.upper()
        self.years = years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.prices = None
        self.normalized_prices = None
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance"""
        logger.info(f"Fetching {self.years} years of {self.ticker} data...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.years * 365)
        
        try:
            self.data = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            if self.data.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            # Handle multi-level columns
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data = self.data[self.ticker] if self.ticker in self.data.columns.get_level_values(0) else self.data
                self.data.columns = [col[0] if isinstance(col, tuple) else col for col in self.data.columns]
            
            logger.info(f"✓ Downloaded {len(self.data)} data points")
            return self.data
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize and prepare data for LSTM"""
        if self.data is None:
            self.fetch_data()
        
        # Extract close prices
        self.prices = self.data['Close'].values.reshape(-1, 1)
        logger.info(f"✓ Extracted {len(self.prices)} close prices")
        
        # Normalize
        self.normalized_prices = self.scaler.fit_transform(self.prices)
        logger.info(f"✓ Normalized prices (range: {self.normalized_prices.min():.4f} to {self.normalized_prices.max():.4f})")
        
        return self.normalized_prices, self.prices
    
    def create_sequences(self, lookback: int = LOOKBACK_DAYS) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        if self.normalized_prices is None:
            self.prepare_data()
        
        X, y = [], []
        
        for i in range(lookback, len(self.normalized_prices)):
            X.append(self.normalized_prices[i - lookback:i, 0])
            y.append(self.normalized_prices[i, 0])
        
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y).reshape(-1, 1)
        
        logger.info(f"✓ Created {len(X)} sequences of length {lookback}")
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Split data into train, validation, and test sets"""
        n_train = int(len(X) * (1 - TEST_SIZE - VALIDATION_SIZE))
        n_val = int(len(X) * VALIDATION_SIZE)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        logger.info(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_scaler_params(self) -> Dict[str, float]:
        """Get scaler min and max for denormalization"""
        return {
            'min': float(self.scaler.data_min_[0]),
            'max': float(self.scaler.data_max_[0])
        }
    
    def save_scaler(self, path: str = None):
        """Save scaler for later use"""
        if path is None:
            from config import SCALER_PATH
            path = SCALER_PATH
        
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"✓ Scaler saved to {path}")
    
    @staticmethod
    def load_scaler(path: str) -> MinMaxScaler:
        """Load saved scaler"""
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler


def get_stock_data(ticker: str, days: int = 1000, progress: bool = False) -> pd.DataFrame:
    """Utility function to fetch stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = yf.download(ticker, start=start_date, end=end_date, progress=progress)
    
    if isinstance(data.columns, pd.MultiIndex):
        data = data[ticker.upper()] if ticker.upper() in data.columns.get_level_values(0) else data
        data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    
    return data
