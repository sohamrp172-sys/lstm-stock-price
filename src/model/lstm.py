"""
LSTM model architecture for stock price prediction
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import logging
from typing import Dict, Any
import json
import sys
sys.path.insert(0, str(__file__).split('src')[0])
from config import (
    LSTM_UNITS_LAYER_1, LSTM_UNITS_LAYER_2, LSTM_UNITS_LAYER_3,
    DROPOUT_RATE, LEARNING_RATE, LSTM_MODEL_PATH, METADATA_PATH
)

logger = logging.getLogger(__name__)


class LSTMModel:
    """LSTM Neural Network for stock price prediction"""
    
    def __init__(self, lookback: int = 180):
        self.lookback = lookback
        self.model = None
        self.history = None
        
    def build(self) -> models.Sequential:
        """Build LSTM model architecture"""
        self.model = models.Sequential([
            # Layer 1
            layers.LSTM(
                LSTM_UNITS_LAYER_1,
                return_sequences=True,
                input_shape=(self.lookback, 1),
                name='lstm_1'
            ),
            layers.Dropout(DROPOUT_RATE),
            
            # Layer 2
            layers.LSTM(
                LSTM_UNITS_LAYER_2,
                return_sequences=True,
                name='lstm_2'
            ),
            layers.Dropout(DROPOUT_RATE),
            
            # Layer 3
            layers.LSTM(
                LSTM_UNITS_LAYER_3,
                name='lstm_3'
            ),
            layers.Dropout(DROPOUT_RATE),
            
            # Output layer
            layers.Dense(1, name='output')
        ])
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("✓ LSTM Model built successfully")
        logger.info(f"\nModel Summary:")
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              verbose: int = 1) -> Dict[str, Any]:
        """Train the LSTM model"""
        
        if self.model is None:
            self.build()
        
        logger.info(f"Training LSTM for {epochs} epochs...")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info("✓ Training completed")
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'mae': self.history.history['mae'],
            'val_mae': self.history.history['val_mae']
        }
    
    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        return self.model.predict(X, verbose=verbose)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test data"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        
        logger.info("Evaluating model on test data...")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'test_loss': float(results[0]),
            'test_mae': float(results[1]),
            'test_mape': float(results[2])
        }
        
        logger.info(f"✓ Test Loss (MSE): {metrics['test_loss']:.6f}")
        logger.info(f"✓ Test MAE: ${metrics['test_mae']:.6f}")
        logger.info(f"✓ Test MAPE: {metrics['test_mape']*100:.2f}%")
        
        return metrics
    
    def save(self, path: str = None) -> str:
        """Save model to disk"""
        if self.model is None:
            raise ValueError("Model not built.")
        
        path = path or str(LSTM_MODEL_PATH)
        self.model.save(path)
        logger.info(f"✓ Model saved to {path}")
        
        return path
    
    def load(self, path: str = None):
        """Load model from disk"""
        path = path or str(LSTM_MODEL_PATH)
        self.model = keras.models.load_model(path)
        logger.info(f"✓ Model loaded from {path}")
        
        return self
    
    @staticmethod
    def load_model(path: str = None) -> keras.Model:
        """Static method to load a model"""
        path = path or str(LSTM_MODEL_PATH)
        return keras.models.load_model(path)


def save_metadata(metadata: Dict[str, Any], path: str = None):
    """Save model metadata"""
    path = path or str(METADATA_PATH)
    
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata saved to {path}")


def load_metadata(path: str = None) -> Dict[str, Any]:
    """Load model metadata"""
    path = path or str(METADATA_PATH)
    
    with open(path, 'r') as f:
        return json.load(f)
