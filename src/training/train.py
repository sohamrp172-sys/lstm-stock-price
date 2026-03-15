"""
Training pipeline for LSTM Stock Price Prediction
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any
import sys
sys.path.insert(0, str(__file__).split('src')[0])

from config import (
    STOCK_TICKER, LOOKBACK_DAYS, PREDICT_DAYS, EPOCHS, BATCH_SIZE,
    SCALER_PATH, METADATA_PATH, LSTM_MODEL_PATH
)
from src.data.data_loader import StockDataLoader
from src.model.lstm import LSTMModel, save_metadata, load_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, ticker: str = STOCK_TICKER, years: int = 5):
        self.ticker = ticker
        self.years = years
        self.data_loader = None
        self.model = None
        self.metadata = {}
        
    def load_data(self):
        """Step 1: Load and prepare data"""
        logger.info("=" * 60)
        logger.info("STEP 1: Loading Data")
        logger.info("=" * 60)
        
        self.data_loader = StockDataLoader(ticker=self.ticker, years=self.years)
        self.data_loader.fetch_data()
        self.data_loader.prepare_data()
        
        # Create sequences
        X, y = self.data_loader.create_sequences(lookback=LOOKBACK_DAYS)
        
        # Split data
        splits = self.data_loader.split_data(X, y)
        
        logger.info("✓ Data loading completed\n")
        
        return splits
    
    def build_and_train(self, splits: Dict[str, tuple]):
        """Step 2: Build and train model"""
        logger.info("=" * 60)
        logger.info("STEP 2: Building and Training Model")
        logger.info("=" * 60)
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Build model
        self.model = LSTMModel(lookback=LOOKBACK_DAYS)
        self.model.build()
        
        # Train
        self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        
        # Evaluate
        test_metrics = self.model.evaluate(X_test, y_test)
        
        logger.info("✓ Model training and evaluation completed\n")
        
        return test_metrics, (X_test, y_test)
    
    def calculate_train_metrics(self, splits: Dict[str, tuple]):
        """Calculate training metrics"""
        logger.info("Calculating training metrics...")
        
        X_train, y_train = splits['train']
        val_predictions = self.model.predict(X_train)
        
        # Calculate RMSE and MAPE on training data
        train_mape = np.mean(np.abs((y_train - val_predictions) / y_train)) 
        
        return {
            'train_mape': float(train_mape)
        }
    
    def save_artifacts(self, metrics: Dict[str, Any]):
        """Step 3: Save model and metadata"""
        logger.info("=" * 60)
        logger.info("STEP 3: Saving Artifacts")
        logger.info("=" * 60)
        
        # Save model
        self.model.save(str(LSTM_MODEL_PATH))
        
        # Save scaler
        self.data_loader.save_scaler(str(SCALER_PATH))
        
        # Prepare metadata
        scaler_params = self.data_loader.get_scaler_params()
        
        metadata = {
            'ticker': self.ticker,
            'lookback_days': LOOKBACK_DAYS,
            'predict_days': PREDICT_DAYS,
            'years_of_data': self.years,
            'data_points': len(self.data_loader.prices),
            'scaler_min': scaler_params['min'],
            'scaler_max': scaler_params['max'],
            'test_rmse': float(np.sqrt(metrics['test_loss'])),
            'test_mae': metrics['test_mae'],
            'test_mape': metrics['test_mape'],
            'train_mape': metrics.get('train_mape', 0.0),
            'train_rmse': 0.0,
            'model_path': str(LSTM_MODEL_PATH),
            'scaler_path': str(SCALER_PATH),
            'created_at': datetime.now().isoformat(),
            'architecture': {
                'layers': 3,
                'units': [50, 50, 25],
                'dropout': 0.2,
                'activation': 'relu',
                'output_activation': 'linear'
            }
        }
        
        save_metadata(metadata, str(METADATA_PATH))
        
        logger.info("✓ Model artifacts saved")
        logger.info(f"  - Model: {LSTM_MODEL_PATH}")
        logger.info(f"  - Scaler: {SCALER_PATH}")
        logger.info(f"  - Metadata: {METADATA_PATH}\n")
        
        return metadata
    
    def run(self):
        """Run complete training pipeline"""
        logger.info("\n")
        logger.info("╔" + "=" * 58 + "╗")
        logger.info("║" + " " * 10 + "LSTM STOCK PRICE PREDICTION TRAINING" + " " * 11 + "║")
        logger.info("╚" + "=" * 58 + "╝")
        logger.info(f"\nTicker: {self.ticker}")
        logger.info(f"Years: {self.years}")
        logger.info(f"Lookback: {LOOKBACK_DAYS} days")
        logger.info(f"Predict: {PREDICT_DAYS} days\n")
        
        try:
            # Step 1: Load data
            splits = self.load_data()
            
            # Step 2: Build and train
            test_metrics, test_data = self.build_and_train(splits)
            
            # Calculate training metrics
            train_metrics = self.calculate_train_metrics(splits)
            test_metrics.update(train_metrics)
            
            # Step 3: Save artifacts
            metadata = self.save_artifacts(test_metrics)
            
            # Summary
            logger.info("=" * 60)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Test RMSE: ${metadata['test_rmse']:.6f}")
            logger.info(f"Test MAPE: {metadata['test_mape']*100:.2f}%")
            logger.info(f"Test MAE: ${metadata['test_mae']:.6f}")
            logger.info("=" * 60)
            logger.info("\n✓ Training pipeline completed successfully!\n")
            
            return metadata
            
        except Exception as e:
            logger.error(f"✗ Training failed: {e}", exc_info=True)
            raise


def train():
    """Entry point for training"""
    pipeline = TrainingPipeline(ticker=STOCK_TICKER, years=5)
    return pipeline.run()


if __name__ == '__main__':
    train()
