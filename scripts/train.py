#!/usr/bin/env python
"""
Entry point for training LSTM model
Usage: python scripts/train.py
"""

import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.training.train import train

if __name__ == '__main__':
    try:
        metadata = train()
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)
