#!/usr/bin/env python
"""
Entry point for FastAPI backend
Usage: python scripts/run_backend.py
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

from src.backend.api import run_server

if __name__ == '__main__':
    try:
        print("\n" + "=" * 60)
        print("Starting FastAPI Backend Server...")
        print("=" * 60)
        print("API running at: http://localhost:8000")
        print("API docs at: http://localhost:8000/docs")
        print("=" * 60 + "\n")
        run_server()
    except Exception as e:
        print(f"Failed to start backend: {e}")
        sys.exit(1)
