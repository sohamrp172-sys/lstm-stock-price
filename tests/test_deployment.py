#!/usr/bin/env python
"""
Test script to verify full deployment
Usage: python tests/test_deployment.py
"""

import requests
import sys

def test_fastapi():
    """Test FastAPI backend"""
    print("=" * 60)
    print("Testing FastAPI Backend")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ FastAPI Root: {response.status_code}")
            print(f"  Status: {data['status']}")
            print(f"  Model: {data['model']['type']}")
            print(f"  Test MAPE: {data['model']['test_mape']}")
            return True
        else:
            print(f"✗ FastAPI Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ FastAPI Connection Error: {e}")
        return False


def test_streamlit():
    """Test Streamlit frontend"""
    print("\n" + "=" * 60)
    print("Testing Streamlit Frontend")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8501/', timeout=5)
        print(f"✓ Streamlit: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ Streamlit Error: {e}")
        return False


def test_prediction():
    """Test prediction endpoint"""
    print("\n" + "=" * 60)
    print("Testing Prediction Endpoint")
    print("=" * 60)
    
    try:
        response = requests.post(
            'http://localhost:8000/predict', 
            json={'ticker': 'AAPL', 'days_ahead': 30}, 
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction Success")
            print(f"  Ticker: {result['ticker']}")
            print(f"  Current Price: ${result['current_price']:.2f}")
            print(f"  Predicted Price: ${result['predicted_price']:.2f}")
            print(f"  Confidence: {result['prediction_confidence']:.1%}")
            return True
        else:
            print(f"✗ Prediction Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Prediction Error: {e}")
        return False


def test_current_price():
    """Test current price endpoint"""
    print("\n" + "=" * 60)
    print("Testing Current Price Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/current/AAPL', timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Current Price Success")
            print(f"  Ticker: {result['ticker']}")
            print(f"  Price: ${result['current_price']:.2f}")
            print(f"  High: ${result['high']:.2f}")
            print(f"  Low: ${result['low']:.2f}")
            return True
        else:
            print(f"✗ Current Price Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Current Price Error: {e}")
        return False


def main():
    """Run all tests"""
    results = {
        'FastAPI': test_fastapi(),
        'Streamlit': test_streamlit(),
        'Current Price': test_current_price(),
        'Prediction': test_prediction(),
    }
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ ALL TESTS PASSED - Deployment is working!")
        print("  Streamlit at: http://localhost:8501")
        print("  API Docs at: http://localhost:8000/docs")
    else:
        print("✗ SOME TESTS FAILED - Check the errors above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
