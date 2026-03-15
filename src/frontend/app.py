"""
Streamlit Frontend for LSTM Stock Price Prediction
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import sys
sys.path.insert(0, str(__file__).split('src')[0])

from config import API_BASE_URL

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="LSTM Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 LSTM Stock Price Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">Deep Learning Forecasting with FastAPI Backend</p>', unsafe_allow_html=True)


# Helper functions
@st.cache_data(ttl=3600)
def get_api_health():
    """Check API status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=3600)
def get_model_metadata():
    """Get model metadata"""
    try:
        response = requests.get(f"{API_BASE_URL}/metadata", timeout=5)
        return response.json()
    except:
        return None


def make_prediction(ticker: str, days_ahead: int = 30):
    """Make prediction"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"ticker": ticker, "days_ahead": days_ahead},
            timeout=15
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_current_price(ticker: str):
    """Get current price"""
    try:
        response = requests.get(f"{API_BASE_URL}/current/{ticker}", timeout=10)
        return response.json() if response.status_code == 200 else None
    except:
        return None


def get_historical_data(ticker: str, days: int = 180):
    """Get historical data"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/historical",
            json={"ticker": ticker, "days": days},
            timeout=15
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="e.g., AAPL, GOOGL, MSFT").upper()
    days_ahead = st.slider("Days to Predict", 1, 90, 30)
    lookback_days = st.slider("Lookback Window (days)", 30, 365, 180)
    
    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    
    is_healthy = get_api_health()
    if is_healthy:
        st.success("✓ API Backend Running")
        metadata = get_model_metadata()
        if metadata:
            st.info(f"""
**Model Config:**
- Architecture: {metadata['architecture']['layers']}-layer LSTM
- Lookback: {metadata['lookback_days']} days
- Accuracy (MAPE): {metadata['test_mape']*100:.2f}%
            """)
    else:
        st.error("✗ API Backend Offline")


# Main content
if not is_healthy:
    st.markdown("""
    <div class="info-box">
    <h3>⚠️ Backend Not Connected</h3>
    <p>Start the backend with: <code>python scripts/run_backend.py</code></p>
    </div>
    """, unsafe_allow_html=True)
else:
    current_data = get_current_price(ticker)
    
    if current_data:
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 Dashboard", "🎯 Predictions", "📈 Historical", "ℹ️ About"]
        )
        
        # Tab 1: Dashboard
        with tab1:
            st.markdown("### Current Market Data")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(ticker, f"${current_data['current_price']:.2f}")
            with col2:
                st.metric("High", f"${current_data['high']:.2f}")
            with col3:
                st.metric("Low", f"${current_data['low']:.2f}")
            with col4:
                st.metric("Volume", f"{current_data['volume']:,.0f}")
        
        # Tab 2: Predictions
        with tab2:
            st.markdown("### AI Price Prediction")
            
            with st.spinner("Generating prediction..."):
                pred = make_prediction(ticker, days_ahead)
            
            if pred:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Current Price", f"${pred['current_price']:.2f}")
                    st.metric("Predicted Price", f"${pred['predicted_price']:.2f}")
                
                with col2:
                    change = pred['predicted_price'] - pred['current_price']
                    change_pct = (change / pred['current_price']) * 100
                    st.metric("Predicted Change", f"${change:.2f}", f"{change_pct:+.2f}%")
                    st.metric("Confidence", f"{pred['prediction_confidence']:.0%}")
                
                st.markdown("---")
                st.markdown(f"**Prediction Date:** {pred['prediction_date']}")
                st.markdown(f"**Recent Trend:** {pred['recent_trend']}")
                
                # Chart
                dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
                forecast_prices = np.linspace(pred['current_price'], pred['predicted_price'], days_ahead)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[datetime.now()],
                    y=[pred['current_price']],
                    name='Current Price',
                    mode='markers',
                    marker=dict(size=10, color='blue')
                ))
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=forecast_prices,
                    name='Forecast',
                    mode='lines',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                fig.update_layout(
                    title=f"{ticker} {days_ahead}-Day Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tab 3: Historical
        with tab3:
            st.markdown("### Historical Data & Model Predictions")
            
            with st.spinner("Fetching historical data..."):
                hist = get_historical_data(ticker, lookback_days)
            
            if hist:
                df = pd.DataFrame({
                    'Date': hist['dates'],
                    'Price': hist['prices'],
                    'Predicted': hist['predictions']
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Price'],
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['Predicted'],
                    name='Model Prediction',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                fig.update_layout(
                    title=f"{ticker} Historical ({lookback_days} days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Data Table")
                st.dataframe(df.tail(20), use_container_width=True, hide_index=True)
        
        # Tab 4: About
        with tab4:
            st.markdown("""
            ## About This Application
            
            ### Technology Stack
            - **Backend:** FastAPI with Python
            - **ML Model:** LSTM Neural Network (3 layers)
            - **Framework:** TensorFlow/Keras
            - **Frontend:** Streamlit
            - **Data:** Yahoo Finance
            
            ### Model Details
            - **Input:** 180 days of historical prices
            - **Output:** Future price prediction
            - **Architecture:** 3-layer LSTM with dropout
            - **Accuracy:** ~7.6% MAPE
            
            ### How to Use
            1. Select a stock ticker
            2. Adjust prediction horizon
            3. View predictions and trends
            4. Explore historical patterns
            
            ### Disclaimer
            For educational purposes only. Not financial advice.
            """)
    else:
        st.error("Unable to fetch current price data")
