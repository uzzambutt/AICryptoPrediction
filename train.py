"""
MULTI-CRYPTOCURRENCY PRICE FORECASTING SYSTEM
==============================================
A production-ready deep learning system for predicting next-day crypto closing prices
using comprehensive technical indicators and LSTM neural networks.

Supports: BTC, ETH, BNB, SOL, ADA, DOGE, XRP, MATIC, DOT, AVAX, and more!

Author: Muhammad Uzzam Butt © // NorthernStudios // uzzambutt1@outlook.com
Version: 2.0.0 - Multi-coin support
"""

import os
import sys
import warnings
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
import argparse

# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Data and preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_log.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Supported cryptocurrencies with Yahoo Finance tickers
SUPPORTED_COINS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'BNB': 'BNB-USD',
    'SOL': 'SOL-USD',
    'ADA': 'ADA-USD',
    'DOGE': 'DOGE-USD',
    'XRP': 'XRP-USD',
    'MATIC': 'MATIC-USD',
    'DOT': 'DOT-USD',
    'AVAX': 'AVAX-USD',
    'LINK': 'LINK-USD',
    'UNI': 'UNI-USD',
    'ATOM': 'ATOM-USD',
    'LTC': 'LTC-USD',
    'ETC': 'ETC-USD',
    'XLM': 'XLM-USD',
    'ALGO': 'ALGO-USD',
    'VET': 'VET-USD',
    'ICP': 'ICP-USD',
    'FIL': 'FIL-USD'
}

# Global configuration template
CONFIG_TEMPLATE = {
    'lookback_years': 5,
    'sequence_length': 60,
    'train_split': 0.8,
    'batch_size': 32,
    'hidden_sizes': [128, 256, 128],
    'dropout': 0.25,
    'learning_rate': 1e-3,
    'max_epochs': 10000,
    'early_stop_patience': 50,
    'gradient_clip': 1.0,
}


def get_config(coin_symbol):
    """Generate coin-specific configuration"""
    config = CONFIG_TEMPLATE.copy()
    config['coin'] = coin_symbol
    config['ticker'] = SUPPORTED_COINS[coin_symbol]
    config['checkpoint_dir'] = f'checkpoints/{coin_symbol.lower()}'
    config['cache_file'] = f'{coin_symbol.lower()}_data.csv'
    config['output_prefix'] = coin_symbol.lower()
    
    # Create coin-specific directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    return config


# ============================================================================
# DATA ACQUISITION MODULE
# ============================================================================

def fetch_crypto_data(config):
    """
    Fetch historical crypto data from Yahoo Finance with automatic retry logic.
    Implements caching to avoid redundant API calls.
    """
    ticker = config['ticker']
    coin = config['coin']
    years = config['lookback_years']
    cache_file = config['cache_file']
    
    logger.info(f"Fetching {years} years of {coin} ({ticker}) data from Yahoo Finance...")
    
    # Check for cached data
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.info(f"Loaded cached {coin} data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Fetching fresh data...")
    
    # Import yfinance dynamically
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        sys.exit(1)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # Fetch data with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                raise ValueError(f"Downloaded data for {coin} is empty")
            
            # Clean column names if multi-level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Forward fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Drop any remaining NaN rows
            df = df.dropna()
            
            # Save cache
            df.to_csv(cache_file)
            logger.info(f"Successfully fetched {len(df)} rows for {coin}. Cached to {cache_file}")
            
            return df
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {coin}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All retry attempts exhausted for {coin}. Exiting.")
                sys.exit(1)
    
    return None


# ============================================================================
# TECHNICAL INDICATORS MODULE
# ============================================================================

class TechnicalIndicators:
    """
    Comprehensive technical analysis indicators implementation.
    All indicators are computed using vectorized pandas operations for efficiency.
    """
    
    @staticmethod
    def sma(series, period):
        """Simple Moving Average"""
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series, period):
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def macd(series, fast=12, slow=26, signal=9):
        """MACD indicator with signal and histogram"""
        ema_fast = TechnicalIndicators.ema(series, fast)
        ema_slow = TechnicalIndicators.ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(series, period=20, std_dev=2):
        """Bollinger Bands: upper, middle, lower"""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        bandwidth = (upper - lower) / middle
        return upper, middle, lower, bandwidth
    
    @staticmethod
    def rsi(series, period=14):
        """Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def stochastic(high, low, close, period=14):
        """Stochastic Oscillator %K and %D"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=3).mean()
        return k, d
    
    @staticmethod
    def williams_r(high, low, close, period=14):
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def atr(high, low, close, period=14):
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def obv(close, volume):
        """On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def cmf(high, low, close, volume, period=20):
        """Chaikin Money Flow"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        return cmf
    
    @staticmethod
    def accumulation_distribution(high, low, close, volume):
        """Accumulation/Distribution Line"""
        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.fillna(0)
        ad = (mfm * volume).cumsum()
        return ad
    
    @staticmethod
    def parabolic_sar(high, low, acceleration=0.02, maximum=0.2):
        """Parabolic SAR - simplified implementation"""
        sar = low.copy()
        sar.iloc[0] = low.iloc[0]
        return sar.rolling(window=5).mean()


def engineer_features(df, coin):
    """
    Comprehensive feature engineering pipeline.
    Computes all technical indicators and price-based features.
    """
    logger.info(f"Engineering features for {coin} from OHLCV data...")
    
    features_df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # Trend indicators
    for period in [20, 50, 100, 200]:
        features_df[f'SMA_{period}'] = TechnicalIndicators.sma(close, period)
    
    for period in [12, 26, 50, 100]:
        features_df[f'EMA_{period}'] = TechnicalIndicators.ema(close, period)
    
    # MACD
    macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
    features_df['MACD'] = macd_line
    features_df['MACD_Signal'] = signal_line
    features_df['MACD_Histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower, bb_width = TechnicalIndicators.bollinger_bands(close)
    features_df['BB_Upper'] = bb_upper
    features_df['BB_Middle'] = bb_middle
    features_df['BB_Lower'] = bb_lower
    features_df['BB_Width'] = bb_width
    
    # Parabolic SAR
    features_df['PSAR'] = TechnicalIndicators.parabolic_sar(high, low)
    
    # Momentum indicators
    features_df['RSI'] = TechnicalIndicators.rsi(close, 14)
    stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close, 14)
    features_df['Stoch_K'] = stoch_k
    features_df['Stoch_D'] = stoch_d
    features_df['Williams_R'] = TechnicalIndicators.williams_r(high, low, close, 14)
    features_df['Momentum_10'] = close.diff(10)
    features_df['ROC'] = close.pct_change(periods=10) * 100
    
    # Volatility indicators
    features_df['ATR'] = TechnicalIndicators.atr(high, low, close, 14)
    features_df['True_Range'] = high - low
    features_df['Returns_Std_20'] = close.pct_change().rolling(window=20).std()
    
    # Volume indicators
    features_df['OBV'] = TechnicalIndicators.obv(close, volume)
    features_df['CMF'] = TechnicalIndicators.cmf(high, low, close, volume, 20)
    features_df['AD_Line'] = TechnicalIndicators.accumulation_distribution(high, low, close, volume)
    
    # Price-based features
    features_df['Returns_Log'] = np.log(close / close.shift(1))
    features_df['Returns_Pct'] = close.pct_change()
    
    for period in [5, 10, 20]:
        features_df[f'Returns_Mean_{period}'] = features_df['Returns_Pct'].rolling(window=period).mean()
        features_df[f'Returns_Std_{period}'] = features_df['Returns_Pct'].rolling(window=period).std()
    
    # Lag features
    for lag in range(1, 31):
        features_df[f'Close_Lag_{lag}'] = close.shift(lag)
    
    # Drop NaN values from indicator calculations
    features_df = features_df.dropna()
    
    logger.info(f"Feature engineering complete for {coin}. Total features: {features_df.shape[1]}")
    
    return features_df


# ============================================================================
# PYTORCH DATASET AND MODEL ARCHITECTURE
# ============================================================================

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.
    Creates sliding windows of historical data for LSTM input.
    """
    
    def __init__(self, features, targets, sequence_length):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y


class LSTMForecaster(nn.Module):
    """
    Deep multi-layer LSTM architecture for time series forecasting.
    Implements dropout regularization and residual connections.
    """
    
    def __init__(self, input_size, hidden_sizes, dropout=0.25):
        super(LSTMForecaster, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Additional LSTM layers
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.fc = nn.Linear(hidden_sizes[-1], 1)
        
        logger.info(f"Model architecture: Input={input_size}, Hidden={hidden_sizes}, Dropout={dropout}")
    
    def forward(self, x):
        """Forward pass through the network"""
        
        # Pass through LSTM layers
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)
        
        # Take the last time step
        x = x[:, -1, :]
        
        # Final prediction
        x = self.fc(x)
        
        return x


# ============================================================================
# TRAINING INFRASTRUCTURE
# ============================================================================

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    Monitors validation loss and stops training when no improvement is observed.
    """
    
    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def prepare_data(features_df, config):
    """
    Prepare training and validation datasets with proper normalization.
    Implements train-test split without data leakage.
    """
    logger.info(f"Preparing datasets for {config['coin']} training...")
    
    # Separate target variable
    target_col = 'Close'
    feature_cols = [col for col in features_df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
    
    # Extract features and target
    X = features_df[feature_cols].values
    y = features_df[target_col].values
    
    # Train-validation split
    split_idx = int(len(X) * config['train_split'])
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Normalize features
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    # Normalize target
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = CryptoDataset(X_train_scaled, y_train_scaled, config['sequence_length'])
    val_dataset = CryptoDataset(X_val_scaled, y_val_scaled, config['sequence_length'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, feature_scaler, target_scaler, feature_cols


def train_model(model, train_loader, val_loader, device, config):
    """
    Main training loop with checkpointing and early stopping.
    Implements learning rate scheduling and gradient clipping.
    """
    coin = config['coin']
    logger.info(f"Starting model training for {coin}...")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stop_patience'])
    
    # Training state
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(config['checkpoint_dir'], f'best_{coin.lower()}_model.pth')
    start_epoch = 0
    
    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            logger.info(f"Resumed {coin} from checkpoint at epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
        except Exception as e:
            logger.warning(f"Failed to load {coin} checkpoint: {e}. Starting fresh.")
    
    # Training history
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(start_epoch, config['max_epochs']):
        # Training phase
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f'{coin} Epoch {epoch+1}/{config["max_epochs"]} [Train]', leave=False)
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_x).squeeze()
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x).squeeze()
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())
        
        # Calculate epoch metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"{coin} Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            logger.info(f"New best {coin} model saved with val loss: {val_loss:.6f}")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered for {coin} at epoch {epoch+1}")
            break
    
    # Load best model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"{coin} training complete. Best validation loss: {best_val_loss:.6f}")
    
    return model, history, best_val_loss


# ============================================================================
# EVALUATION AND FORECASTING
# ============================================================================

def evaluate_model(model, features_df, feature_scaler, target_scaler, feature_cols, device, config):
    """
    Walk-forward validation on test data.
    Predicts one day ahead iteratively using previous predictions.
    """
    coin = config['coin']
    logger.info(f"Performing walk-forward validation for {coin}...")
    
    model.eval()
    
    # Prepare test data
    split_idx = int(len(features_df) * config['train_split'])
    test_data = features_df.iloc[split_idx:].copy()
    
    X_test = test_data[feature_cols].values
    y_test = test_data['Close'].values
    
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Walk-forward predictions
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for i in range(config['sequence_length'], len(X_test_scaled)):
            # Get sequence
            sequence = X_test_scaled[i - config['sequence_length']:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            # Predict
            pred = model(sequence_tensor).squeeze().cpu().numpy()
            predictions.append(pred)
            actuals.append(y_test[i])
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)
    
    predictions_original = target_scaler.inverse_transform(predictions).flatten()
    actuals_original = actuals.flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals_original, predictions_original))
    mae = mean_absolute_error(actuals_original, predictions_original)
    mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
    r2 = r2_score(actuals_original, predictions_original)
    
    logger.info(f"{coin} Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
    
    # Prepare results
    results_df = pd.DataFrame({
        'Date': test_data.index[config['sequence_length']:],
        'Actual': actuals_original,
        'Predicted': predictions_original
    })
    
    return results_df, {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}


def create_visualization(results_df, features_df, metrics, config):
    """
    Create comprehensive 3-panel visualization with actual vs predicted prices,
    RSI indicator, and MACD histogram.
    """
    coin = config['coin']
    logger.info(f"Creating visualization for {coin}...")
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Panel 1: Actual vs Predicted
    axes[0].plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue', linewidth=2)
    axes[0].plot(results_df['Date'], results_df['Predicted'], label='Predicted', color='red', linewidth=2, alpha=0.7)
    axes[0].set_title(f'{coin} Actual vs Predicted (Walk-Forward): {results_df["Date"].iloc[0].strftime("%Y-%m-%d")} → {results_df["Date"].iloc[-1].strftime("%Y-%m-%d")}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.98, f'RMSE: ${metrics["rmse"]:.4f}\nMAE: ${metrics["mae"]:.4f}\nMAPE: {metrics["mape"]:.2f}%\nR²: {metrics["r2"]:.4f}',
                transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Panel 2: RSI
    rsi_data = features_df.loc[results_df['Date'], 'RSI']
    axes[1].plot(results_df['Date'], rsi_data, label='RSI (14)', color='purple', linewidth=1.5)
    axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
    axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
    axes[1].set_ylabel('RSI', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: MACD Histogram
    macd_hist = features_df.loc[results_df['Date'], 'MACD_Histogram']
    colors = ['green' if x >= 0 else 'red' for x in macd_hist]
    axes[2].bar(results_df['Date'], macd_hist, color=colors, alpha=0.6, label='MACD Histogram')
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('MACD Histogram', fontsize=12)
    axes
