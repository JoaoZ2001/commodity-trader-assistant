# Module responsible for extracting historical soybean, corn and wheat data from Yahoo Finance, and cleaning them in order to use them in other project processes.

# Imports
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.trend import ADXIndicator


# Start and end date
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Function to dowload the data. We will only keep the "close" column
def download_data(ticker, start_date = start_date, end_date = end_date):
    data = yf.Ticker(ticker).history(start=start_date, end=end_date)
    data = data[['Close']]
    return data

print("="*40)
print("📥 Starting data extraction from Yahoo Finance...")
print("="*40)

# Dowloading the data
soybean_data = download_data("ZS=F", start_date, end_date)
corn_data = download_data("ZC=F", start_date, end_date)
wheat_data = download_data("ZW=F", start_date, end_date)

# Convert index to datetime
soybean_data.index = pd.to_datetime(soybean_data.index)
corn_data.index = pd.to_datetime(corn_data.index)
wheat_data.index = pd.to_datetime(wheat_data.index)

# Transform the index into a string in the format 'YYYY/MM/DD'
soybean_data.index = soybean_data.index.map(lambda x: x.strftime('%Y/%m/%d'))
corn_data.index = corn_data.index.map(lambda x: x.strftime('%Y/%m/%d'))
wheat_data.index = wheat_data.index.map(lambda x: x.strftime('%Y/%m/%d'))

# Feature Engineering

# Returns: Calculates the daily percentage change in the closing price.
soybean_data['returns'] = soybean_data.Close.pct_change()
corn_data['returns'] = corn_data.Close.pct_change()
wheat_data['returns'] = wheat_data.Close.pct_change()

# Bollinger Bonds
bb_soybean = BollingerBands(soybean_data['Close'], window=14)
bb_corn = BollingerBands(corn_data['Close'], window=14)
bb_wheat = BollingerBands(wheat_data['Close'], window=14)
soybean_data['bb_high'] = bb_soybean.bollinger_hband()
soybean_data['bb_low'] = bb_soybean.bollinger_lband()
corn_data['bb_high'] = bb_corn.bollinger_hband()
corn_data['bb_low'] = bb_corn.bollinger_lband()
wheat_data['bb_high'] = bb_wheat.bollinger_hband()
wheat_data['bb_low'] = bb_wheat.bollinger_lband()

# MACD
soybean_macd_indicator = MACD(soybean_data['Close'])
soybean_data['macd'] = soybean_macd_indicator.macd()
soybean_data['macd_signal_line'] = soybean_macd_indicator.macd_signal()
soybean_data['macd_diff'] = soybean_macd_indicator.macd_diff()
corn_macd_indicator = MACD(corn_data['Close'])
corn_data['macd'] = corn_macd_indicator.macd()
corn_data['macd_signal_line'] = corn_macd_indicator.macd_signal()
corn_data['macd_diff'] = corn_macd_indicator.macd_diff()
wheat_macd_indicator = MACD(wheat_data['Close'])
wheat_data['macd'] = wheat_macd_indicator.macd()
wheat_data['macd_signal_line'] = wheat_macd_indicator.macd_signal()
wheat_data['macd_diff'] = wheat_macd_indicator.macd_diff()

# RSI
soybean_data['rsi'] = RSIIndicator(soybean_data['Close']).rsi()
corn_data['rsi'] = RSIIndicator(corn_data['Close']).rsi()
wheat_data['rsi'] = RSIIndicator(wheat_data['Close']).rsi()

# SMA
soybean_data['sma'] = SMAIndicator(soybean_data['Close'], window = 14).sma_indicator()
corn_data['sma'] = SMAIndicator(corn_data['Close'], window = 14).sma_indicator()
wheat_data['sma'] = SMAIndicator(wheat_data['Close'], window = 14).sma_indicator()

# EMA
soybean_data['ema'] = EMAIndicator(soybean_data['Close'], window = 14).ema_indicator()
corn_data['ema'] = EMAIndicator(corn_data['Close'], window = 14).ema_indicator()
wheat_data['ema'] = EMAIndicator(wheat_data['Close'], window = 14).ema_indicator()

# ROC
soybean_data['roc'] = soybean_data['Close'].pct_change(periods=14) * 100
corn_data['roc'] = corn_data['Close'].pct_change(periods=14) * 100
wheat_data['roc'] = wheat_data['Close'].pct_change(periods=14) * 100

# Excluding missing values
soybean_data = soybean_data.dropna()
corn_data = corn_data.dropna()
wheat_data = wheat_data.dropna()

# Save
soybean_data.to_csv('/commodities_datasets/soybean_data.csv', index=True)
corn_data.to_csv('/commodities_datasets/corn_data.csv', index=True)
wheat_data.to_csv('/commodities_datasets/wheat_data.csv', index=True)

print("✅ Extraction completed and saved datasets")