import requests
import pandas as pd
import numpy as np
from nixtla import NixtlaClient
from statsforecast.models import NHITS
from statsforecast import StatsForecast
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------------------------
# 1. FETCH MARKET DATA (no CSV needed)
# --------------------------------------------
COIN = "bitcoin"
CURRENCY = "usd"
DAYS = 90

url = f"https://api.coingecko.com/api/v3/coins/{COIN}/market_chart"
params = {"vs_currency": CURRENCY, "days": DAYS}

print(f"📡 Fetching {DAYS} days of {COIN} data from CoinGecko...")
r = requests.get(url, params=params)
r.raise_for_status()
data = r.json()

prices = data["prices"]
df = pd.DataFrame(prices, columns=["timestamp", "price"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df["date"] = df["timestamp"].dt.date

# Daily average (smooths hourly spikes)
df = df.groupby("date")["price"].mean().reset_index()

print(f"✅ Got {len(df)} daily prices from {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

# --------------------------------------------
# 2. PREPARE DATA FOR Nixtla / StatsForecast
# --------------------------------------------
df = df.rename(columns={"date": "ds", "price": "y"})
df["unique_id"] = "BTC-USD"

# Train/test split
train_df = df.iloc[:-7]
test_df = df.iloc[-7:]

# --------------------------------------------
# 3. TRAIN Nixtla NHITS MODEL (CPU)
# --------------------------------------------
print("🧠 Training NHITS model (CPU mode)...")
sf = StatsForecast(
    models=[NHITS(h=7, input_size=30, max_epochs=20, random_seed=42)],
    freq="D",
    n_jobs=1
)
sf.fit(train_df)

# --------------------------------------------
# 4. FORECAST NEXT 7 DAYS
# --------------------------------------------
forecast_df = sf.predict()
print("\n📈 Forecast complete!")
print(forecast_df)

# --------------------------------------------
# 5. MERGE & VISUALIZE
# --------------------------------------------
merged = pd.concat([df.set_index("ds"), forecast_df.set_index("ds")])
merged[["y", "NHITS"]].plot(title="Bitcoin Price Forecast", figsize=(10, 5))
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: save forecast
forecast_df.to_csv("forecast_output.csv", index=False)
print("\n💾 Saved forecast_output.csv")
