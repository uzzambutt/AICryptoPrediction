#!/usr/bin/env python3
"""
auto_train_btc_forecast.py

✅ Fetches BTC data (5 years)
✅ Handles rate limits (CoinGecko or cached)
✅ Trains NHITS model with checkpointing (resume-able)
✅ Forecasts next 7 days
✅ Plots actual vs predicted prices on same graph
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE
import cloudpickle

# ==== CONFIG ====
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
VS_CURRENCY = "usd"
DAYS = 1825  # 5 years
FORECAST_HORIZON = 7
INPUT_SIZE = 365
MAX_STEPS = 50000
CHUNK_SIZE = 1500  # training chunks to checkpoint more often

# ==== FUNCTIONS ====

def fetch_btc_data():
    """Fetch BTC price data from CoinGecko with fallback to cache."""
    backup_path = DATA_DIR / f"btc_market_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    target_path = DATA_DIR / "btc_market.csv"
    url = f"{COINGECKO_URL}?vs_currency={VS_CURRENCY}&days={DAYS}"
    retries, delay = 4, 8
    for i in range(retries):
        try:
            print("📊 Fetching BTC data from CoinGecko...")
            resp = requests.get(url, timeout=20)
            if resp.status_code == 429:
                print(f"⚠️ Rate limited (429). Waiting {delay}s and retrying... ({i+1}/{retries})")
                time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            prices = data.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["y"] = df["price"]
            df["unique_id"] = "bitcoin"
            df = df[["unique_id", "ds", "y"]]
            df.to_csv(target_path, index=False)
            df.to_csv(backup_path, index=False)
            print(f"✅ Saved {len(df)} fresh rows to {target_path.name}")
            return df
        except requests.RequestException as e:
            print(f"❌ Request error (attempt {i+1}/{retries}): {e}")
            time.sleep(delay)
    # fallback
    files = sorted(DATA_DIR.glob("btc_market_backup_*.csv"))
    if files:
        latest = files[-1]
        print(f"⚠️ API fetch failed after retries. Attempting to use cached data from {latest.name}...")
        df = pd.read_csv(latest, parse_dates=["ds"])
        print(f"✅ Using cached dataset instead ({len(df)} rows).")
        df.to_csv(target_path, index=False)
        return df
    raise RuntimeError("No API data and no cached file available.")


def save_checkpoint(nf, step_done):
    ck_path = CHECKPOINT_DIR / "latest.pkl"
    with open(ck_path, "wb") as f:
        cloudpickle.dump({"nf": nf, "step_done": step_done}, f)
    print(f"💾 Checkpoint saved at {ck_path} (steps_done={step_done})")


def load_checkpoint():
    ck_path = CHECKPOINT_DIR / "latest.pkl"
    if ck_path.exists():
        with open(ck_path, "rb") as f:
            obj = cloudpickle.load(f)
            print("✅ Checkpoint loaded successfully.")
            return obj
    return None


def plot_forecast(df, forecast_df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Actual Price", color="blue")
    plt.plot(forecast_df["ds"], forecast_df["yhat"], label="Predicted / Forecasted Price", color="green")
    plt.axvline(df["ds"].max(), color="gray", linestyle="--", label="Forecast Start")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (USD)")
    plt.title("BTC Actual vs Forecasted Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "btc_forecast_plot.png")
    print(f"📈 Forecast plot saved to btc_forecast_plot.png")


# ==== MAIN LOGIC ====

print(f"⚠️ WARNING: max_steps = {MAX_STEPS} — this can take a very long time on CPU.\n")
print(f"=== 🚀 Auto-trainer cycle started at {datetime.utcnow().isoformat()} UTC ===")

try:
    df = fetch_btc_data()
except Exception as e:
    print(f"❌ Error fetching data: {e}")
    exit(1)

df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["y"])
df = df.sort_values("ds").reset_index(drop=True)

print(f"Data ready: n_train={len(df)}, input_size={INPUT_SIZE}")

checkpoint = load_checkpoint()
if checkpoint:
    nf = checkpoint["nf"]
    steps_done = checkpoint.get("step_done", 0)
    print(f"Resuming from checkpoint (steps_done={steps_done})")
else:
    steps_done = 0
    print("🆕 Starting new NHITS model...")
    nf = NeuralForecast(
        models=[
            NHITS(
                input_size=INPUT_SIZE,
                h=FORECAST_HORIZON,
                loss=MAE(),
                max_steps=CHUNK_SIZE,
                val_check_steps=100,
                scaler_type="robust",
                learning_rate=1e-3,
                num_layers=2,
                hidden_size=256,
                random_seed=42,
            )
        ],
        freq="D",
    )

# Train in chunks until total steps reached
try:
    while steps_done < MAX_STEPS:
        print(f"\n=== Chunk start — steps_done={steps_done}, this_chunk={CHUNK_SIZE}, remaining={MAX_STEPS - steps_done} ===")
        nf.fit(df)
        steps_done += CHUNK_SIZE
        save_checkpoint(nf, steps_done)
        print(f"✅ Finished chunk — total steps_done={steps_done}")
        if steps_done >= MAX_STEPS:
            break
        print("💤 Sleeping 5 seconds before next chunk...")
        time.sleep(5)
except KeyboardInterrupt:
    print("\n🛑 Interrupted manually, saving checkpoint...")
    save_checkpoint(nf, steps_done)
    exit(0)
except Exception as e:
    print(f"❌ Error during training: {e}")
    save_checkpoint(nf, steps_done)
    exit(1)

# ==== FORECAST ====
print("\n🔮 Generating 7-day future forecast...")
try:
    # Prepare future df for next 7 days
    last_date = df["ds"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=FORECAST_HORIZON, freq="D")
    future_df = pd.DataFrame({
        "unique_id": ["bitcoin"] * FORECAST_HORIZON,
        "ds": future_dates
    })

    forecast_df = nf.predict(future_df)
    forecast_df = forecast_df.rename(columns={forecast_df.columns[-1]: "yhat"})
    forecast_df.to_csv(DATA_DIR / "btc_forecast.csv", index=False)
    print(f"✅ Forecast saved to btc_forecast.csv ({len(forecast_df)} rows)")
    plot_forecast(df, forecast_df)

except Exception as e:
    print(f"❌ Forecast error: {e}")

print("\n✅ All done.")
