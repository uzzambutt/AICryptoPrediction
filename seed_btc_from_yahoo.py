# seed_btc_from_yahoo.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

os.makedirs("data", exist_ok=True)

print("📥 Downloading BTC-USD 5y from Yahoo Finance...")
df = yf.download("BTC-USD", period="5y", interval="1d", progress=False)
if df.empty:
    raise SystemExit("No data downloaded from Yahoo Finance.")

# prepare Nixtla format
df = df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
df['ds'] = pd.to_datetime(df['ds'])
df['unique_id'] = 'bitcoin'
df = df[['unique_id','ds','y']]

# save canonical latest and timestamped copy
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"data/btc_market_{ts}.csv", index=False)
df.to_csv("data/btc_market.csv", index=False)
print(f"✅ Saved {len(df)} rows to data/btc_market.csv and data/btc_market_{ts}.csv")
