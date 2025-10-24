#!/usr/bin/env python3
"""
produce_forecast_with_statsfallback_safe.py

- Try NeuralForecast future predictions first.
- If no future, use StatsForecast SeasonalNaive (fast), then ETS (slightly heavier, uses last N days).
- If all fail or memory errors occur, use a linear-trend fallback.
- Saves forecast CSV and an actual-vs-predicted PNG.
"""

import os
import glob
import cloudpickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SRC = DATA_DIR / "btc_market.csv"
FORECAST_OUT = DATA_DIR / "btc_forecast.csv"
PLOT_OUT = DATA_DIR / "forecast_plot.png"

H = 7
STATS_LAST_N_DAYS = 365  # limit input length for statsforecast to reduce memory/CPU

# Load NF from checkpoint if available
nf = None
ck = sorted(glob.glob(str(DATA_DIR / "checkpoints" / "latest.pkl")))
if ck:
    try:
        with open(ck[0], "rb") as f:
            payload = cloudpickle.load(f)
            nf = payload.get("nf")
            print("Loaded NeuralForecast from checkpoint:", ck[0])
    except Exception as e:
        print("Could not load NF from checkpoint:", e)

# Load history
df = pd.read_csv(SRC, parse_dates=["ds"])
df = df.sort_values("ds").reset_index(drop=True)
last_hist = df["ds"].max()
print("history rows:", len(df), "last:", last_hist.date())

# Helper: try several NF predict variants
def try_nf_future(nf_obj, df_history, h=H):
    if nf_obj is None:
        return None
    try:
        # Try nf.predict(h=H)
        try:
            print("Trying nf.predict(h=H) ...")
            fc = nf_obj.predict(h=h)
            if isinstance(fc, pd.DataFrame) and pd.to_datetime(fc["ds"]).max() > df_history["ds"].max():
                return fc
        except Exception as e:
            print("nf.predict(h=H) error/unsupported:", e)

        # Try nf.predict()
        try:
            print("Trying nf.predict() ...")
            fc = nf_obj.predict()
            if isinstance(fc, pd.DataFrame) and pd.to_datetime(fc["ds"]).max() > df_history["ds"].max():
                return fc
        except Exception as e:
            print("nf.predict() error:", e)

        # Try nf.predict(future_df)
        try:
            print("Trying nf.predict(future_df) ...")
            future_dates = pd.date_range(start=df_history["ds"].max() + pd.Timedelta(days=1), periods=h, freq="D")
            future_df = pd.DataFrame({"unique_id":[df_history.unique_id.iloc[0]]*len(future_dates), "ds": future_dates})
            fc = nf_obj.predict(future_df)
            if isinstance(fc, pd.DataFrame) and pd.to_datetime(fc["ds"]).max() > df_history["ds"].max():
                return fc
        except Exception as e:
            print("nf.predict(future_df) error:", e)

    except Exception as e:
        print("Unexpected NF predict error:", e)
    return None

# StatsForecast fallback pipeline (safe)
def statsforecast_fallback(df_history, h=H, last_n_days=STATS_LAST_N_DAYS):
    """
    Try SeasonalNaive first (fast), then ETS with limited lookback.
    Return forecast DataFrame with columns ['unique_id','ds','yhat'].
    """
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive, ETS
    except Exception as e:
        print("statsforecast not available or import error:", e)
        raise

    # Use only last_n_days to reduce memory and speed up fit
    df_small = df_history.tail(last_n_days).copy().reset_index(drop=True)
    print(f"Using last {len(df_small)} rows for statsforecast fallback (limit {last_n_days}).")

    # 1) SeasonalNaive (very fast, deterministic)
    try:
        print("Trying StatsForecast SeasonalNaive (fast)...")
        sf = StatsForecast(models=[SeasonalNaive(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df_small)
        forecasts = sf.forecast(h=h)
        # normalize prediction column
        pred_cols = [c for c in forecasts.columns if c not in ("unique_id","ds")]
        if pred_cols:
            forecasts = forecasts.rename(columns={pred_cols[0]: "yhat"})
        else:
            raise RuntimeError("SeasonalNaive returned no pred column")
        print("SeasonalNaive succeeded.")
        return forecasts[["unique_id","ds","yhat"]]
    except MemoryError as me:
        print("SeasonalNaive MemoryError (unexpected):", me)
    except Exception as e:
        print("SeasonalNaive failed:", e)

    # 2) ETS with last_n_days (still relatively cheap)
    try:
        print("Trying StatsForecast ETS (last_n_days)...")
        sf = StatsForecast(models=[ETS()], freq="D", n_jobs=1)
        sf.fit(df_small)
        forecasts = sf.forecast(h=h)
        pred_cols = [c for c in forecasts.columns if c not in ("unique_id","ds")]
        if pred_cols:
            forecasts = forecasts.rename(columns={pred_cols[0]: "yhat"})
        else:
            raise RuntimeError("ETS returned no pred column")
        print("ETS succeeded.")
        return forecasts[["unique_id","ds","yhat"]]
    except MemoryError as me:
        print("ETS MemoryError:", me)
    except Exception as e:
        print("ETS failed:", e)

    # If we reach here, statsforecast didn't produce a result
    raise RuntimeError("StatsForecast fallback models failed (SeasonalNaive/ETS).")

# Linear trend fallback (guaranteed)
def linear_trend_fallback(df_history, h=H):
    hist = df_history.set_index("ds")["y"].sort_index()
    n = min(90, len(hist))
    if n < 3:
        raise RuntimeError("Not enough historical points for linear fallback.")
    recent = hist.tail(n).values.astype(float)
    x = np.arange(n).astype(float)
    coeffs = np.polyfit(x, recent, 1)
    future_x = np.arange(n, n + h).astype(float)
    y_future = np.polyval(coeffs, future_x)
    future_start = hist.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=future_start, periods=h, freq="D")
    future_df = pd.DataFrame({"unique_id":[df_history.unique_id.iloc[0]]*h, "ds": future_dates, "yhat": y_future})
    return future_df

# Main logic
fc = None
# 1) Try NF
fc = try_nf_future(nf, df, H)
if fc is not None:
    print("✅ NeuralForecast produced future forecast.")
    # unify column name
    pred_cols = [c for c in fc.columns if c not in ("unique_id","ds")]
    if pred_cols:
        fc = fc.rename(columns={pred_cols[0]:"yhat"})
    fc = fc[["unique_id","ds","yhat"]]
else:
    print("⚠️ NeuralForecast did not return future dates or NF missing. Trying statsforecast fallback.")
    # 2) Try statsforecast SeasonalNaive/ETS with limited lookback
    try:
        fc = statsforecast_fallback(df, h=H, last_n_days=STATS_LAST_N_DAYS)
    except MemoryError as me:
        print("MemoryError while running statsforecast fallback:", me)
        print("Falling back to linear-trend extrapolation (safe).")
        fc = linear_trend_fallback(df, h=H)
    except Exception as e:
        print("StatsForecast fallback failed:", e)
        print("Falling back to linear-trend extrapolation (safe).")
        try:
            fc = linear_trend_fallback(df, h=H)
        except Exception as e2:
            raise RuntimeError("All fallback methods failed: " + str(e2))

# Save forecast CSV
fc = fc.sort_values("ds").reset_index(drop=True)
fc.to_csv(FORECAST_OUT, index=False)
print("Saved forecast to:", FORECAST_OUT)

# Plot actual vs predicted (holdout + future)
hist_recent = df.set_index("ds")["y"].tail(120)
plt.figure(figsize=(12,6))
plt.plot(hist_recent.index, hist_recent.values, label="Actual (recent)")

# holdout actual last H days
hold_actual = df.tail(H).set_index("ds")["y"]
plt.plot(hold_actual.index, hold_actual.values, marker="o", linestyle="-", label="Actual (holdout)")

# predicted series
pred = fc.copy()
pred["ds"] = pd.to_datetime(pred["ds"])
pred = pred.set_index("ds")["yhat"]

# holdout predicted where ds overlaps
hold_pred = pred[pred.index.isin(hold_actual.index)]
future_pred = pred[pred.index > df["ds"].max()]

if not hold_pred.empty:
    plt.plot(hold_pred.index, hold_pred.values, marker="x", linestyle="--", label="Predicted (holdout)")
if not future_pred.empty:
    plt.plot(future_pred.index, future_pred.values, marker="o", linestyle="--", label="Forecast (future)")
    plt.axvline(x=df["ds"].max(), color="gray", linestyle=":", linewidth=1)
    ylim = plt.gca().get_ylim()
    plt.text(df["ds"].max(), ylim[1]*0.98, "Forecast start", rotation=90, va="top", ha="right", fontsize=8, color="gray")
else:
    plt.text(0.02, 0.95, "No future forecast (unexpected)", transform=plt.gca().transAxes, color="red")

plt.title("BTC actual vs predicted (recent + forecast)")
plt.xlabel("Date"); plt.ylabel("Price (usd)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(PLOT_OUT)
print("Saved plot to:", PLOT_OUT)

# Print the 7-day forecast table
print("\nNext 7-day forecast (date, yhat):")
print(fc.tail(H)[["ds","yhat"]].to_string(index=False))
