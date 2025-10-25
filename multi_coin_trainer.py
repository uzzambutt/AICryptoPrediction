#!/usr/bin/env python3
"""
deep_multi_coin_trainer.py (single-checkpoint per iteration + indicator plots + unscaled train MAE)

Overwrite your existing file with this one.

Outputs:
 - data_deep/cache/<coin>_YYYY-MM-DD_YYYY-MM-DD.csv    (cached chunks)
 - data_deep/models/<coin>_iter{iter}.pt              (single checkpoint file per iteration, overwritten each epoch)
 - data_deep/plots/<coin>_chunk_iter{iter}.png        (actual vs predicted + RSI + MACD hist)
 - data_deep/plots/<coin>_future_overlay_iter{iter}.png
 - data_deep/<coin>_forecast_iter{iter}.csv
 - data_deep/train_log.csv
"""
import os
import math
import time
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import requests

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
COINS = ["litecoin"]          # Example; change as needed
DATA_DIR = Path("data_deep")
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = DATA_DIR / "models"
PLOTS_DIR = DATA_DIR / "plots"
LOG_FILE = DATA_DIR / "train_log.csv"
for d in (DATA_DIR, CACHE_DIR, MODEL_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DAYS_PER_CHUNK = 365
YEARS_TO_FETCH = 5
COINGECKO_PUBLIC_BASE = "https://api.coingecko.com/api/v3/coins/{id}/market_chart/range"
COINGECKO_PRO_BASE = "https://pro-api.coingecko.com/api/v3/coins/{id}/market_chart/range"

# model/training
INPUT_SIZE = 210
H = 30
FEATURE_LAGS = 30
BATCH_SIZE = 16
EPOCHS = 3500        # you used big numbers earlier; set as needed
LR = 5e-4
WEIGHT_DECAY = 1e-6
PATIENCE = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model params
HIDDEN_SIZE = 96
NUM_LAYERS = 4
DROPOUT = 0.25

# behavior
TRAIN_MIN_ROWS = 120
MIN_VAL_SPLIT = 0.1
MAX_ITER_NO_IMPROVE = 3
FORCE_FULL_TRAIN = False

# trade thresholds
BUY_RETURN_THRESHOLD = 0.03
SELL_RETURN_THRESHOLD = -0.03
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# fetch retries/backoff
FETCH_MAX_RETRIES = 1
FETCH_BACKOFF_BASE = 8

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
# ----------------------------------------

# ---------- util / fetch ----------
def ts_to_unix(dt: datetime) -> int:
    return int(dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())

def cache_filename(coin_id: str, start_date: datetime.date, end_date: datetime.date) -> Path:
    return CACHE_DIR / f"{coin_id}_{start_date.isoformat()}_{end_date.isoformat()}.csv"

def read_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)

def _choose_base_url_and_params():
    api_key = os.getenv("COINGECKO_API_KEY", "").strip()
    if api_key:
        return COINGECKO_PRO_BASE, {"x_cg_pro_api_key": api_key}
    return COINGECKO_PUBLIC_BASE, {}

def fetch_range_api(coin_id: str, from_date: datetime.date, to_date: datetime.date, max_retries=FETCH_MAX_RETRIES) -> Optional[pd.DataFrame]:
    from_ts = ts_to_unix(datetime(from_date.year, from_date.month, from_date.day))
    to_ts = ts_to_unix(datetime(to_date.year, to_date.month, to_date.day) + timedelta(days=1))
    base_url, base_params = _choose_base_url_and_params()
    params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
    params.update(base_params)
    attempt = 0; backoff = FETCH_BACKOFF_BASE
    while attempt < max_retries:
        attempt += 1
        try:
            url = base_url.format(id=coin_id)
            print(f"📡 API fetch attempt {attempt}: {url} ({from_date} -> {to_date})")
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 401 and base_url is COINGECKO_PRO_BASE:
                print("⚠️ PRO 401 — switching to public endpoint")
                base_url = COINGECKO_PUBLIC_BASE
                params = {"vs_currency": "usd", "from": from_ts, "to": to_ts}
                r = requests.get(base_url.format(id=coin_id), params=params, timeout=30)
            if r.status_code == 429:
                print(f"⚠️ Rate limited (429). Waiting {backoff}s and retrying... ({attempt}/{max_retries})")
                time.sleep(backoff); backoff *= 2; continue
            r.raise_for_status()
            payload = r.json()
            prices = payload.get("prices", [])
            volumes = payload.get("total_volumes", [])
            if not prices:
                print("⚠️ API returned no price points.")
                return None
            dfp = pd.DataFrame(prices, columns=["ts", "price"])
            dfv = pd.DataFrame(volumes, columns=["ts", "volume"])
            df = pd.merge(dfp, dfv, on="ts", how="left")
            df["ds"] = pd.to_datetime(df["ts"], unit="ms")
            df = df.set_index("ds").resample("D").agg({"price":"mean","volume":"sum"}).reset_index()
            df = df.rename(columns={"price":"y"})
            df["unique_id"] = coin_id
            df = df[["unique_id","ds","y","volume"]].sort_values("ds").reset_index(drop=True)
            return df
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 429:
                print(f"⚠️ HTTP 429 — backoff {backoff}s.")
                time.sleep(backoff); backoff *= 2; continue
            print("⚠️ HTTP error:", e); return None
        except Exception as e:
            print("⚠️ Fetch error:", e); time.sleep(backoff); backoff *= 2; continue
    print("⚠️ Exhausted fetch retries"); return None

def ensure_chunk_cached(coin_id: str, start, end) -> pd.DataFrame:
    fn = cache_filename(coin_id, start, end)
    if fn.exists():
        print(f"🔁 Exact cached chunk found: {fn.name}")
        return read_cache(fn)
    cached = sorted(CACHE_DIR.glob(f"{coin_id}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cached:
        parts=[]
        for p in cached:
            try: parts.append(read_cache(p))
            except: pass
        if parts:
            combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            min_c = combined["ds"].min().date(); max_c = combined["ds"].max().date()
            if min_c <= start and max_c >= end:
                slice_df = combined[(combined["ds"].dt.date >= start) & (combined["ds"].dt.date <= end)].reset_index(drop=True)
                slice_df.to_csv(fn,index=False); print(f"🔁 Created slice {fn.name} from cache"); return slice_df
            to_fetch=[]
            if min_c > start: to_fetch.append((start, min_c - timedelta(days=1)))
            if max_c < end: to_fetch.append((max_c + timedelta(days=1), end))
            fetched=[]
            for a,b in to_fetch:
                d = fetch_range_api(coin_id, a, b)
                if d is not None: fetched.append(d)
                else: print(f"⚠️ Could not fetch {a}->{b}")
            merged = pd.concat([combined] + fetched, ignore_index=True).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)
            merged_start = merged["ds"].min().date(); merged_end = merged["ds"].max().date()
            merged_fn = cache_filename(coin_id, merged_start, merged_end); merged.to_csv(merged_fn,index=False); print(f"💾 Merged cache saved: {merged_fn.name}")
            if merged_start <= start and merged_end >= end:
                chunk_df = merged[(merged["ds"].dt.date >= start) & (merged["ds"].dt.date <= end)].reset_index(drop=True)
                chunk_df.to_csv(fn,index=False); print(f"🔁 Returning slice cached as {fn.name}"); return chunk_df
            chunk_df = merged[(merged["ds"].dt.date >= start) & (merged["ds"].dt.date <= end)].reset_index(drop=True)
            if not chunk_df.empty:
                chunk_df.to_csv(fn,index=False); print("⚠️ Partial coverage returned and cached"); return chunk_df
    fetched_full = fetch_range_api(coin_id, start, end)
    if fetched_full is None:
        if cached:
            print("⚠️ Fetch failed; using most recent cached file as fallback.")
            try:
                fallback = read_cache(cached[0]); fallback.to_csv(fn,index=False); return fallback
            except: pass
        raise RuntimeError("Could not fetch data and no usable cache.")
    fetched_full.to_csv(fn,index=False); print(f"💾 Cached fetched chunk: {fn.name}"); return fetched_full

# ---------- indicators & features ----------
def SMA(s,w): return s.rolling(window=w,min_periods=1).mean()
def EMA(s,span): return s.ewm(span=span,adjust=False).mean()
def MACD(s,fast=12,slow=26,sig=9):
    f=EMA(s,fast); sl=EMA(s,slow); macd=f-sl; sigv=EMA(macd,sig); hist=macd-sigv; return macd,sigv,hist
def RSI(s,length=14):
    d = s.diff(); up = d.clip(lower=0); down = -d.clip(upper=0)
    ma_up = up.ewm(alpha=1/length,adjust=False).mean(); ma_down = down.ewm(alpha=1/length,adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12); return 100 - (100/(1+rs))
def bollinger(s,window=20,n_std=2):
    sma = SMA(s,window); std = s.rolling(window=window,min_periods=1).std(); return sma + n_std*std, sma - n_std*std

def make_features(df: pd.DataFrame, lags=FEATURE_LAGS) -> pd.DataFrame:
    df = df.sort_values("ds").reset_index(drop=True).copy()
    s = df["y"]
    df["ret_1"] = s.pct_change(1); df["ret_7"] = s.pct_change(7); df["logy"] = np.log1p(s)
    df["sma_7"] = SMA(s,7); df["sma_21"] = SMA(s,21); df["ema_12"] = EMA(s,12); df["ema_26"] = EMA(s,26)
    macd, macd_sig, macd_hist = MACD(s); df["macd"]=macd; df["macd_sig"]=macd_sig; df["macd_hist"]=macd_hist
    df["rsi_14"] = RSI(s,14)
    up, lo = bollinger(s,20,2); df["bb_upper"]=up; df["bb_lower"]=lo; df["bb_width"] = (up-lo)/(SMA(s,21)+1e-12)
    df["vol_7"] = df["volume"].rolling(7,min_periods=1).mean(); df["vol_21"] = df["volume"].rolling(21,min_periods=1).mean()
    for lag in range(1, lags+1):
        df[f"lag_{lag}"] = df["y"].shift(lag); df[f"ret_lag_{lag}"] = df["ret_1"].shift(lag)
    df = df.dropna().reset_index(drop=True); return df

# ---------- Dataset & model ----------
class SeqDataset(Dataset):
    def __init__(self, df_feat: pd.DataFrame, feature_cols: List[str], input_size: int, h:int):
        self.df = df_feat.reset_index(drop=True); self.feature_cols = feature_cols
        self.input_size = input_size; self.h = h; self.X=[]; self.y=[]
        N = len(self.df)
        for i in range(0, N - input_size - h + 1):
            xw = self.df.loc[i:i+input_size-1, feature_cols].values.astype(np.float32)
            yw = self.df.loc[i+input_size:i+input_size+h-1, "y"].values.astype(np.float32)
            self.X.append(xw); self.y.append(yw)
        self.X = np.stack(self.X) if self.X else np.empty((0,input_size,len(feature_cols)),dtype=np.float32)
        self.y = np.stack(self.y) if self.y else np.empty((0,h),dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class LSTMForecast(nn.Module):
    def __init__(self, n_features:int, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, h=H):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, max(8,hidden_size//2)), nn.ReLU(), nn.Dropout(dropout), nn.Linear(max(8,hidden_size//2), h))
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

# ---------- training helpers ----------
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0; n=0; preds=[]; trues=[]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb); loss = criterion(pred, yb)
            total_loss += float(loss.item()) * xb.size(0); n += xb.size(0)
            preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
    if preds:
        preds = np.vstack(preds); trues = np.vstack(trues)
    else:
        preds = np.empty((0,H)); trues = np.empty((0,H))
    return (total_loss / n) if n>0 else None, preds, trues

def save_checkpoint_single(path: Path, model, optimizer, epoch:int, scaler_x, scaler_y, meta:dict):
    payload = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict() if optimizer is not None else None, "epoch": int(epoch), "scaler_x": scaler_x, "scaler_y": scaler_y, "meta": meta}
    torch.save(payload, str(path))
    # single file only (overwrites)
    print(f"💾 Checkpoint saved (single file): {path.name}")

def load_checkpoint(path: Path, model, optimizer=None, device=DEVICE):
    d = torch.load(str(path), map_location=device)
    model.load_state_dict(d["model_state"])
    if optimizer is not None and d.get("optim_state") is not None:
        try: optimizer.load_state_dict(d["optim_state"])
        except: pass
    return d

def iterative_forecast(df_feat: pd.DataFrame, feature_cols: List[str], model: nn.Module, scaler_x: StandardScaler, scaler_y: StandardScaler, input_size:int=INPUT_SIZE, h:int=H):
    df = df_feat.sort_values("ds").reset_index(drop=True).copy()
    ts = list(df["ds"]); y_series = list(df["y"].values); vol_series = list(df["volume"].values)
    model.eval(); preds=[]
    for step in range(h):
        temp_df = pd.DataFrame({"ds": ts, "y": y_series, "volume": vol_series})
        feat = make_features(temp_df, lags=FEATURE_LAGS)
        if len(feat) < input_size:
            base_x = np.zeros((input_size, len(feature_cols)), dtype=np.float32)
        else:
            last_window = feat.tail(input_size)
            base_x = last_window[feature_cols].values.astype(np.float32)
        x_flat = base_x.reshape(-1, len(feature_cols))
        x_scaled = scaler_x.transform(x_flat).reshape(1, input_size, len(feature_cols)).astype(np.float32)
        xb = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = model(xb).cpu().numpy().reshape(-1)
        first_scaled = out[0]
        inv = scaler_y.inverse_transform(np.array(first_scaled).reshape(-1,1)).reshape(-1)[0]
        next_date = pd.to_datetime(df["ds"].max()) + pd.Timedelta(days=step+1)
        preds.append((next_date, inv))
        ts.append(next_date); y_series.append(inv); vol_series.append(vol_series[-1] if len(vol_series)>0 else 0.0)
    out_df = pd.DataFrame(preds, columns=["ds","yhat"]); out_df["unique_id"]=df_feat["unique_id"].iloc[0]; return out_df[["unique_id","ds","yhat"]]

def suggest_trades(future_forecast: pd.DataFrame, last_indicators_row: pd.Series):
    last_price = float(last_indicators_row["y"]); last_fc_price = float(future_forecast["yhat"].iloc[-1])
    exp_ret = (last_fc_price - last_price) / (last_price + 1e-12)
    last_rsi = float(last_indicators_row.get("rsi_14", 50.0))
    rec = {"expected_return": float(exp_ret), "rsi": last_rsi, "recommendation":"HOLD", "reason":""}
    if exp_ret >= BUY_RETURN_THRESHOLD and last_rsi <= RSI_OVERSOLD:
        rec.update({"recommendation":"BUY", "reason":f"Expected {exp_ret:.3%} & RSI {last_rsi:.1f}"})
    elif exp_ret <= SELL_RETURN_THRESHOLD or (last_rsi >= RSI_OVERBOUGHT and exp_ret < 0.01):
        rec.update({"recommendation":"SELL", "reason":f"Expected {exp_ret:.3%} or RSI {last_rsi:.1f}"})
    else:
        rec.update({"recommendation":"HOLD", "reason":f"Expected {exp_ret:.3%}; RSI {last_rsi:.1f}"})
    return rec

# ---------- main pipeline ----------
def run_coin(coin_id: str):
    print(f"\n=== STARTING DEEP-TRAIN PIPELINE for {coin_id} ===")
    today = datetime.utcnow().date(); fetch_end = today; fetch_start = fetch_end - timedelta(days=DAYS_PER_CHUNK - 1)
    cumulative = None; best_val = math.inf; no_improve = 0

    for iter_i in range(YEARS_TO_FETCH):
        print(f"\n--- Iteration {iter_i+1} (chunk {fetch_start} -> {fetch_end}) ---")
        try:
            chunk = ensure_chunk_cached(coin_id, fetch_start, fetch_end)
        except Exception as e:
            print("Fetch failed:", e); break

        if cumulative is None: cumulative = chunk.copy()
        else:
            cumulative = pd.concat([chunk, cumulative], ignore_index=True).drop_duplicates(subset=["ds"]).sort_values("ds").reset_index(drop=True)

        print(f"📚 cumulative rows: {len(cumulative)} (chunk rows: {len(chunk)})")
        feat = make_features(cumulative, lags=FEATURE_LAGS)
        if len(feat) < TRAIN_MIN_ROWS:
            print("Not enough rows for training yet; sliding window insufficient. Move to previous chunk.")
            fetch_end = fetch_start - timedelta(days=1); fetch_start = fetch_end - timedelta(days=DAYS_PER_CHUNK - 1); continue

        feature_cols = [c for c in feat.columns if c not in ("unique_id","ds","y","volume")]
        feature_cols = sorted(feature_cols, key=lambda x: (not x.startswith("lag_"), x))
        print(f"Using {len(feature_cols)} features (sample): {feature_cols[:8]}{'...' if len(feature_cols)>8 else ''}")

        N = len(feat); val_size = max(int(N * MIN_VAL_SPLIT), 1)
        train_df = feat.iloc[:-val_size].reset_index(drop=True); val_df = feat.iloc[-val_size:].reset_index(drop=True)
        train_ds = SeqDataset(train_df, feature_cols, INPUT_SIZE, H); val_ds = SeqDataset(val_df, feature_cols, INPUT_SIZE, H)
        print(f"Sliding windows — train: {len(train_ds)}, val: {len(val_ds)}")
        do_validation = (len(val_ds) > 0)
        if not do_validation:
            print("⚠️ Validation dataset empty — scheduler & early stopping disabled (unless FORCE_FULL_TRAIN=True).")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # scalers
        X_all = train_df[feature_cols].values.astype(np.float32); y_all = train_df["y"].values.astype(np.float32)
        scaler_x = StandardScaler(); scaler_x.fit(X_all.reshape(-1, len(feature_cols)))
        scaler_y = StandardScaler(); scaler_y.fit(y_all.reshape(-1,1))

        model = LSTMForecast(n_features=len(feature_cols), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT, h=H).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        start_epoch = 0

        # single checkpoint path per iteration (overwrite each epoch)
        ckpt_path = MODEL_DIR / f"{coin_id}_iter{iter_i+1}.pt"
        if ckpt_path.exists():
            try:
                d = load_checkpoint(ckpt_path, model, optimizer=opt, device=DEVICE)
                scaler_x = d.get("scaler_x", scaler_x); scaler_y = d.get("scaler_y", scaler_y)
                start_epoch = int(d.get("epoch",0)) + 1
                print(f"🔁 Resumed from single checkpoint {ckpt_path.name} epoch {start_epoch}")
            except Exception as e:
                print("⚠️ Could not load checkpoint:", e)

        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

        best_local_val = math.inf; patience_left = PATIENCE

        # TRAIN LOOP
        for epoch in range(start_epoch, EPOCHS):
            model.train(); total_loss = 0.0; nbat = 0
            for Xb, yb in train_loader:
                B, T, F = Xb.shape
                Xb2 = Xb.reshape(-1, F).numpy()
                Xb2s = scaler_x.transform(Xb2)
                Xb_scaled = torch.tensor(Xb2s.reshape(B, T, F), dtype=torch.float32).to(DEVICE)
                yb_scaled = scaler_y.transform(yb.numpy().reshape(-1,1)).reshape(B, H)
                yb_scaled_t = torch.tensor(yb_scaled, dtype=torch.float32).to(DEVICE)
                opt.zero_grad()
                out = model(Xb_scaled)
                loss = criterion(out, yb_scaled_t)
                loss.backward(); opt.step()
                total_loss += float(loss.item()) * B; nbat += B
            train_loss = total_loss / max(1, nbat)

            # compute unscaled train MAE (meaningful metric in price units)
            model.eval()
            preds_un = []; trues_un = []
            with torch.no_grad():
                for Xb, yb in train_loader:
                    B, T, F = Xb.shape
                    Xb2 = Xb.reshape(-1, F).numpy()
                    Xb2s = scaler_x.transform(Xb2)
                    Xb_scaled = torch.tensor(Xb2s.reshape(B, T, F), dtype=torch.float32).to(DEVICE)
                    out = model(Xb_scaled).cpu().numpy().reshape(-1, H)
                    # inverse scale each predicted horizon element: scaler_y fitted on single values; treat first-step
                    out_un = scaler_y.inverse_transform(out.reshape(-1,1)).reshape(-1, H)
                    preds_un.append(out_un)
                    trues_un.append(yb.numpy())
            if preds_un:
                preds_un = np.vstack(preds_un); trues_un = np.vstack(trues_un)
                # compute MAE on first-step predictions as a compact metric
                train_mae_real = mean_absolute_error(trues_un[:, 0], preds_un[:, 0])
            else:
                train_mae_real = float("nan")

            # validation (if available)
            val_loss = None; val_mae_agg = None
            if do_validation:
                val_loss, val_preds_scaled, val_trues = eval_epoch(model, val_loader, criterion, DEVICE)
                if val_loss is not None and val_preds_scaled.size > 0:
                    val_preds_un = scaler_y.inverse_transform(val_preds_scaled.reshape(-1,1)).reshape(-1, H)
                    val_trues_un = val_trues
                    val_mae_agg = float(np.mean(np.abs(val_trues_un - val_preds_un)))
                if val_loss is not None:
                    scheduler.step(val_loss)

            # print both scaled loss and meaningful MAE
            val_loss_print = f"{val_loss:.6f}" if val_loss is not None else "N/A"
            val_mae_print = f"{val_mae_agg:.6f}" if val_mae_agg is not None else "N/A"
            print(f"epoch {epoch+1}/{EPOCHS} — train_loss(scaled)={train_loss:.6f} train_MAE(unscaled)={train_mae_real:.6f} val_loss={val_loss_print} val_mae={val_mae_print}")

            # save single checkpoint overwriting previous
            save_checkpoint_single(ckpt_path, model, opt, epoch, scaler_x, scaler_y, {"coin": coin_id, "iter": iter_i+1})

            # early stopping if validation available
            if do_validation and (val_loss is not None):
                if val_loss + 1e-12 < best_local_val:
                    best_local_val = val_loss; patience_left = PATIENCE
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print("⏸️ Early stopping triggered (validation)."); break
            else:
                # no validation -> if not forcing full train, continue to run until epochs complete
                pass

        # finished epoch loop for this iteration — load single checkpoint (best/last saved)
        try:
            d = load_checkpoint(ckpt_path, model, optimizer=None, device=DEVICE)
            scaler_x = d.get("scaler_x", scaler_x); scaler_y = d.get("scaler_y", scaler_y)
            print(f"🔁 Loaded checkpoint {ckpt_path.name} after training")
        except Exception as e:
            print("⚠️ Could not reload checkpoint after training:", e)

        # walk-forward predict on chunk-year
        chunk_feat = make_features(chunk, lags=FEATURE_LAGS)
        preds = []; mae_chunk = None; rmse_chunk = None
        if len(chunk_feat) >= INPUT_SIZE:
            for idx in range(INPUT_SIZE, len(chunk_feat)-H+1):
                window = chunk_feat.iloc[idx-INPUT_SIZE:idx]
                Xw = window[feature_cols].values.astype(np.float32)
                Xw_scaled = scaler_x.transform(Xw.reshape(-1, len(feature_cols))).reshape(1, INPUT_SIZE, len(feature_cols))
                xb = torch.tensor(Xw_scaled, dtype=torch.float32).to(DEVICE)
                model.eval()
                with torch.no_grad():
                    out_scaled = model(xb).cpu().numpy().reshape(-1)
                out_unscaled = scaler_y.inverse_transform(out_scaled.reshape(-1,1)).reshape(-1)
                preds.append((chunk_feat.loc[idx, "ds"], out_unscaled[0]))
            whole = chunk_feat[["ds","y"]].copy(); whole["y_pred"] = np.nan
            for (dts,p) in preds: whole.loc[whole["ds"]==dts, "y_pred"] = p
            mask_eval = ~whole["y_pred"].isna()
            if mask_eval.sum() > 0:
                mae_chunk = mean_absolute_error(whole.loc[mask_eval,"y"], whole.loc[mask_eval,"y_pred"])
                rmse_chunk = math.sqrt(mean_squared_error(whole.loc[mask_eval,"y"], whole.loc[mask_eval,"y_pred"]))

        # future forecast
        fc = None
        try:
            fc = iterative_forecast(feat, feature_cols, model, scaler_x, scaler_y, input_size=INPUT_SIZE, h=H)
            fc_path = DATA_DIR / f"{coin_id}_forecast_iter{iter_i+1}.csv"
            fc.to_csv(fc_path, index=False)
            print(f"🔮 Future {H}-day forecast saved to {fc_path.name}")
        except Exception as e:
            print("⚠️ iterative_forecast failed:", e); fc=None

        # plotting: main figure contains price vs predicted, RSI, MACD hist
        if 'whole' in locals() and whole is not None:
            try:
                # prepare indicator series for chunk
                chunk_ind = make_features(chunk, lags=FEATURE_LAGS)
                fig, axes = plt.subplots(3,1, figsize=(14,10), sharex=True, gridspec_kw={"height_ratios":[3,1,1]})
                axes[0].plot(whole["ds"], whole["y"], label="Actual", color="tab:blue")
                axes[0].plot(whole["ds"], whole["y_pred"], label="Predicted (walk-forward)", color="tab:green")
                axes[0].set_title(f"{coin_id.upper()} actual vs predicted for {fetch_start} → {fetch_end}")
                axes[0].legend(); axes[0].grid(True)

                # RSI
                if "rsi_14" in chunk_ind.columns:
                    axes[1].plot(chunk_ind["ds"], chunk_ind["rsi_14"], label="RSI(14)")
                    axes[1].axhline(70, color="red", linestyle="--", linewidth=0.6)
                    axes[1].axhline(30, color="green", linestyle="--", linewidth=0.6)
                    axes[1].set_ylabel("RSI"); axes[1].grid(True)
                else:
                    axes[1].text(0.5,0.5,"RSI not available", ha="center", va="center")

                # MACD hist
                if "macd_hist" in chunk_ind.columns:
                    axes[2].bar(chunk_ind["ds"], chunk_ind["macd_hist"], label="MACD hist")
                    axes[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
                    axes[2].set_ylabel("MACD hist"); axes[2].grid(True)
                else:
                    axes[2].text(0.5,0.5,"MACD hist not available", ha="center", va="center")

                plt.tight_layout()
                plot_path = PLOTS_DIR / f"{coin_id}_chunk_iter{iter_i+1}.png"
                plt.savefig(plot_path); plt.close(fig)
                print(f"🖼️ Plot (indicators) saved to {plot_path.name}")
            except Exception as e:
                print("⚠️ Indicator plotting failed:", e)
        else:
            print("⚠️ No walk-forward predictions available to plot for this chunk.")

        # overlay recent + future
        if fc is not None:
            try:
                recent = cumulative.sort_values("ds").tail(120)
                plt.figure(figsize=(12,6))
                plt.plot(recent["ds"], recent["y"], label="Actual (recent)", color="tab:blue")
                plt.plot(fc["ds"], fc["yhat"], label=f"Forecast next {H} days", color="tab:green", linestyle="--", marker="o")
                plt.axvline(recent["ds"].max(), color="gray", linestyle=":", linewidth=1)
                plt.legend(); plt.grid(True); plt.tight_layout()
                overlay = PLOTS_DIR / f"{coin_id}_future_overlay_iter{iter_i+1}.png"
                plt.savefig(overlay); plt.close()
                print(f"🖼️ Future overlay saved to {overlay.name}")
            except Exception as e:
                print("⚠️ overlay plotting failed:", e)

        # trade suggestion
        suggestion = {"recommendation":"HOLD","reason":"no forecast"}
        if fc is not None:
            last_actual_row = feat.iloc[-1]
            suggestion = suggest_trades(fc, last_actual_row)
            print(f"💡 Trade suggestion: {suggestion['recommendation']} — {suggestion['reason']}")

        # logging
        log_row = {"timestamp": datetime.utcnow().isoformat(), "coin": coin_id, "iter": iter_i+1, "chunk_start": fetch_start.isoformat(), "chunk_end": fetch_end.isoformat(), "rows_cumulative": len(cumulative), "rows_feat": len(feat), "mae": mae_chunk if mae_chunk is not None else "", "rmse": rmse_chunk if rmse_chunk is not None else "", "model_checkpoint": str(ckpt_path), "forecast_csv": str(fc_path) if fc is not None else "", "plot": str(plot_path) if 'plot_path' in locals() else "", "recommendation": suggestion.get("recommendation",""), "recommendation_reason": suggestion.get("reason",""), "expected_return": suggestion.get("expected_return","")}
        if not LOG_FILE.exists(): pd.DataFrame([log_row]).to_csv(LOG_FILE, index=False)
        else: pd.DataFrame([log_row]).to_csv(LOG_FILE, index=False, mode="a", header=False)

        # improvement check
        if mae_chunk is not None:
            if mae_chunk + 1e-12 < best_val:
                print(f"✅ MAE improved from {best_val:.6f} -> {mae_chunk:.6f}"); best_val = mae_chunk; no_improve = 0
            else:
                no_improve += 1; print(f"⚠️ No MAE improvement (count {no_improve}/{MAX_ITER_NO_IMPROVE})")
        else:
            no_improve += 1

        if no_improve >= MAX_ITER_NO_IMPROVE:
            print("🛑 No improvement for several iterations — stopping early for this coin."); break

        # slide window one year older
        fetch_end = fetch_start - timedelta(days=1); fetch_start = fetch_end - timedelta(days=DAYS_PER_CHUNK - 1)

    print(f"=== FINISHED for {coin_id} ===\n")

def main():
    print("Deep trainer starting. Device:", DEVICE)
    print("Hint: set COINGECKO_API_KEY env var to use pro endpoint (optional).")
    for coin in COINS:
        try: run_coin(coin)
        except Exception as e: print(f"Error processing {coin}: {e}")

if __name__ == "__main__":
    main()
