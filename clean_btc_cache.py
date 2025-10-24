# clean_btc_cache.py
import pandas as pd
import os
from datetime import datetime

SRC = "data/btc_market_20251022_175416.csv"  # replace if needed or pass via env
OUT = "data/btc_market.csv"
BACKUP_TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
BACKUP = f"data/btc_market_backup_{BACKUP_TS}.csv"

print("Reading:", SRC)
# read without forcing dtypes; allow pandas to parse headers
df = pd.read_csv(SRC, dtype=str, keep_default_na=False)

print("Raw columns:", list(df.columns))
print("First 5 raw rows:")
print(df.head(5).to_string(index=False))

# Heuristic cleaning:
# If file looks like it contains a header row as data (e.g. first row has 'BTC-USD' in some column),
# try to detect and remove it.
# Convert columns to a working set: possible names -> target names
possible_price_cols = ["y", "price", "Close", "close", "adjclose"]
possible_date_cols = ["ds", "Date", "date", "timestamp"]
possible_id_cols = ["unique_id", "ticker", "symbol"]

# lower-case column mapping for detection
cols_lower = {c.lower(): c for c in df.columns}

# try to find columns
date_col = None
price_col = None
id_col = None

for cand in possible_date_cols:
    if cand.lower() in cols_lower:
        date_col = cols_lower[cand.lower()]
        break

for cand in possible_price_cols:
    if cand.lower() in cols_lower:
        price_col = cols_lower[cand.lower()]
        break

for cand in possible_id_cols:
    if cand.lower() in cols_lower:
        id_col = cols_lower[cand.lower()]
        break

# If none found, try positional guesses
if date_col is None and "0" in df.columns:
    date_col = df.columns[0]
if price_col is None and len(df.columns) >= 2:
    price_col = df.columns[1]

print("Detected columns -> date:", date_col, ", price:", price_col, ", id:", id_col)

# If first row looks like a leftover header (non-date in date_col), drop it
def looks_like_header_row(row):
    # if date_col exists and is not parseable as a date -> likely header
    if date_col:
        val = str(row.get(date_col, "")).strip()
        try:
            pd.to_datetime(val)
            return False
        except Exception:
            # if it's empty or contains non-date text, it's likely header
            return True
    return False

# If the first row is header-like, drop it
if looks_like_header_row(df.iloc[0]):
    print("Detected header-like first row; dropping it.")
    df = df.iloc[1:].reset_index(drop=True)

# Build normalized dataframe with columns ['unique_id', 'ds', 'y']
norm = pd.DataFrame()

# fill ds
if date_col:
    norm["ds"] = pd.to_datetime(df[date_col], errors="coerce")
else:
    # try parsing an ISO-like first column
    norm["ds"] = pd.to_datetime(df.iloc[:,0], errors="coerce")

# fill y (price)
if price_col:
    # coerce numeric
    norm["y"] = pd.to_numeric(df[price_col].str.replace(",",""), errors="coerce")
else:
    # try second column
    norm["y"] = pd.to_numeric(df.iloc[:,1].astype(str).str.replace(",",""), errors="coerce")

# fill unique_id
if id_col:
    norm["unique_id"] = df[id_col].fillna("").astype(str)
else:
    # if there is a column with BTC or BTC-USD in first data row use it, else default to 'bitcoin'
    first_row_vals = df.iloc[0].astype(str).str.lower().values
    if any("btc" in v for v in first_row_vals):
        norm["unique_id"] = "bitcoin"
    else:
        norm["unique_id"] = "bitcoin"

# Drop rows with invalid dates or prices
before = len(norm)
norm = norm.dropna(subset=["ds","y"]).reset_index(drop=True)
after = len(norm)
print(f"Dropped {before-after} rows with invalid date/price")

# Ensure ds sorted and unique
norm = norm.sort_values("ds").drop_duplicates(subset=["ds"]).reset_index(drop=True)

# Save backup of original and write canonical file
print("Backing up original to", BACKUP)
os.makedirs(os.path.dirname(BACKUP), exist_ok=True)
df.to_csv(BACKUP, index=False)

print("Writing cleaned file to", OUT)
norm[["unique_id","ds","y"]].to_csv(OUT, index=False, date_format="%Y-%m-%d")

print("Cleaned file summary:")
print(norm.head(5).to_string(index=False))
print(norm.tail(5).to_string(index=False))
print("rows:", len(norm))
