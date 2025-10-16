# backend/data_loader.py
import os
import pandas as pd

DATA_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_stream", "ETHUSDT_live.csv")

def load_historical(symbol: str = "ETHUSDT", limit: int = 200):
    """
    Loads historical CSV if present, otherwise returns an empty DataFrame.
    """
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
        # ensure timestamp column parsed if present
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.tail(limit).reset_index(drop=True)
    else:
        # return empty DataFrame (upstream code will handle mock fallback)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
