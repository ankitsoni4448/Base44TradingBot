import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for model input (fill missing values, etc.)
    """
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df
