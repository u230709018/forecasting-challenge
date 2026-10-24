from __future__ import annotations

import pandas as pd


HOLDOUT_SIZE = 24

def split_holdout(table: pd.DataFrame, holdout_size: int = HOLDOUT_SIZE) -> tuple[pd.DataFrame, pd.Series]:

    if len(table) <= holdout_size:
        raise ValueError(f"Insufficient row number: {len(table)}. Row number must be greater than {holdout_size}.")
    
    if not (table["timestamp"].is_monotonic_increasing and table["timestamp"].is_unique):
        raise ValueError("timestamps must be sorted and unique.")
    
    train = table.iloc[:-holdout_size]
    holdout_timestamps = table["timestamp"].iloc[-holdout_size:]
    return train, holdout_timestamps
    