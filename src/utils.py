import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def convert_timeframe_to_seconds(timeframe: str) -> int:
    """Convert a timeframe string to its equivalent in seconds.

    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d').

    Returns:
        int: Equivalent timeframe in seconds.

    Raises:
        ValueError: If the timeframe format is invalid.
    """
    unit_multipliers = {
        'm': 60,
        'h': 3600,
        'd': 86400,
        'w': 604800,
        'M': 2592000,  # Approximate month as 30 days
    }

    try:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit not in unit_multipliers:
            raise ValueError(f"Invalid timeframe unit: {unit}")
        return value * unit_multipliers[unit]
    except (IndexError, ValueError) as e:
        raise ValueError(f"Invalid timeframe format: {timeframe}") from e
    
def save_parquet(df: pd.DataFrame, folder: str, chunk_size: int = 1_000_000):
    """Save a DataFrame as multiple Parquet files in a specified folder.
    
    Args:
        df: The DataFrame to save.
        folder: The folder to save the Parquet files in.
        chunk_size: Number of rows per Parquet file.
    """
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        table = pa.Table.from_pandas(chunk)
        pq.write_table(table, f"{folder}/part_{i//chunk_size}.parquet")

def load_parquet(folder: str) -> pd.DataFrame:
    """Load multiple Parquet files from a specified folder into a single DataFrame. 

    Args:
        folder: The folder containing the Parquet files.  

    Returns:
        The combined DataFrame.
    """
    dfs = []
    for file in os.listdir(folder):
        if file.endswith(".parquet"):
            table = pq.read_table(os.path.join(folder, file))
            dfs.append(table.to_pandas())
    return pd.concat(dfs, ignore_index=True)