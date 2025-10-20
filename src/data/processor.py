"""Data processing helpers for preparing and validating datasets.

This module provides the `DataProcessor` class which contains utilities for
building preprocessing pipelines, applying them to dataframes, splitting
datasets for training/validation/testing (both random and time-based), and
performing common data quality checks such as time continuity and missing
value detection.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Sampler, Dataset

from .config import DataPipelineConfig
from .pipeline_builder import PipelineBuilder

logger = logging.getLogger(__name__)


class DataProcessor:
    """Utility collection for preparing and validating tabular data.

    Methods in this class are implemented as static and class methods so they
    can be used without instantiating the class. Typical usage patterns include
    building a pipeline from a configuration, transforming a DataFrame with a
    fitted pipeline, splitting the DataFrame into train/test (and optional
    validation) sets or validating data quality.

    Constants:
		PROJECT_ROOT: Path to the project root inferred from this file's location.
		DATA_FOLDER: Top-level data directory inside the project.
		PROCESSED_FOLDER: Folder containing processed data.
    """

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_FOLDER = PROJECT_ROOT / "data"
    PROCESSED_FOLDER = DATA_FOLDER / "processed"

    @staticmethod
    def process_data(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
        """Transform a DataFrame using a fitted preprocessing pipeline.

        The pipeline must be fitted before calling this function. The returned
        DataFrame will use the pipeline's output feature names as columns.

        Args:
            df: The input DataFrame to transform.
            pipeline: A fitted sklearn Pipeline or ColumnTransformer.

        Returns:
            A DataFrame containing the transformed features.
        """
        logger.debug("Processing data with pipeline: \n%s", pipeline)
        int_cols = [f"remainder__{col}" for col in df.select_dtypes(include=['int']).columns.tolist()]
        processed_array = pipeline.transform(df)
        processed_df = pd.DataFrame(processed_array, columns=pipeline.get_feature_names_out())
        processed_df[int_cols] = processed_df[int_cols].astype("int64")
        return processed_df

    @staticmethod
    def make_dataloader(
        X: NDArray,
        y: Optional[NDArray] = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        sampler: Optional[Sampler] = None,
        device: Optional[torch.device] = None,
        x_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64
    ) -> DataLoader:
        """Create a PyTorch DataLoader from feature and target arrays.

        Args:
            X: Feature array.
            y: Optional target array.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle data at each epoch.
            num_workers: Number of subprocesses for data loading.
            pin_memory: Whether to use pinned memory.
            sampler: Optional custom sampler for data loading.
            device: Device to which tensors will be moved (default: auto-detect).
            x_dtype: Data type for feature tensors.
            y_dtype: Data type for target tensors.

        Returns:
            A PyTorch DataLoader instance.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)

        if y is not None:
            dataset = TensorDataset(torch.tensor(X, dtype=x_dtype), torch.tensor(y, dtype=y_dtype))
        else:
            dataset = TensorDataset(torch.tensor(X, dtype=x_dtype))
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=sampler,
            shuffle=shuffle, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )

    @staticmethod
    def build_pipeline(config: DataPipelineConfig) -> Pipeline:
        """Construct a preprocessing pipeline from a configuration.

        Args:
            config: PipelineConfig describing preprocessing behavior for feature groups.

        Returns:
            A sklearn Pipeline instance constructed according to the configuration.
        """
        return PipelineBuilder.build(config)

    @classmethod
    def split_data_random(
        cls, 
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None, 
        random_state: int = 42
    ) -> Tuple[NDArray, NDArray, Optional[NDArray], NDArray, NDArray, Optional[NDArray]]:
        """Randomly split a DataFrame into train/test/(optional)validation sets.

        When `val_size` is provided the method returns three feature splits and
        three target splits; otherwise the validation positions in the returned
        tuple are None. The function validates the arguments before splitting.

        Args:
            df: The full DataFrame to split.
            target_col: Name of the target column to separate from features.
            test_size: Fraction of data to reserve for the test set.
            val_size: Optional fraction to reserve for a validation set.
            random_state: Seed controlling randomness for reproducible splits.

        Returns:
            A tuple in the order (X_train, X_test, X_val_or_None, y_train, y_test, y_val_or_None).
        """
        cls._validate_arguments_for_splitting(df, target_col, test_size, val_size)
        train_size = 1 - test_size - (val_size if val_size is not None else 0)
        
        if val_size is not None:
            train_df, temp_df = train_test_split(df, test_size=(test_size + val_size), random_state=random_state)
            relative_val_size = val_size / (test_size + val_size)
            test_df, val_df = train_test_split(temp_df, test_size=relative_val_size, random_state=random_state + 1)

            X_train = train_df.drop(columns=[target_col]).to_numpy()
            X_test  = test_df.drop(columns=[target_col]).to_numpy()
            X_val   = val_df.drop(columns=[target_col]).to_numpy()
            y_train = train_df[target_col].to_numpy()
            y_test  = test_df[target_col].to_numpy()
            y_val   = val_df[target_col].to_numpy()

            logger.debug(
                "Splitted data: train_dataset=%s (%.2f%%), test_dataset=%s (%.2f%%), val_dataset=%s (%.2f%%)",
                train_df.shape[0], train_size*100, test_df.shape[0], test_size*100, val_df.shape[0], val_size*100
            )
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            X_train = train_df.drop(columns=[target_col]).to_numpy()
            X_test  = test_df.drop(columns=[target_col]).to_numpy()
            y_train = train_df[target_col].to_numpy()
            y_test  = test_df[target_col].to_numpy()

            logger.debug(
                "Splitted data: train_dataset=%s (%.2f%%), test_dataset=%s (%.2f%%)",
                train_df.shape[0], train_size*100, test_df.shape[0], test_size*100
            )
            return X_train, X_test, None, y_train, y_test, None

    @classmethod
    def split_data_time_based(
        cls,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray], NDArray, NDArray, Optional[NDArray]]:
        """Split a DataFrame into train/test/(optional)validation while preserving order.

        This method slices the DataFrame by index so that temporal order is not
        disturbed — useful for time-series experiments where leakage must be
        avoided.

        Args:
            df: The full DataFrame to split.
            target_col: Name of the target column to separate from features.
            test_size: Fraction of data to reserve for the test set.
            val_size: Optional fraction to reserve for a validation set.

        Returns:
            A tuple in the order (X_train, X_test, X_val_or_None, y_train, y_test, y_val_or_None).
        """
        cls._validate_arguments_for_splitting(df, target_col, test_size, val_size)
        train_size = 1 - test_size - (val_size if val_size is not None else 0)

        n = len(df)
        if val_size is not None:
            n_train = int(n * train_size)
            n_val   = int(n * val_size)

            train_df = df.iloc[:n_train]
            val_df   = df.iloc[n_train:n_train + n_val]
            test_df  = df.iloc[n_train + n_val:]

            y_train = train_df[target_col].to_numpy()
            y_test  = test_df[target_col].to_numpy()
            y_val   = val_df[target_col].to_numpy()
            X_train = train_df.drop(columns=[target_col]).to_numpy()
            X_test  = test_df.drop(columns=[target_col]).to_numpy()
            X_val   = val_df.drop(columns=[target_col]).to_numpy()

            logger.debug(
                "Splitted data: train_dataset=%s (%.2f), test_dataset=%s (%.2f), val_dataset=%s (%.2f)",
                train_df.shape[0], train_size, test_df.shape[0], test_size, val_df.shape[0], val_size
            )
            return X_train, X_test, X_val, y_train, y_test, y_val
        else:
            n_train  = int(n * train_size)
            train_df = df.iloc[:n_train]
            test_df  = df.iloc[n_train:]

            y_train = train_df[target_col].to_numpy()
            y_test  = test_df[target_col].to_numpy()
            X_train = train_df.drop(columns=[target_col]).to_numpy()
            X_test  = test_df.drop(columns=[target_col]).to_numpy()

            logger.debug(
                "Splitted data: train_dataset=%s (%.2f), test_dataset=%s (%.2f)",
                train_df.shape[0], train_size, test_df.shape[0], test_size
            )
            return X_train, X_test, None, y_train, y_test, None

    @classmethod
    def split_dataset_time_based(
        cls,
        dataset: Dataset,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
    ) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """Split a PyTorch Dataset into train/test/(optional)validation while preserving order.

        This method slices the Dataset by index so that temporal order is not
        disturbed — useful for time-series experiments where leakage must be
        avoided.

        Args:
            dataset: The full PyTorch Dataset to split.
            test_size: Fraction of data to reserve for the test set.
            val_size: Optional fraction to reserve for a validation set.

        Returns:
            A tuple in the order (train_dataset, test_dataset, val_dataset_or_None).
        """
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch.utils.data.Dataset instance")
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if val_size is not None and (val_size <= 0 or val_size >= 1):
            raise ValueError("val_size must be between 0 and 1")
        if val_size is not None and (test_size + val_size) >= 1:
            raise ValueError("The sum of test_size and val_size must be less than 1")

        n = len(dataset)
        train_size = 1 - test_size - (val_size if val_size is not None else 0)

        if val_size is not None:
            n_train = int(n * train_size)
            n_val   = int(n * val_size)

            train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
            val_dataset   = torch.utils.data.Subset(dataset, range(n_train, n_train + n_val))
            test_dataset  = torch.utils.data.Subset(dataset, range(n_train + n_val, n))

            logger.debug(
                "Splitted dataset: train_dataset=%s (%.2f), test_dataset=%s (%.2f), val_dataset=%s (%.2f)",
                len(train_dataset), train_size, len(test_dataset), test_size, len(val_dataset), val_size
            )
            return train_dataset, test_dataset, val_dataset
        else:
            n_train  = int(n * train_size)
            train_dataset = torch.utils.data.Subset(dataset, range(0, n_train))
            test_dataset  = torch.utils.data.Subset(dataset, range(n_train, n))

            logger.debug(
                "Splitted dataset: train_dataset=%s (%.2f), test_dataset=%s (%.2f)",
                len(train_dataset), train_size, len(test_dataset), test_size
            )
            return train_dataset, test_dataset, None

    @staticmethod
    def _validate_arguments_for_splitting(
        df: pd.DataFrame,
        target_col: str,
        test_size: float,
        val_size: Optional[float] = None
    ) -> None:
        """Run sanity checks on splitting arguments and raise helpful errors.

        The function ensures proportions are sensible, the target column exists
        and is numeric, and the DataFrame is not empty.

        Args:
            df: DataFrame that will be split.
            target_col: Name of the target column.
            test_size: Fraction reserved for test set.
            val_size: Optional fraction reserved for validation set.

        Raises:
            ValueError: If any argument is invalid.
        """
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if val_size is not None and (val_size <= 0 or val_size >= 1):
            raise ValueError("val_size must be between 0 and 1")
        if val_size is not None and (test_size + val_size) >= 1:
            raise ValueError("The sum of test_size and val_size must be less than 1")
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            raise ValueError(f"target_col '{target_col}' must be numeric")

    @staticmethod
    def sort_by_time(df: pd.DataFrame, time_col: str = "open_time", ascending: bool = True) -> pd.DataFrame:
        """Return a DataFrame sorted by a specified time column.

        Args:
            df: DataFrame to sort.
            time_col: Name of the column containing timestamps.

        Returns:
            A new DataFrame sorted by the time column.

        Raises:
            ValueError: If the DataFrame is empty or the time column is missing.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        if time_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{time_col}' column")

        sorted_df = df.sort_values(by=time_col, kind="mergesort", ascending=ascending).reset_index(drop=True)
        return sorted_df

    @staticmethod
    def check_time_continuity(df: pd.DataFrame, delta_ms: int = 60_000, time_col: str = "open_time") -> Dict[str, Any]:
        """Check timestamps for gaps, duplicates, or intervals that are too short.

        The function assumes timestamps are integer milliseconds since epoch. It
        returns a dictionary with three lists under keys: "missing",
        "duplicate", and "too_close" describing problematic intervals.

        Args:
            df: DataFrame containing a time column.
            delta_ms: Expected minimum interval between consecutive timestamps in milliseconds.
            time_col: Column name that holds the timestamp values.

        Returns:
            A dictionary summarizing detected issues.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        if time_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{time_col}' column")
        if not pd.api.types.is_integer_dtype(df[time_col]):
            raise ValueError(f"'{time_col}' column must be of integer type representing milliseconds since epoch")

        arr = df[time_col].to_numpy(copy=False)
        if not np.all(arr[:-1] <= arr[1:]):
            arr = np.sort(arr, kind="mergesort")

        diffs = np.diff(arr)

        result = {"missing": [], "duplicate": [], "too_close": []}

        miss_mask = diffs > delta_ms
        if miss_mask.any():
            idx = np.nonzero(miss_mask)[0]
            result["missing"] = [
                {
                    "prev_time_ms": int(arr[i]),
                    "cur_time_ms": int(arr[i + 1]),
                    "diff_ms": int(diffs[i]),
                    "missing_rows": int(diffs[i] // delta_ms - 1),
                }
                for i in idx
            ]

        dup_mask = diffs == 0
        if dup_mask.any():
            idx = np.nonzero(dup_mask)[0]
            result["duplicate"] = [
                {
                    "prev_time_ms": int(arr[i]),
                    "cur_time_ms": int(arr[i + 1]),
                    "diff_ms": 0,
                }
                for i in idx
            ]

        close_mask = (diffs > 0) & (diffs < delta_ms)
        if close_mask.any():
            idx = np.nonzero(close_mask)[0]
            result["too_close"] = [
                {
                    "prev_time_ms": int(arr[i]),
                    "cur_time_ms": int(arr[i + 1]),
                    "diff_ms": int(diffs[i]),
                }
                for i in idx
            ]

        return result

    @staticmethod
    def check_no_missing(df: pd.DataFrame) -> List[Tuple[Union[int, str], str]]:
        """Find cells that are missing or contain common invalid tokens.

        The check treats a set of values as invalid (empty strings, various
        case variants of 'null', NaN, and infinities). It returns a list of
        (index, column_name) pairs for each offending cell.

        Args:
            df: DataFrame to inspect.

        Returns:
            A list of tuples identifying the location of missing/invalid cells.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        invalid = {"", None, "None", "nan", "NaN", "NULL", "null", np.nan, np.inf, -np.inf}
        mask = df.isna() | df.astype(str).isin(invalid)
        rows, cols = np.where(mask)
        return [(df.index[i], df.columns[j]) for i, j in zip(rows, cols)]
  
    @staticmethod
    def check_no_duplicates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Identify duplicate rows in the DataFrame.

        The check considers duplicates based on the specified columns. It
        returns a DataFrame containing the duplicate rows if any are found,
        otherwise an empty DataFrame is returned.

        Args:
            df: DataFrame to inspect.
            columns: List of column names to consider for identifying duplicates.

        Returns:
            A DataFrame containing the duplicate rows found.
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"column '{col}' not found in DataFrame")
            
        return df[df.duplicated(subset=columns, keep=False)]

    @staticmethod
    def has_future_leakage(
        data1: Union[pd.DataFrame, NDArray], 
        data2: Union[pd.DataFrame, NDArray], 
        time_col: Union[str, int] = "open_time"
    ) -> bool:
        """Check if two datasets have overlapping timestamps.

        This function extracts timestamps from the specified column in both
        datasets, normalizes them to integer milliseconds since epoch if they
        are in datetime format, and checks for any overlap. It supports both
        pandas DataFrames and numpy arrays as input.

        Args:
            data1: First dataset as a DataFrame or numpy array.
            data2: Second dataset as a DataFrame or numpy array.
            time_col: Column name or index containing timestamps.

        Returns:
            True if overlapping timestamps are found, False otherwise.
        """
        def _extract_timestamps(x):
            if isinstance(x, pd.DataFrame):
                if isinstance(time_col, int):
                    return x.iloc[:, time_col].to_numpy()
                return x[time_col].to_numpy()
            elif isinstance(x, np.ndarray):
                if x.ndim == 1:
                    return x
                elif x.ndim == 2:
                    if not isinstance(time_col, int):
                        raise TypeError("For numpy 2D time_col must be a column index (int)")
                    return x[:, time_col]
                else:
                    raise TypeError("Only 1D and 2D numpy arrays are supported")
            else:
                raise TypeError("data1 and data2 must be pandas.DataFrame or numpy.ndarray")

        data1_vals = np.asarray(_extract_timestamps(data1))
        data2_vals = np.asarray(_extract_timestamps(data2))

        if np.issubdtype(data1_vals.dtype, np.datetime64):
            data1_vals = data1_vals.astype("datetime64[ms]").astype(np.int64)
        if np.issubdtype(data2_vals.dtype, np.datetime64):
            data2_vals = data2_vals.astype("datetime64[ms]").astype(np.int64)

        data1_set = set(data1_vals.tolist())
        data2_set = set(data2_vals.tolist())

        return len(data1_set.intersection(data2_set)) > 0