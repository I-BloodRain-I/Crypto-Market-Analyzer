from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MultiHorizonDataset(Dataset):
    """
    Fast multi-horizon dataset backed by pandas DataFrames or NumPy arrays.

    Returns items compatible with MultiHorizonTransformer:
        - past_feats: torch.Tensor shaped [L, F_p]
        - future_feats: torch.Tensor shaped [H, F_f]
        - static_feats: torch.Tensor shaped [F_s] or None
        - target: torch.Tensor shaped [H] or [H, num_targets]

    Args:
        past_features: DataFrame or 2D array with past (historical) features.
        future_features: DataFrame or 2D array with future (known) features.
        targets: DataFrame or 2D array with target variables.
        context_len: Historical context length (L).
        max_horizon: Prediction horizon length (H).
        target_horizons: List of target horizons to predict.
        stride: Stride between windows.
        static_features: DataFrame or 2D array with static features (optional).
    """
    def __init__(
        self,
        past_features: Union[pd.DataFrame, np.ndarray],
        future_features: Union[pd.DataFrame, np.ndarray],
        targets: Union[pd.DataFrame, np.ndarray],
        context_len: int,
        max_horizon: int,
        target_horizons: List[int],
        stride: int = 1,
        static_features: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ):
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.target_horizons = torch.tensor([th-1 for th in target_horizons])
        self.stride = stride

        self.total_rows = len(past_features)
        
        if len(future_features) != self.total_rows:
            raise ValueError("past_features and future_features must have the same length")
        if len(targets) != self.total_rows:
            raise ValueError("past_features and targets must have the same length")
        if static_features is not None and len(static_features) != self.total_rows:
            raise ValueError("past_features and static_features must have the same length")

        self.past_arrays = self._to_column_arrays(past_features)
        self.future_arrays = self._to_column_arrays(future_features)
        self.target_arrays = self._to_column_arrays(targets)
        
        if static_features is not None:
            self.static_arrays = self._to_column_arrays(static_features)
        else:
            self.static_arrays = None

        if len(self.target_arrays) != len(self.target_horizons):
            raise ValueError("Number of target columns must equal number of target_horizons")

        max_start = self.total_rows - (self.context_len + self.max_horizon)
        if max_start < 0:
            self.num_windows = 0
        else:
            self.num_windows = max_start // self.stride + 1

    def _to_column_arrays(self, data: Union[pd.DataFrame, np.ndarray]) -> List[np.ndarray]:
        if isinstance(data, pd.DataFrame):
            return [data[col].values for col in data.columns]
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return [data]
            elif data.ndim == 2:
                return [data[:, i] for i in range(data.shape[1])]
            else:
                raise ValueError(f"NumPy array must be 1D or 2D, got {data.ndim}D")
        else:
            raise TypeError(f"Expected pd.DataFrame or np.ndarray, got {type(data)}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_windows:
            raise IndexError("Index out of range")

        start_idx = idx * self.stride

        past_feats = np.column_stack([
            arr[start_idx:start_idx + self.context_len]
            for arr in self.past_arrays
        ]).astype(np.float32)

        future_start = start_idx + self.context_len
        future_feats = np.column_stack([
            arr[future_start:future_start + self.max_horizon]
            for arr in self.future_arrays
        ]).astype(np.float32)

        horizons = self.target_horizons.tolist()
        target = np.array([
            arr[future_start + int(h)]
            for arr, h in zip(self.target_arrays, horizons)
        ], dtype=np.int64)
        target_tensor = torch.from_numpy(target)

        if self.static_arrays is not None:
            static_feats = np.array([
                arr[start_idx] for arr in self.static_arrays
            ], dtype=np.float32)
            static_tensor = torch.from_numpy(static_feats)
        else:
            static_tensor = None

        past_tensor = torch.from_numpy(past_feats)
        future_tensor = torch.from_numpy(future_feats)

        inputs = [past_tensor, future_tensor]
        if static_tensor is not None:
            inputs.append(static_tensor)

        return inputs, target_tensor