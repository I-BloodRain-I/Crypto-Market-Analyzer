import os
from re import S
from typing import Union, List, Optional

import torch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """
    Fast sliding-window dataset backed by PyArrow parquet.

    Returns items compatible with MultiHorizonTransformer:
        - past_feats: torch.Tensor shaped [L, F_p]
        - future_feats: torch.Tensor shaped [H, F_f]
        - static_feats: torch.Tensor shaped [F_s] or None
        - target: torch.Tensor shaped [H] or [H, num_targets]

    Args:
        parquet_path: Path to parquet files.
        context_len: Historical context length (L).
        max_horizon: Prediction horizon length (H).
        target_horizons: List of target horizons to predict (optional).
        stride: Stride between windows.
        past_feature_indices: List of column indices or names to use as past features.
        future_feature_indices: List of column indices or names to use as future (known) features.
        static_feature_indices: List of column indices or names to use as static features (optional).
        target: Column name, column index, or list of names/indices for target(s).
    """
    def __init__(
        self,
        parquet_path: str,
        context_len: int = 120,
        max_horizon: int = 15,
        target_horizons: Optional[List[int]] = None,
        stride: int = 15,
        past_feature_indices: Optional[List[Union[int, str]]] = None,
        future_feature_indices: Optional[List[Union[int, str]]] = None,
        static_feature_indices: Optional[List[Union[int, str]]] = None,
        target: Union[int, str, List[Union[int, str]]] = 3
    ):
        self.context_len = context_len
        self.max_horizon = max_horizon
        self.target_horizons = torch.tensor([th-1 for th in target_horizons]) if target_horizons is not None else None
        self.stride = stride

        files = sorted([os.path.join(parquet_path, f) for f in os.listdir(parquet_path) if f.endswith(".parquet")])
        tables = [pq.read_table(f) for f in files]
        self.table = pa.concat_tables(tables)
        self.total_rows = len(self.table)

        names = list(self.table.schema.names)

        def resolve_cols(cols):
            if cols is None:
                return None
            if isinstance(cols, list):
                resolved = []
                for c in cols:
                    if isinstance(c, int):
                        resolved.append(names[c])
                    else:
                        resolved.append(str(c))
                return resolved
            if isinstance(cols, int):
                return [names[cols]]
            return [str(cols)]

        self.past_feature_names = resolve_cols(past_feature_indices) or names
        self.future_feature_names = resolve_cols(future_feature_indices) or [names[i] for i in (0, 1, 2) if i < len(names)]
        self.static_feature_names = resolve_cols(static_feature_indices)
        self.target_names = resolve_cols(target)

        self.past_arrays = [
            self.table.column(name).to_numpy()
            for name in self.past_feature_names
        ]
        self.future_arrays = [
            self.table.column(name).to_numpy()
            for name in self.future_feature_names
        ]
        if self.static_feature_names is not None:
            self.static_arrays = [
                self.table.column(name).to_numpy()
                for name in self.static_feature_names
            ]
        else:
            self.static_arrays = None

        # target can be one or multiple columns
        if self.target_names is None:
            raise ValueError("target must be provided as column name/index or list thereof")
        self.target_arrays = [
            self.table.column(name).to_numpy()
            for name in self.target_names
        ]

        max_start = self.total_rows - (self.context_len + self.max_horizon)
        if max_start < 0:
            self.num_windows = 0
        else:
            self.num_windows = max_start // self.stride + 1

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

        # target: handle single or multi-column targets
        if len(self.target_arrays) == 1:
            target = self.target_arrays[0][
                future_start:future_start + self.max_horizon
            ].astype(np.float32)
            target_tensor = torch.from_numpy(target)
        else:
            target = np.column_stack([
                arr[future_start:future_start + self.max_horizon]
                for arr in self.target_arrays
            ]).astype(np.float32)
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

        if self.target_horizons is not None:
            target_tensor = target_tensor[self.target_horizons]
        return inputs, target_tensor