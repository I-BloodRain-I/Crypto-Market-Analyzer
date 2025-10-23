import numpy as np
import pandas as pd
from typing import Iterable, List, Sequence, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class OrderbookNormalizer(BaseEstimator, TransformerMixin):
    """Normalize orderbook depth and notional columns for sklearn pipelines.

    This transformer performs row-wise normalization of depth and notional features
    relative to a reference price (``center_col``). It applies a signed log
    compression and can optionally clip the resulting values. The transformer
    keeps the original features and appends normalized columns (with suffix),
    and can compute per-row aggregate features describing side totals and
    imbalances.

    Args:
        depth_prefix: Prefix used for depth feature column names (levels appended with "_<level>").
        notional_prefix: Prefix used for notional feature column names (levels appended with "_<level>").
        levels: Sequence of integer price levels that will be used to build column names.
        center_col: Column name containing the reference price used to normalize each row.
        eps: Small constant added to denominators to avoid division by zero.
        clip_log: If set, clip the log-compressed values to the range [-clip_log, clip_log].
        add_suffix: Suffix appended to normalized column names. If None, "_n" is used.
        add_aggregates: If True, compute per-row aggregate features (totals and imbalance measures).
        copy: If True, operate on a copy of the input DataFrame; otherwise modify in place.
    """

    def __init__(
        self,
        depth_prefix: str = "depth",
        notional_prefix: str = "notional",
        levels: Sequence[int] = tuple(list(range(-5, 0)) + list(range(1, 6))),
        center_col: str = "close",
        eps: float = 1e-8,
        clip_log: Optional[float] = None,
        add_suffix: str = "_n",
        add_aggregates: bool = True,
        copy: bool = True,
    ):
        """Initialize the normalizer with configuration options.

        The constructor only stores configuration and does not inspect the input data.
        Call fit() to validate columns and prepare feature name output.
        """
        self.depth_prefix = depth_prefix
        self.notional_prefix = notional_prefix
        self.levels = tuple(levels)
        self.center_col = center_col
        self.eps = float(eps)
        self.clip_log = clip_log
        self.add_suffix = add_suffix
        self.add_aggregates = add_aggregates
        self.copy = copy

        # Will be set in fit() for compatibility with get_feature_names_out
        self._feature_names_out: Optional[List[str]] = None

        self._drop_columns: List[str] = [center_col]
        self._drop_columns.extend(
            [f"{self.depth_prefix}_{i}" for i in self.levels])
        self._drop_columns.extend(
            [f"{self.notional_prefix}_{i}" for i in self.levels]
        )

    def fit(self, X: pd.DataFrame, y=None):
        """Validate input columns and compute output feature names.

        Args:
            X: Input DataFrame used to validate column names and infer feature names.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted transformer (self).
        """
        self._validate_columns(X)
        self._feature_names_out = self._compute_future_column_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization and optional aggregate computation to the DataFrame.

        The transformer performs the following steps:
        1. Validate required columns are present.
        2. Row-wise normalize depth and notional blocks relative to `center_col`.
        3. Apply signed log compression and optional clipping to the normalized blocks.
        4. Append normalized columns to the DataFrame.
        5. If `add_aggregates` is True, compute per-row totals and imbalance metrics
           for bids and asks and append them as new columns.

        Args:
            X: Input DataFrame containing required depth, notional and center columns.

        Returns:
            DataFrame augmented with normalized and (optionally) aggregate columns.
        """
        self._validate_columns(X)

        df = X.copy() if self.copy else X

        depth_cols = [f"{self.depth_prefix}_{i}" for i in self.levels]
        notional_cols = [f"{self.notional_prefix}_{i}" for i in self.levels]

        depth_norm = self._norm_log_block(df, self.depth_prefix, depth_cols)
        notional_norm = self._norm_log_block(df, self.notional_prefix, notional_cols)

        df[depth_norm.columns] = depth_norm
        df[notional_norm.columns] = notional_norm

        new_cols = list(depth_norm.columns) + list(notional_norm.columns)

        if self.add_aggregates:
            bid_levels = [i for i in self.levels if i < 0]
            ask_levels = [i for i in self.levels if i > 0]

            d_bid_cols = [f"{self.depth_prefix}_{i}" for i in bid_levels]
            d_ask_cols = [f"{self.depth_prefix}_{i}" for i in ask_levels]
            df["total_depth_bid"] = np.log(1 + df[d_bid_cols].sum(axis=1).div(df[self.center_col] + self.eps))
            df["total_depth_ask"] = np.log(1 + df[d_ask_cols].sum(axis=1).div(df[self.center_col] + self.eps))
            df["total_depth"] = df["total_depth_bid"] + df["total_depth_ask"]

            n_bid_cols = [f"{self.notional_prefix}_{i}" for i in bid_levels]
            n_ask_cols = [f"{self.notional_prefix}_{i}" for i in ask_levels]
            df["total_notional_bid"] = np.log(1 + df[n_bid_cols].sum(axis=1).div(np.power(df[self.center_col] + self.eps, 2)))
            df["total_notional_ask"] = np.log(1 + df[n_ask_cols].sum(axis=1).div(np.power(df[self.center_col] + self.eps, 2)))
            df["total_notional"] = df["total_notional_bid"] + df["total_notional_ask"]

            eps = self.eps
            df["depth_imbalance_ratio"] = df["total_depth_bid"] / (df["total_depth_ask"] + eps)
            df["depth_imbalance_signed"] = (
                (df["total_depth_bid"] - df["total_depth_ask"]) / (df["total_depth"] + eps)
            )

            df["notional_imbalance_ratio"] = df["total_notional_bid"] / (df["total_notional_ask"] + eps)
            df["notional_imbalance_signed"] = (
                (df["total_notional_bid"] - df["total_notional_ask"]) / (df["total_notional"] + eps)
            )

            new_cols += [
                "total_depth_bid", "total_depth_ask", "total_depth",
                "total_notional_bid", "total_notional_ask", "total_notional",
                "depth_imbalance_ratio", "depth_imbalance_signed",
                "notional_imbalance_ratio", "notional_imbalance_signed",
            ]

        return df[new_cols]

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        """Return output feature names produced by transform().

        If the transformer was fitted, cached feature names are returned. If fit()
        was not called, provide `input_features` to compute the expected output names.

        Args:
            input_features: Optional list of input feature names used to compute output names if
                fit() was not previously called.

        Returns:
            Array of output feature names (dtype object).

        Raises:
            ValueError: If `input_features` is None and fit() was not called prior to this call.
        """
        if self._feature_names_out is not None:
            return np.array(self._feature_names_out, dtype=object)
        if input_features is None:
            raise ValueError("Pass input_features or call fit() before get_feature_names_out().")
        return np.array(self._compute_future_column_names(), dtype=object)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Ensure required columns for normalization are present in the DataFrame.

        The method checks for the presence of the reference center column as well as
        all expected depth and notional level columns and raises KeyError if any are
        missing.

        Args:
            df: DataFrame to validate.

        Raises:
            KeyError: If the reference column or any required level columns are missing.
        """
        if self.center_col not in df.columns:
            raise KeyError(f"Required center column '{self.center_col}' is missing.")
        
        depth_cols = [f"{self.depth_prefix}_{i}" for i in self.levels]
        notional_cols = [f"{self.notional_prefix}_{i}" for i in self.levels]
        required = depth_cols + notional_cols

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _norm_log_block(self, df: pd.DataFrame, prefix: str, cols: List[str]) -> pd.DataFrame:
        """Normalize a block of columns by the center column and apply signed log.

        The block is divided by the `center_col` value (per-row), then transformed
        via sign(x) * log1p(abs(x)). Optionally the result is clipped to the
        symmetric range [-clip_log, clip_log]. The returned DataFrame uses the
        configured suffix for column names.

        Args:
            df: Input DataFrame containing the columns to normalize and the center column.
            prefix: Prefix indicating whether the block is depth or notional features.
            cols: List of column names that will be normalized.

        Returns:
            DataFrame with normalized columns and suffixed names.
        """
        eps = self.eps
        center = (df[self.center_col].astype(float) + eps)

        if prefix == self.notional_prefix:
            denom = np.power(center + eps, 2)
        else:
            denom = center + eps

        block = df[cols].astype(float).div(denom, axis=0)
        block = np.sign(block) * np.log1p(np.abs(block))

        if self.clip_log is not None:
            clipv = float(self.clip_log)
            block = block.clip(lower=-clipv, upper=clipv)

        suffix = self.add_suffix if self.add_suffix is not None else "_n"
        block.columns = [f"{c}{suffix}" for c in cols]
        return block

    def _compute_future_column_names(self) -> List[str]:
        """Compute the list of column names that transform() will produce.

        Returns:
            List of output column names in the order they will appear after transform().
        """
        depth_cols = [f"{self.depth_prefix}_{i}" for i in self.levels]
        notional_cols = [f"{self.notional_prefix}_{i}" for i in self.levels]
        suffix = self.add_suffix or "_n"

        out_cols = []
        out_cols += [f"{c}{suffix}" for c in depth_cols]
        out_cols += [f"{c}{suffix}" for c in notional_cols]

        if self.add_aggregates:
            out_cols += [
                "total_depth_bid", "total_depth_ask", "total_depth",
                "total_notional_bid", "total_notional_ask", "total_notional",
                "depth_imbalance_ratio", "depth_imbalance_signed",
                "notional_imbalance_ratio", "notional_imbalance_signed",
            ]
        return out_cols
