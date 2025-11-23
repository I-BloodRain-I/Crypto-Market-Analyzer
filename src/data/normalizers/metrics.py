from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class MetricsNormalizer(BaseEstimator, TransformerMixin):
    """Transformer that normalizes trading metrics for model input.

    The transformer computes relative changes (percentage), rolling z-scores,
    and derived ratios from raw on-chain/trade metrics. It is designed to be
    used in scikit-learn pipelines and ColumnTransformer objects.

    The transform performs the following high-level steps:
    - percentage change for selected columns (with clipping and inf handling)
    - rolling mean/std and rolling z-score for selected columns
    - compute derived ratios (value per contract, average trade size) and z-scores
    - select a final list of feature columns and return a DataFrame limited to them

    Args:
        window: rolling window size used for means and standard deviations.
        clip_quantiles: tuple of (low, high) quantiles used to compute clipping
            thresholds per-feature during fit and applied in transform.
        eps: small constant to avoid division by zero.
        copy: whether to copy the input DataFrame or modify in place.

    Attributes:
        feature_cols: list of feature column names produced by transform.
    """

    def __init__(
        self, 
        window: int = 200, 
        clip_quantiles: Tuple[float, float] = (0.01, 0.99),
        eps: float = 1e-9,
        copy: bool = True
    ):
        self.window = window
        self.clip_quantiles = clip_quantiles
        self.eps = eps
        self.copy = copy

        self._feature_quantiles: Dict[str, Tuple[float, float]] = {}

        # will be set during fit()
        self._feature_names_out: Optional[List[str]] = None
        self._warmup_tail: Optional[pd.DataFrame] = None
        self._z_scaler = RobustScaler(quantile_range=(1.0, 99.0))

    def fit(self, X: pd.DataFrame, y=None):
        """Validate columns and prepare output feature names.

        Stores a warm-up tail of (window-1) rows to preserve rolling context
        across train/val/prod splits.

        Args:
            X: Input DataFrame containing required metric columns.
            y: Ignored (for sklearn compatibility).

        Returns:
            Fitted transformer.
        """
        self._validate_columns(X)

        n_tail = max(self.window - 1, 0)
        self._warmup_tail = X[
            ["taker_buy_volume", "sum_open_interest", "sum_open_interest_value", "count"]
        ].tail(n_tail).copy() if n_tail > 0 else None
        self._z_scaler.fit(X[[
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio"
        ]])

        self._feature_names_out = self._compute_future_column_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization and feature extraction to the input DataFrame.

        The input DataFrame is expected to contain the following columns at a
        minimum: ``taker_buy_volume``, ``sum_open_interest``,
        ``sum_open_interest_value``, and ``count``. The method will produce a
        set of derived features (percentage changes, z-scores and ratios) and
        return a DataFrame containing only the selected feature columns.

        Args:
            X: DataFrame with raw metric columns.

        Returns:
            DataFrame containing the final selected feature columns.
        """
        self._validate_columns(X)

        use_ctx = self._warmup_tail is not None and len(self._warmup_tail) > 0
        src = pd.concat([self._warmup_tail, X], axis=0) if use_ctx else X
        df = src.copy() if self.copy else src

        eps = self.eps

        df["taker_buy_volume"] += eps
        df["sum_open_interest"] += eps
        df["sum_open_interest_value"] += eps
        df["taker_buy_volume_rel"] = df["taker_buy_volume"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-3, 3)
        df["sum_open_interest_rel"] = df["sum_open_interest"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-3, 3) * 1000
        df["sum_open_interest_value_rel"] = df["sum_open_interest_value"].pct_change().replace([np.inf, -np.inf], np.nan).clip(-3, 3) * 1000

        for col in ["taker_buy_volume", "sum_open_interest", "sum_open_interest_value"]:
            mean_ = df[col].rolling(self.window, min_periods=10).mean()
            std_ = df[col].rolling(self.window, min_periods=10).std()
            z = (df[col] - mean_) / (std_ + eps)
            df[f"{col}_z"] = z

        df["oi_value_per_contract"] = df["sum_open_interest_value"] / (df["sum_open_interest"] + eps)
        mean_ = df["oi_value_per_contract"].rolling(self.window, min_periods=10).mean()
        std_ = df["oi_value_per_contract"].rolling(self.window, min_periods=10).std()
        z = (df["oi_value_per_contract"] - mean_) / (std_ + eps)
        df["oi_value_per_contract_z"] = z

        mean_ = df["count"].rolling(self.window, min_periods=10).mean()
        std_ = df["count"].rolling(self.window, min_periods=10).std()
        z = (df["count"] - mean_) / (std_ + eps)
        df["count_z"] = z

        df["avg_trade_size"] = df["taker_buy_volume"] / (df["count"] + eps)
        mean_ = df["avg_trade_size"].rolling(self.window, min_periods=10).mean()
        std_ = df["avg_trade_size"].rolling(self.window, min_periods=10).std()
        z = (df["avg_trade_size"] - mean_) / (std_ + eps)
        df["avg_trade_size_z"] = z

        df[[
            "count_toptrader_long_short_ratio_z",
            "sum_toptrader_long_short_ratio_z",
            "count_long_short_ratio_z",
            "sum_taker_long_short_vol_ratio_z"
        ]] = self._z_scaler.transform(df[[
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio"
        ]])

        if len(self._feature_quantiles) == 0:
            clip_low, clip_high = self.clip_quantiles

            for name in self._feature_names_out:
                low = df[name].quantile(clip_low)
                high = df[name].quantile(clip_high)
                self._feature_quantiles[name] = (low, high)

        for col, (low, high) in self._feature_quantiles.items():
            df[col] = df[col].clip(lower=low, upper=high)

        out = df.iloc[len(self._warmup_tail):].copy() if use_ctx else df

        n_tail = max(self.window - 1, 0)
        self._warmup_tail = src.tail(n_tail).copy() if n_tail > 0 else None

        return out[self._compute_future_column_names()]

    def get_feature_names_out(self, input_features: Optional[Iterable[str]] = None) -> np.ndarray:
        """Return names of features produced by transform()."""
        if self._feature_names_out is not None:
            return np.array(self._feature_names_out, dtype=object)
        return np.array(self._compute_future_column_names(), dtype=object)
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check that required columns are present in the input DataFrame."""
        required = [
            "taker_buy_volume", "sum_open_interest", "sum_open_interest_value", "count",
            "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
            "count_long_short_ratio", "sum_taker_long_short_vol_ratio"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _compute_future_column_names(self) -> List[str]:
        """Compute the list of output column names that transform() will produce."""
        return [
            "taker_buy_volume_rel",
            "sum_open_interest_rel",
            "sum_open_interest_value_rel",
            "taker_buy_volume_z",
            "sum_open_interest_z",
            "sum_open_interest_value_z",
            "oi_value_per_contract_z",
            "count_z",
            "avg_trade_size_z",
            "count_toptrader_long_short_ratio_z", 
            "sum_toptrader_long_short_ratio_z",
            "count_long_short_ratio_z", 
            "sum_taker_long_short_vol_ratio_z"
        ]