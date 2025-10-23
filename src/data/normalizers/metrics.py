import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        clip_val: absolute value to clip computed z-scores to.
        eps: small constant to avoid division by zero.

    Attributes:
        feature_cols: list of feature column names produced by transform.
    """

    def __init__(self, window: int = 200, clip_val: float = 5.0, eps: float = 1e-9):
        """Initialize the normalizer.

        Args:
            window: rolling window size used for means and standard deviations.
            clip_val: absolute value to clip computed z-scores to.
            eps: small constant to avoid division by zero.
        """
        self.window = window
        self.clip_val = clip_val
        self.eps = eps
        self.feature_cols = []

    def fit(self, X, y=None):
        """No-op fit to comply with scikit-learn's transformer API.

        Returns:
            self
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization and feature extraction to the input DataFrame.

        The input DataFrame is expected to contain the following columns at a
        minimum: ``taker_buy_volume``, ``sum_open_interest``,
        ``sum_open_interest_value``, and ``count``. The method will produce a
        set of derived features (percentage changes, z-scores and ratios) and
        return a DataFrame containing only the selected feature columns.

        Args:
            df: DataFrame with raw metric columns.

        Returns:
            DataFrame containing the final selected feature columns.

        Raises:
            KeyError: if required input columns are missing from ``df``.
        """
        df = df.copy()

        df["taker_buy_volume_rel"] = (
            df["taker_buy_volume"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(-3, 3)
        )

        df["sum_open_interest_rel"] = (
            df["sum_open_interest"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(-3, 3)
        ) * 100

        df["sum_open_interest_value_rel"] = (
            df["sum_open_interest_value"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .clip(-3, 3)
        ) * 100

        for col in ["taker_buy_volume", "sum_open_interest", "sum_open_interest_value"]:
            mean_ = df[col].rolling(self.window, min_periods=10).mean()
            std_ = df[col].rolling(self.window, min_periods=10).std()
            df[f"{col}_z"] = ((df[col] - mean_) / (std_ + self.eps)).clip(-self.clip_val, self.clip_val)

        df["oi_value_per_contract"] = df["sum_open_interest_value"] / (df["sum_open_interest"] + self.eps)
        df["oi_value_per_contract_z"] = (
            (df["oi_value_per_contract"] - df["oi_value_per_contract"].rolling(self.window).mean())
            / (df["oi_value_per_contract"].rolling(self.window).std() + self.eps)
        ).clip(-self.clip_val, self.clip_val)

        df["count_z"] = (
            (df["count"] - df["count"].rolling(self.window).mean())
            / (df["count"].rolling(self.window).std() + self.eps)
        ).clip(-self.clip_val, self.clip_val)

        df["avg_trade_size"] = (
            df["taker_buy_volume"] / (df["count"] + self.eps)
        ).replace([np.inf, -np.inf], np.nan).fillna(0)

        df["avg_trade_size_z"] = (
            (df["avg_trade_size"] - df["avg_trade_size"].rolling(self.window).mean())
            / (df["avg_trade_size"].rolling(self.window).std() + self.eps)
        ).clip(-self.clip_val, self.clip_val)

        self.feature_cols = [
            "taker_buy_volume_rel",
            "sum_open_interest_rel",
            "sum_open_interest_value_rel",
            "taker_buy_volume_z",
            "sum_open_interest_z",
            "sum_open_interest_value_z",
            "oi_value_per_contract_z",
            "count_z",
            "avg_trade_size_z",
        ]

        return df[self.feature_cols]

    def get_feature_names_out(self, _):
        """Return the names of the features produced by transform.

        The method is implemented to be compatible with scikit-learn's
        ColumnTransformer and Pipeline utilities.

        Args:
            _: ignored argument (kept for sklearn compatibility)

        Returns:
            numpy.ndarray containing the feature names as objects.
        """
        return np.asarray(self.feature_cols, dtype=object)