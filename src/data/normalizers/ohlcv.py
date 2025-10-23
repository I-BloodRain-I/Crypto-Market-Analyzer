from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OHLCVNormalizer(BaseEstimator, TransformerMixin):
    """
    Engineer and normalize OHLCV candlestick features for sklearn pipelines.

    This transformer constructs relative price features, volume metrics, and temporal
    patterns from OHLCV data, then applies z-score normalization.
    All engineered features are then z-score normalized using statistics computed from
    training data: ``(feature - mean) / std``. Normalized features have ``_z`` suffix.
    The transformer handles NaN and infinite values by replacing them with NaN during
    computation and filling with 0.0 in final output.

    Args:
        o_col: Column name for open prices
        h_col: Column name for high prices
        l_col: Column name for low prices
        c_col: Column name for close prices
        v_col: Column name for volume
        lags: Iterable of integers specifying lag periods (e.g., [2, 5, 10]).
            None disables lagged features. Only positive integers are used.
        add_raw_relative: If True, output includes both normalized (``*_z``) and
            raw engineered features. If False, only normalized features.
        eps: Small constant added to denominators and inside logarithms for
            numerical stability (prevents division by zero and log(0)).
        clip_pct_change: Absolute clipping threshold for volume percent change.
            None disables clipping. Default 10.0 clips at ±1000%.

    Notes:
        Initial rows contain NaN due to shifting (``shift(1)`` for base features,
        ``shift(k)`` for lags). Final output fills these with 0.0. Infinite values
        from divisions are replaced with NaN before statistics computation, ensuring
        robust normalization even with extreme price movements or zero volumes.
    """

    def __init__(
        self,
        o_col: str = "open",
        h_col: str = "high",
        l_col: str = "low",
        c_col: str = "close",
        v_col: str = "volume",
        lags: Optional[Iterable[int]] = (2, 3, 5, 10, 15),
        add_raw_relative: bool = False,
        eps: float = 1e-12,
        clip_pct_change: Optional[float] = 10.0,
    ):
        self.o_col = o_col
        self.h_col = h_col
        self.l_col = l_col
        self.c_col = c_col
        self.v_col = v_col
        self.lags = lags
        self.add_raw_relative = add_raw_relative
        self.eps = eps
        self.clip_pct_change = clip_pct_change

        self.feature_stats_: Dict[str, Dict[str, float]] = {}
        self.fitted_: bool = False
        self._last_feature_order_: List[str] = []

    def _zscore_base_features(self) -> List[str]:
        """
        Get list of base feature names that will be z-score normalized.
        
        Returns:
            List of base feature column names
        """
        return [
            "open_rel_prev", "high_rel_prev", "low_rel_prev", "close_rel_prev",
            "body_rel_open", "wick_top_rel_open", "wick_bot_rel_open", "hl_range_rel",
            "true_range_rel",
            "ret_1", "gap_open",
            "volume_log", "volume_pct", "dollar_volume_log",
            "price_impact_vol",
        ]

    def _zscore_snapshot_lagged(self) -> List[str]:
        """
        Get list of lagged snapshot feature names that will be z-score normalized.
        
        Returns:
            List of lagged snapshot feature column names, empty if no lags configured
        """
        if not getattr(self, "_lags_", []):
            return []
        base_snapshots = [
            "body_rel_open", "wick_top_rel_open", "wick_bot_rel_open",
            "hl_range_rel", "true_range_rel",
            "volume_log", "volume_pct", "dollar_volume_log",
            "price_impact_vol",
        ]
        names = []
        for k in self._lags_:
            for b in base_snapshots:
                names.append(f"{b}_lag{k}")
        return names

    def _zscore_relative_to_past(self) -> List[str]:
        """
        Get list of relative-to-past feature names that will be z-score normalized.
        
        Returns:
            List of relative-to-past feature column names, empty if no lags configured
        """
        if not getattr(self, "_lags_", []):
            return []
        names = []
        for k in self._lags_:
            names.extend([
                f"ret_{k}",
                f"gap_open_{k}",
                f"dclose_{k}",
                f"dopen_{k}",
                f"dhigh_{k}",
                f"dlow_{k}",
            ])
        return names

    def _zscore_feature_list(self) -> List[str]:
        """
        Get complete list of all features that will be z-score normalized.
        
        Returns:
            Combined list of base, snapshot lagged, and relative-to-past features
        """
        return self._zscore_base_features() + \
               self._zscore_snapshot_lagged() + \
               self._zscore_relative_to_past()

    def _build_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct base OHLCV features without lags.
        
        Creates relative price features, body/wick features, volume features,
        and price impact features. All features are relative to avoid look-ahead bias.

        Args:
            df: DataFrame containing OHLCV columns

        Returns:
            DataFrame with original columns plus new base features
        """
        eps = float(self.eps)
        df = df.copy()

        o = df[self.o_col].astype(float)
        h = df[self.h_col].astype(float)
        l = df[self.l_col].astype(float)
        c = df[self.c_col].astype(float)
        v = df[self.v_col].astype(float)

        prev_c = c.shift(1)

        inv_prev_c = 1.0 / (np.abs(prev_c) + eps)
        df["open_rel_prev"]  = o * inv_prev_c - 1.0
        df["high_rel_prev"]  = h * inv_prev_c - 1.0
        df["low_rel_prev"]   = l * inv_prev_c - 1.0
        df["close_rel_prev"] = c * inv_prev_c - 1.0

        denom_o = np.abs(o) + eps
        df["body_rel_open"]     = (c - o) / denom_o
        df["wick_top_rel_open"] = (h - np.maximum(o, c)) / denom_o
        df["wick_bot_rel_open"] = (np.minimum(o, c) - l) / denom_o
        df["hl_range_rel"]      = (h - l) / denom_o

        tr = np.maximum.reduce([(h - l), (h - prev_c).abs(), (l - prev_c).abs()])
        df["true_range_rel"] = tr / (np.abs(prev_c) + eps)

        df["ret_1"]    = np.log((c + eps) / (prev_c + eps))
        df["gap_open"] = (o / (prev_c + eps)) - 1.0

        v_pos = np.maximum(v, 0.0)
        df["volume_log"] = np.log1p(v_pos)

        vol_pct = v.pct_change()
        if self.clip_pct_change is not None:
            vol_pct = vol_pct.clip(-self.clip_pct_change, self.clip_pct_change)
        df["volume_pct"] = vol_pct

        dollar_vol = v * c
        df["dollar_volume_log"] = np.log1p(np.maximum(dollar_vol, 0.0))

        df["price_impact_vol"] = df["hl_range_rel"] * df["volume_log"]

        for col in [
            "open_rel_prev", "high_rel_prev", "low_rel_prev", "close_rel_prev",
            "body_rel_open", "wick_top_rel_open", "wick_bot_rel_open", "hl_range_rel",
            "true_range_rel", "ret_1", "gap_open",
            "volume_log", "volume_pct", "dollar_volume_log", "price_impact_vol",
        ]:
            s = df[col].astype(float)
            df[col] = s.replace([np.inf, -np.inf], np.nan)

        return df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged versions of snapshot features and compute relative-to-past features.
        
        Creates lagged copies of key features and computes returns, gaps, and deltas
        relative to past timesteps specified by lags.

        Args:
            df: DataFrame with base features already computed

        Returns:
            DataFrame with original features plus lagged and relative-to-past features
        """
        df = df.copy()
        if not self._lags_:
            return df

        snapshot_cols = [
            "body_rel_open", "wick_top_rel_open", "wick_bot_rel_open",
            "hl_range_rel", "true_range_rel",
            "volume_log", "volume_pct", "dollar_volume_log",
            "price_impact_vol",
        ]
        for k in self._lags_:
            for col in snapshot_cols:
                df[f"{col}_lag{k}"] = df[col].shift(k)

        o = df[self.o_col].astype(float)
        h = df[self.h_col].astype(float)
        l = df[self.l_col].astype(float)
        c = df[self.c_col].astype(float)
        eps = float(self.eps)

        for k in self._lags_:
            c_k = c.shift(k)
            df[f"ret_{k}"] = np.log((c + eps) / (c_k + eps))
            df[f"gap_open_{k}"] = (o / (c_k + eps)) - 1.0

            o_k = o.shift(k); h_k = h.shift(k); l_k = l.shift(k)
            df[f"dclose_{k}"] = (c - c_k) / (np.abs(c_k) + eps)
            df[f"dopen_{k}"]  = (o - o_k) / (np.abs(o_k) + eps)
            df[f"dhigh_{k}"]  = (h - h_k) / (np.abs(h_k) + eps)
            df[f"dlow_{k}"]   = (l - l_k) / (np.abs(l_k) + eps)

        for col in df.columns:
            if col in [self.o_col, self.h_col, self.l_col, self.c_col, self.v_col]:
                continue
            if df[col].dtype.kind in "fc":
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        return df

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and standard deviation for each feature to be z-score normalized.
        
        Uses nanmean and nanstd to handle NaN values properly during statistics computation.

        Args:
            df: DataFrame containing all computed features

        Returns:
            Dictionary mapping feature names to their statistics (mean, std)
        """
        stats: Dict[str, Dict[str, float]] = {}
        for col in self._zscore_feature_list():
            if col in df.columns:
                x = df[col].astype(float).values
                mean = float(np.nanmean(x))
                std = float(np.nanstd(x))
                std = std if std > float(self.eps) else 1.0
                stats[col] = {"mean": mean, "std": std}
        return stats

    def _apply_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply z-score normalization to features using fitted statistics.
        
        Creates new columns with '_z' suffix containing z-score normalized values.

        Args:
            df: DataFrame containing raw feature values

        Returns:
            DataFrame with original features plus z-score normalized versions
        """
        out = df.copy()
        for col, ms in self.feature_stats_.items():
            if col in out.columns:
                out[f"{col}_z"] = (out[col].astype(float) - ms["mean"]) / ms["std"]
        return out

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the normalizer by computing statistics on the training data.
        
        Validates lags parameter, builds all features, and computes their statistics
        for later z-score normalization.

        Args:
            X: Training DataFrame containing OHLCV columns
            y: Ignored, present for sklearn compatibility

        Returns:
            self
        """
        if self.lags is None:
            self._lags_ = []
        else:
            _lags = []
            for k in self.lags:
                try:
                    k = int(k)
                    if k >= 1:
                        _lags.append(k)
                except Exception:
                    continue
            self._lags_ = sorted(set(_lags))

        feats = self._build_base_features(X)
        feats = self._add_lagged_features(feats)
        self.feature_stats_ = self._compute_stats(feats)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform data by computing features and applying z-score normalization.
        
        Uses statistics computed during fit() to normalize features. Returns only
        z-score normalized features by default, optionally includes raw features.

        Args:
            X: DataFrame containing OHLCV columns to transform

        Returns:
            DataFrame with z-score normalized features (and raw features if configured)
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        feats = self._build_base_features(X)
        feats = self._add_lagged_features(feats)
        out = self._apply_zscore(feats)

        keep = [c for c in out.columns if c.endswith("_z")]

        if self.add_raw_relative:
            for c in out.columns:
                if c not in keep and c not in [self.o_col, self.h_col, self.l_col, self.c_col, self.v_col]:
                    keep.append(c)

        seen, ordered = set(), []
        for c in keep:
            if c in out.columns and c not in seen:
                ordered.append(c); seen.add(c)

        final = out[ordered].copy()
        final = final.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self._last_feature_order_ = ordered
        return final

    def fit_transform(self, X: pd.DataFrame, y=None):
        """
        Fit the normalizer and transform data in one step.

        Args:
            X: Training DataFrame containing OHLCV columns
            y: Ignored, present for sklearn compatibility

        Returns:
            Transformed DataFrame with normalized features
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, _) -> List[str]:
        """
        Get output feature names in the order they appear in transform output.
        
        Returns feature names from last transform call, or expected feature names
        if transform has not been called yet.

        Args:
            _: Input feature names (ignored, present for sklearn compatibility)

        Returns:
            List of output feature names in order
        """
        if self._last_feature_order_:
            return list(self._last_feature_order_)
        cols = [f"{c}_z" for c in self._zscore_feature_list()]
        if self.add_raw_relative:
            cols += self._zscore_feature_list()
        seen, ordered = set(), []
        for c in cols:
            if c not in seen:
                ordered.append(c); seen.add(c)
        return ordered

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get computed feature statistics (mean and std) used for normalization.

        Returns:
            Dictionary mapping feature names to their computed statistics
        """
        return {k: dict(v) for k, v in self.feature_stats_.items()}
