import logging
from itertools import product
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
 
logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Generate 5-class labels and related diagnostics from price and order book data.

    This class centralizes the full labeling pipeline:
    - compute multi-horizon log-returns
    - estimate volatility-driven base thresholds
    - adjust thresholds for imbalance and liquidity
    - produce hard 5-class labels and soft probabilistic labels
    - compute sample weights and diagnostic metrics

    Use `build_labels` as the primary entry point. The class keeps configuration
    parameters (horizons, alpha, imbalance/liquidity weights, etc.) so that
    the same settings are applied consistently across all steps.
    """

    def __init__(
        self,
        horizons: List[int],
        alpha: float = 1.0,
        k_imb: float = 0.3,
        k_liq: float = 0.2,
        gamma_strong: float = 2.0,
        span_sig: int = 100,
        band_cols: Tuple[int, ...] = (1, 2, 3, 4, 5),
        floor_bp: float = 2e-4,
        roll_liq: int = 200
    ) -> None:
        """
        Initialize the label generator with configuration parameters.

        Args:
            horizons: List of integer horizons to compute labels for.
            alpha: Volatility scaling multiplier for base thresholds.
            k_imb: Weight controlling how imbalance shifts thresholds.
            k_liq: Weight controlling liquidity-driven threshold adjustments.
            gamma_strong: Multiplier for strong-move (t2) thresholds.
            span_sig: EWMA span used to estimate volatility.
            band_cols: Order-book band identifiers used to compute depth.
            floor_bp: Minimum allowed threshold value.
            roll_liq: Rolling window size for liquidity estimation.

        The constructor stores parameters; no heavy computation is done here.
        """
        self.horizons = horizons
        self.alpha = alpha
        self.k_imb = k_imb
        self.k_liq = k_liq
        self.gamma_strong = gamma_strong
        self.span_sig = span_sig
        self.band_cols = list(band_cols)
        self.floor_bp = floor_bp
        self.roll_liq = roll_liq

    def logret_h(self, price: pd.Series) -> pd.DataFrame:
        """
        Compute multi-horizon log-returns.

        Args:
            price: Price time series indexed by timestamps.

        Returns:
            DataFrame with a column per horizon containing forward log-returns.
        """
        log_price = np.log(price)
        return pd.DataFrame(
            {horizon: log_price.shift(-horizon) - log_price for horizon in self.horizons}, 
            index=price.index
        )

    def ewma_sigma(self, price: pd.Series) -> pd.Series:
        """
        Estimate volatility using EWMA on log-returns.

        Args:
            price: Price time series.

        Returns:
            A series with the EWMA volatility estimate aligned to price index.
        """
        log_diff = np.log(price).diff().fillna(0)
        var_ewma = log_diff.pow(2).ewm(span=self.span_sig, adjust=False).mean()
        return np.sqrt(var_ewma)

    def _weights_for_bands(self, scheme: str = 'inv') -> Dict[int, float]:
        """
        Produce scalar weights for each order-book band.

        Args:
            scheme: Weighting scheme name. Supported: 'inv', 'inv2', others as uniform.

        Returns:
            Mapping from band id to scalar weight.
        """
        weights_map: Dict[int, float] = {}
        for band in self.band_cols:
            band_fraction = band / 100.0
            if scheme == 'inv2':
                weights_map[band] = 1.0 / (band_fraction**2)
            elif scheme == 'inv':
                weights_map[band] = 1.0 / band_fraction
            else:
                weights_map[band] = 1.0
        return weights_map

    def align_and_scale(
        self,
        depth_df: pd.DataFrame,
        lookback: int = 1000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract bids and asks from depth columns and apply point-in-time scaling.

        This corrects for persistent skews between bid/ask volumes using a
        rolling (expanding-limited) median of aggregated band volumes.

        Args:
            depth_df: DataFrame with depth_-5, depth_-4, ..., depth_5 columns.
            lookback: Rolling window size used to compute medians for scaling.

        Returns:
            Tuple of (bids_df, asks_df_scaled) with columns as band ids.
        """
        bid_cols = [col for col in depth_df.columns if col.startswith('depth_-')]
        ask_cols = [col for col in depth_df.columns if col.startswith('depth_') and not col.startswith('depth_-')]
        
        bids_df = depth_df[bid_cols].copy()
        asks_df = depth_df[ask_cols].copy()
        
        bids_df.columns = [int(col.replace('depth_-', '')) for col in bid_cols]
        asks_df.columns = [int(col.replace('depth_', '')) for col in ask_cols]

        bids_stack = bids_df.stack()
        asks_stack = asks_df.stack()

        bids_med = bids_stack.groupby(level=0).sum().rolling(lookback, min_periods=100).median().shift(1)
        asks_med = asks_stack.groupby(level=0).sum().rolling(lookback, min_periods=100).median().shift(1)

        scale = (bids_med / asks_med).fillna(1.0)
        scale = scale.reindex(asks_df.index, method='ffill').fillna(1.0)

        asks_df = asks_df.multiply(scale, axis=0)

        return bids_df, asks_df

    def depth_metrics(
        self,
        bids_df: pd.DataFrame,
        asks_df: pd.DataFrame,
        weight: str = 'inv'
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Compute depth, imbalance and density metrics from percent-band volumes.

        Args:
            bids_df: Bid bands DataFrame.
            asks_df: Ask bands DataFrame.
            weight: Weighting scheme for bands.

        Returns:
            wb: weighted bid depth
            wa: weighted ask depth
            depth: total weighted depth
            imbalance: normalized imbalance in [-1, 1]
            density: density proxy (depth normalized by weight sum)
        """
        band_weights = self._weights_for_bands(weight)
        weighted_bids = sum(band_weights[p] * bids_df[p] for p in self.band_cols)
        weighted_asks = sum(band_weights[p] * asks_df[p] for p in self.band_cols)

        total_depth = weighted_bids + weighted_asks
        imbalance = (weighted_bids - weighted_asks) / (weighted_bids + weighted_asks).replace(0, np.nan)
        imbalance = imbalance.clip(-1, 1).fillna(0)
        
        weight_sum = sum(band_weights[p] for p in self.band_cols)
        density = total_depth / weight_sum
        
        return weighted_bids, weighted_asks, total_depth, imbalance, density

    def base_thresholds(self, sigma: pd.Series) -> Dict[int, pd.Series]:
        """
        Build volatility-proportional base thresholds for each horizon.

        Args:
            sigma: Estimated volatility series.

        Returns:
            Mapping from horizon to base threshold series.
        """
        return {horizon: (self.alpha * sigma * np.sqrt(horizon)).clip(lower=self.floor_bp) 
                          for horizon in self.horizons}

    def asym_thresholds(
        self,
        base_th: Dict[int, pd.Series],
        imbalance: pd.Series,
        density: pd.Series,
        roll: int = 200
    ) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series], Dict[int, pd.Series]]:
        """
        Create asymmetric up/down thresholds using imbalance and liquidity.

        Args:
            base_th: Base threshold mapping per horizon.
            imbalance: Imbalance series.
            density: Liquidity density series.
            roll: Rolling window used for liquidity normalization.

        Returns:
            t1_up, t1_down, t2_up, t2_down: dicts of series representing threshold bands.
        """
        liquidity_ratio = (density / density.rolling(roll, min_periods=1).median().shift(1)).clip(0.25, 4.0)
        t1_up, t1_down, t2_up, t2_down = {}, {}, {}, {}

        for horizon, base_threshold_series in base_th.items():
            adj_up = (1 - self.k_imb * imbalance) * (1 - self.k_liq * (liquidity_ratio - 1) / 2)
            adj_dn = (1 + self.k_imb * imbalance) * (1 - self.k_liq * (liquidity_ratio - 1) / 2)
            thresh_up = (base_threshold_series * adj_up).clip(lower=self.floor_bp)
            thresh_dn = (base_threshold_series * adj_dn).clip(lower=self.floor_bp)

            t1_up[horizon] = thresh_up
            t1_down[horizon] = thresh_dn
            t2_up[horizon] = thresh_up * self.gamma_strong
            t2_down[horizon] = thresh_dn * self.gamma_strong

        return t1_up, t1_down, t2_up, t2_down

    def label_5class(
        self,
        returns_df: pd.DataFrame,
        t1_up: Dict[int, pd.Series],
        t1_down: Dict[int, pd.Series],
        t2_up: Dict[int, pd.Series],
        t2_down: Dict[int, pd.Series]
    ) -> pd.DataFrame:
        """
        Convert returns into discrete 5-class labels using thresholds.

        Args:
            returns_df: DataFrame with returns for each horizon.
            t1_up/t1_down/t2_up/t2_down: Threshold mappings produced by asym_thresholds.

        Returns:
            DataFrame of integer labels in {-2, -1, 0, 1, 2} indexed by timestamp.
        """
        labels_dict = {}
        for horizon in returns_df.columns:
            returns_series = returns_df[horizon]
            labels_series = pd.Series(0, index=returns_series.index, dtype=int)
            labels_series[returns_series >= t2_up[horizon]] = 2
            labels_series[(returns_series >= t1_up[horizon]) & (returns_series < t2_up[horizon])] = 1
            labels_series[returns_series <= -t2_down[horizon]] = -2
            labels_series[(returns_series <= -t1_down[horizon]) & (returns_series > -t2_down[horizon])] = -1
            labels_dict[horizon] = labels_series
        return pd.DataFrame(labels_dict)

    @staticmethod
    def _softmax_mat(matrix: np.ndarray) -> np.ndarray:
        """
        Numerically-stable softmax applied row-wise with NaN handling.

        Args:
            M: 2D numpy array of logits (rows = samples, cols = classes).

        Returns:
            Array of same shape with rows summing to 1. Rows that were all-NaN
            are converted to a uniform distribution.
        """
        all_nan_rows = np.isnan(matrix).all(axis=1)

        if all_nan_rows.any():
            matrix = matrix.copy()
            matrix[all_nan_rows, :] = 0

        matrix = matrix - np.nanmax(matrix, axis=1, keepdims=True)
        exp_mat = np.exp(np.nan_to_num(matrix, nan=-1e9))
        sum_exp = np.sum(exp_mat, axis=1, keepdims=True)

        softmax = exp_mat / np.where(sum_exp == 0, 1, sum_exp)

        if all_nan_rows.any():
            softmax[all_nan_rows, :] = 1.0 / matrix.shape[1]

        return softmax

    def soft_probs_5class(
        self, 
        returns_df: pd.DataFrame, 
        t1_up: Dict[int, pd.Series], 
        t1_down: Dict[int, pd.Series], 
        t2_up: Dict[int, pd.Series], 
        t2_down: Dict[int, pd.Series], 
        tau: float = 0.6, 
        eps: float = 1e-6
    ) -> Dict[int, pd.DataFrame]:
        """
        Produce soft class probabilities for the 5-class scheme.

        The method constructs five logits per sample based on distances to
        thresholds, then applies a softmax to obtain probabilistic labels.

        Args:
            returns_df: Returns DataFrame.
            t1_up/t1_down/t2_up/t2_down: Threshold mappings.
            tau: Temperature parameter controlling softness.
            eps: Small floor to avoid division by zero.

        Returns:
            Dictionary mapping horizon to a DataFrame of probabilities
            with columns ['p_-2','p_-1','p_0','p_1','p_2'].
        """
        P: Dict[int, pd.DataFrame] = {}
        for horizon in returns_df.columns:
            returns_series = returns_df[horizon]

            t1_up_series = t1_up[horizon].clip(lower=eps)
            t1_down_series = t1_down[horizon].clip(lower=eps)
            t2_up_series = t2_up[horizon].clip(lower=eps)
            t2_down_series = t2_down[horizon].clip(lower=eps)

            logit_neg2 = ((-returns_series) - t2_down_series) / tau
            logit_neg1 = ((-returns_series) - t1_down_series) / tau
            logit_zero = -(np.abs(returns_series) / pd.concat([t1_up_series, t1_down_series], axis=1).min(axis=1)) / tau
            logit_pos1 = (returns_series - t1_up_series) / tau
            logit_pos2 = (returns_series - t2_up_series) / tau
            logit_matrix = pd.concat([logit_neg2, logit_neg1, logit_zero, logit_pos1, logit_pos2], axis=1).values

            probs_arr = self._softmax_mat(logit_matrix)
            P[horizon] = pd.DataFrame(
                probs_arr, 
                index=returns_series.index, 
                columns=['p_-2', 'p_-1', 'p_0', 'p_1', 'p_2']
            ).fillna(0)

        return P

    def concurrency(self, labels_df: pd.DataFrame) -> pd.Series:
        """
        Count number of non-zero labels per timestamp (concurrency).

        Args:
            labels_df: DataFrame of integer labels.

        Returns:
            Series with the concurrency count (minimum 1).
        """
        return labels_df.replace(0, np.nan).notna().sum(axis=1).clip(lower=1)

    def weights(
        self,
        price: pd.Series,
        labels_df: pd.DataFrame,
        density: Optional[pd.Series] = None,
        sig_floor: float = 1e-4, 
        w_clip: Tuple[float, float] = (0.01, 50.0)
    ) -> pd.Series:
        """
        Compute sample weights used for training/aggregation.

        We combine volatility-based scaling (inverse sigma) with concurrency
        normalization and optional density adjustment.

        Args:
            price: Price series.
            labels_df: Hard labels DataFrame.
            density: Optional liquidity density to adjust weights.
            sig_floor: Minimum sigma floor.
            w_clip: Min/max clipping for final weights.

        Returns:
            Series of per-timestamp weights.
        """
        sigma_series = self.ewma_sigma(price).reindex(labels_df.index).clip(lower=sig_floor)
        concurrency_counts = self.concurrency(labels_df)
        weights_series = (1 / concurrency_counts) * (1 / sigma_series)

        if density is not None:
            density_series = density.reindex(labels_df.index)
            density_adj = (density_series / density_series.rolling(200, min_periods=1)\
                           .median().shift(1)).clip(0.5, 2.0)
            weights_series = weights_series * density_adj

        weights_series = weights_series.clip(
            lower=w_clip[0], upper=w_clip[1]
        ).replace([np.inf, -np.inf], np.nan).fillna(w_clip[0])

        return weights_series

    def build_labels(
        self, 
        price: pd.Series, 
        depth_df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame], 
        pd.Series, Dict[str, Union[Dict[int, pd.Series], pd.Series]]
    ]:
        """
        End-to-end pipeline to produce returns, labels, probabilities and weights.

        Args:
            price: Price series.
            depth_df: DataFrame with depth_-5, depth_-4, ..., depth_5 columns.

        Returns:
            returns_df: Multi-horizon returns DataFrame.
            labels_5: Hard labels DataFrame (5-class).
            probs_5: Dict of soft-probability DataFrames indexed by horizon.
            sample_weights: Series of per-timestamp sample weights.
            meta: Dictionary with threshold and diagnostic series.
        """
        if price.empty: 
            raise ValueError("Empty price series")
        if depth_df.empty: 
            raise ValueError("Empty order book")
        
        required_depth_cols = [f'depth_{i}' for i in self.band_cols] + [f'depth_-{i}' for i in self.band_cols]
        if not all(col in depth_df.columns for col in required_depth_cols):
            raise ValueError("Missing depth columns")

        bids_df, asks_df = self.align_and_scale(depth_df)
        common_index = price.index.intersection(bids_df.index).intersection(asks_df.index)
        price = price.reindex(common_index).dropna()

        returns_df = self.logret_h(price)
        sigma_series = self.ewma_sigma(price)

        valid_mask = returns_df.notna()
        valid_any = valid_mask.any(axis=1)

        _, _, _, imbalance, density = self.depth_metrics(
            bids_df.reindex(common_index), 
            asks_df.reindex(common_index), 
            weight='inv'
        )

        base_th = self.base_thresholds(sigma_series)
        t1_up, t1_down, t2_up, t2_down = self.asym_thresholds(base_th, imbalance, density, roll=self.roll_liq)

        labels_5 = self.label_5class(returns_df, t1_up, t1_down, t2_up, t2_down)
        labels_5 = labels_5.where(valid_mask)
        probs_5 = self.soft_probs_5class(returns_df, t1_up, t1_down, t2_up, t2_down, tau=0.6)
        probs_5 = {h: df.where(valid_mask[h]) for h, df in probs_5.items()}
        sample_weights = self.weights(price, labels_5, density=density, sig_floor=1e-4, w_clip=(0.01, 50.0))
        sample_weights = sample_weights.where(valid_any)

        meta = {
            't1_up': t1_up,
            't1_down': t1_down,
            't2_up': t2_up,
            't2_down': t2_down,
            'imbalance': imbalance,
            'density': density
        }

        return returns_df, labels_5, probs_5, sample_weights, meta

    def calibrate(
        self,
        price: pd.Series,
        returns_df: pd.DataFrame,
        depth_df: pd.DataFrame,
        target: Tuple[float, ...] = (0.15, 0.35, 0.0, 0.35, 0.15),
        alpha_grid: Optional[np.ndarray] = None,
        gamma_grid: Optional[np.ndarray] = None,
        n_jobs: int = -1,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Calibrate alpha and gamma using grid search to match a target class distribution.

        Args:
            price: Price series used to compute volatility and align book data.
            returns_df: Precomputed returns DataFrame for the target horizons.
            depth_df: DataFrame with depth_-5, depth_-4, ..., depth_5 columns.
            target: Desired distribution over classes [-2,-1,0,1,2].
            alpha_grid: Optional array of alpha values to search.
            gamma_grid: Optional array of gamma values to search.
            n_jobs: Number of parallel jobs for search.
            verbose: If True, print calibration diagnostics.

        Returns:
            Mapping with 'alpha' and 'gamma' found to minimize distribution error.
        """
        if alpha_grid is None:
            alpha_grid = np.linspace(0.4, 2.0, 9)
        if gamma_grid is None:
            gamma_grid = np.linspace(1.4, 2.6, 7)

        bids_df, asks_df = self.align_and_scale(depth_df)
        common_index = price.index.intersection(bids_df.index).intersection(asks_df.index)
        price = price.reindex(common_index).dropna()
        returns_df = returns_df.reindex(common_index)

        sigma_series = self.ewma_sigma(price)
        _, _, _, imbalance, density = self.depth_metrics(
            bids_df.reindex(common_index), 
            asks_df.reindex(common_index), 
            weight='inv'
        )

        def _eval_params(alpha: float, gamma: float):
            base_th = {horizon: (alpha * sigma_series * np.sqrt(horizon)).clip(lower=self.floor_bp) 
                                for horizon in self.horizons}
            t1_up, t1_down, t2_up, t2_down = self.asym_thresholds(
                base_th, imbalance, density, roll=self.roll_liq
            )

            temp_gen = LabelGenerator(
                horizons=self.horizons,
                alpha=alpha,
                gamma_strong=gamma,
                k_imb=self.k_imb,
                k_liq=self.k_liq,
                span_sig=self.span_sig,
                band_cols=self.band_cols,
                floor_bp=self.floor_bp
            )
            labels_5 = temp_gen.label_5class(returns_df, t1_up, t1_down, t2_up, t2_down).dropna()

            err = 0.0
            tgt = np.array(target)
            for horizon in labels_5.columns:
                emp = labels_5[horizon].value_counts(normalize=True).reindex([-2, -1, 0, 1, 2]).fillna(0).values
                err += np.abs(emp - tgt).sum()
            return (alpha, gamma, err)

        grid = list(product(alpha_grid, gamma_grid))
        res = Parallel(n_jobs=n_jobs)(delayed(_eval_params)(a, g) for a, g in grid)
        best = min(res, key=lambda x: x[2])

        if verbose:
            alpha, gamma = best[0], best[1]
            base_th = {horizon: (alpha * sigma_series * np.sqrt(horizon)).clip(lower=self.floor_bp) 
                                 for horizon in self.horizons}
            t1_up, t1_down, t2_up, t2_down = self.asym_thresholds(base_th, imbalance, density, roll=self.roll_liq)

            temp_gen = LabelGenerator(
                horizons=self.horizons,
                alpha=alpha,
                gamma_strong=gamma,
                k_imb=self.k_imb,
                k_liq=self.k_liq,
                span_sig=self.span_sig,
                band_cols=self.band_cols,
                floor_bp=self.floor_bp
            )
            labels_5 = temp_gen.label_5class(returns_df, t1_up, t1_down, t2_up, t2_down).dropna()

            logger.info("Calibration results: alpha=%.3f, gamma=%.3f", alpha, gamma)
            logger.info("Final error: %.4f", best[2])
            for horizon in labels_5.columns:
                dist = labels_5[horizon].value_counts(normalize=True).reindex([-2, -1, 0, 1, 2]).fillna(0)
                logger.info("H=%s: %s", horizon, dist.values)
            logger.info("Target: %s", target)

        return {'alpha': best[0], 'gamma': best[1]}