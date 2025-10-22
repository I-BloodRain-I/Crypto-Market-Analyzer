"""Helpers for threshold-quality checks for 5-class labeling.

Provides independent, testable functions that evaluate properties such as
separation, overlap, threshold stability, purity versus thresholds,
monotonicity, extremes metrics, noise level, and per-class statistics.

Each function accepts pandas Series and returns scalars or small dicts
suitable for unit tests and diagnostic reporting.
"""
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


def cohens_d(series_a: pd.Series, series_b: pd.Series, min_samples: int = 2) -> float:
    """Compute Cohen's d between two samples.

    Args:
        series_a: first sample
        series_b: second sample
        min_samples: minimum number of observations required per sample

    Returns:
        Cohen's d or nan if insufficient data or zero pooled variance.
    """
    series_a, series_b = series_a.dropna(), series_b.dropna()
    if len(series_a) < min_samples or len(series_b) < min_samples:
        return np.nan
    std_a, std_b = series_a.std(ddof=1), series_b.std(ddof=1)
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2.0)
    if pooled_std == 0:
        return np.nan
    return float((series_b.mean() - series_a.mean()) / pooled_std)


def threshold_cv(series: pd.Series, stability_use_mad: bool = True) -> float:
    """Coefficient-of-variation for a threshold series using MAD or std.

    Args:
        series: series of threshold values
        stability_use_mad: use MAD*1.4826 (robust) when True, otherwise std

    Returns:
        CV (scale / |median|) or nan for empty/zero-median input.
    """
    series = series.dropna()
    if series.empty:
        return np.nan
    median_val = np.median(series)
    median_abs = np.abs(median_val)
    if median_abs == 0:
        return np.nan
    if stability_use_mad:
        mad_val = np.median(np.abs(series - median_val))
        scale_val = 1.4826 * mad_val
    else:
        scale_val = series.std(ddof=1)
    return float(scale_val / median_abs) if median_abs != 0 else np.nan


def overlap_score(series_returns: pd.Series, series_labels: pd.Series, overlap_trim_q: float = 0.01) -> float:
    """Symmetric, trimmed overlap score computed only over present classes.

    Args:
        series_returns: returns series
        series_labels: integer label series
        overlap_trim_q: tail trimming quantile used for robustness

    Returns:
        Overlap score in [0,1] or nan if not enough class pairs.
    """
    classes = [-2, -1, 0, 1, 2]
    overlap_accum = 0.0
    pair_count = 0
    present = [c for c in classes if (series_labels == c).any()]
    for i, class1 in enumerate(present):
        returns_class1 = series_returns[series_labels == class1]
        if returns_class1.empty:
            continue
        q_class1_low, q_class1_high = returns_class1.quantile([overlap_trim_q, 1 - overlap_trim_q])
        for class2 in present[i + 1:]:
            returns_class2 = series_returns[series_labels == class2]
            if returns_class2.empty:
                continue
            q_class2_low, q_class2_high = returns_class2.quantile([overlap_trim_q, 1 - overlap_trim_q])
            prop_c1_in_c2 = ((returns_class1 >= q_class2_low) & (returns_class1 <= q_class2_high)).mean()
            prop_c2_in_c1 = ((returns_class2 >= q_class1_low) & (returns_class2 <= q_class1_high)).mean()
            overlap_accum += 0.5 * (float(prop_c1_in_c2) + float(prop_c2_in_c1))
            pair_count += 1
    return (overlap_accum / pair_count) if pair_count > 0 else np.nan


def purity_vs_thresholds(
    series_returns: pd.Series,
    series_labels: pd.Series,
    t1_up_series: pd.Series,
    t1_down_series: pd.Series
) -> Tuple[float, float]:
    """Compute purity for extreme classes vs provided t1 thresholds.

    Args:
        series_returns: returns series
        series_labels: integer label series
        t1_up_series: t1 up series (aligned to returns index)
        t1_down_series: t1 down series (aligned to returns index)

    Returns:
        (purity_up, purity_down) where each is fraction of +2/-2 satisfying threshold.
    """
    idx_up = series_labels.index[series_labels == 2]
    idx_down = series_labels.index[series_labels == -2]
    if len(idx_up) > 0:
        try:
            purity_up = float((series_returns.loc[idx_up] >= t1_up_series.loc[idx_up]).mean())
        except Exception:
            purity_up = float((series_returns.loc[idx_up] >= t1_up_series.reindex(idx_up)).mean())
        if np.isnan(purity_up):
            purity_up = 0.0
    else:
        purity_up = 0.0
    if len(idx_down) > 0:
        try:
            purity_down = float((series_returns.loc[idx_down] <= -t1_down_series.loc[idx_down]).mean())
        except Exception:
            purity_down = float((series_returns.loc[idx_down] <= -t1_down_series.reindex(idx_down)).mean())
        if np.isnan(purity_down):
            purity_down = 0.0
    else:
        purity_down = 0.0
    return purity_up, purity_down


def separation_ratio(series_returns: pd.Series, series_labels: pd.Series) -> float:
    """Separation ratio between -2 and +2 using pooled std.

    Args:
        series_returns: returns series
        series_labels: integer label series

    Returns:
        Separation ratio or nan if insufficient samples.
    """
    returns_m2, returns_p2 = series_returns[series_labels == -2], series_returns[series_labels == 2]
    if len(returns_m2) > 1 and len(returns_p2) > 1:
        std_m2, std_p2 = returns_m2.std(ddof=1), returns_p2.std(ddof=1)
        pooled_std_ext = np.sqrt((std_m2 ** 2 + std_p2 ** 2) / 2.0)
        return (returns_p2.mean() - returns_m2.mean()) / pooled_std_ext if pooled_std_ext != 0 else np.nan
    return np.nan


def monotonicity_test(series_returns: pd.Series, series_labels: pd.Series) -> bool:
    """Check monotonic increase of class means across present classes.

    Args:
        series_returns: returns series
        series_labels: integer label series

    Returns:
        True if means are strictly increasing across present classes.
    """
    classes = [-2, -1, 0, 1, 2]
    class_means = series_returns.groupby(series_labels).mean()
    means_present = class_means.dropna()
    means_present = means_present.loc[[c for c in classes if c in means_present.index]]
    if len(means_present) < 2 or means_present.isna().any():
        return False
    return all(means_present.iloc[i] < means_present.iloc[i + 1] for i in range(len(means_present) - 1))


def extremes_metrics(series_returns: pd.Series, series_labels: pd.Series, extreme_q: float = 0.99) -> Dict[str, float]:
    """Compute precision and recall for extreme (+2/-2) classes.

    Args:
        series_returns: returns series
        series_labels: integer label series
        extreme_q: quantile defining "extremes"

    Returns:
        Dict with keys: extreme_precision_up, extreme_precision_down, extreme_recall_up, extreme_recall_down
    """
    q_hi = series_returns.quantile(extreme_q)
    q_lo = series_returns.quantile(1 - extreme_q)
    mask_hi = series_returns >= q_hi
    mask_lo = series_returns <= q_lo
    if (series_labels == 2).any():
        extreme_precision_up = float((series_returns[series_labels == 2] >= q_hi).mean())
    else:
        extreme_precision_up = 0.0
    if (series_labels == -2).any():
        extreme_precision_down = float((series_returns[series_labels == -2] <= q_lo).mean())
    else:
        extreme_precision_down = 0.0
    extreme_recall_up = float((series_labels[mask_hi] == 2).mean()) if mask_hi.any() else np.nan
    extreme_recall_down = float((series_labels[mask_lo] == -2).mean()) if mask_lo.any() else np.nan
    return {
        'extreme_precision_up': extreme_precision_up,
        'extreme_precision_down': extreme_precision_down,
        'extreme_recall_up': extreme_recall_up,
        'extreme_recall_down': extreme_recall_down,
    }


def noise_ratio(series_returns: pd.Series, series_labels: pd.Series) -> float:
    """Ratio of neutral-class std to overall returns std.

    Args:
        series_returns: returns series
        series_labels: integer label series

    Returns:
        noise ratio or nan when insufficient data.
    """
    std_all = series_returns.std(ddof=1)
    neutral_returns = series_returns[series_labels == 0]
    if std_all and len(neutral_returns) > 1:
        return float(neutral_returns.std(ddof=1) / std_all)
    return np.nan


def class_stats(series_returns: pd.Series, series_labels: pd.Series) -> Dict[int, Dict[str, float]]:
    """Compute mean, std and count per class.

    Args:
        series_returns: returns series
        series_labels: integer label series

    Returns:
        Mapping class -> {'mean':..., 'std':..., 'count':...}
    """
    classes = [-2, -1, 0, 1, 2]
    stats = {}
    total_n = len(series_labels)
    for cls in classes:
        mask = series_labels == cls
        count = int(mask.sum())
        if count == 0:
            continue
        mean_val = float(series_returns[mask].mean())
        std_val = float(series_returns[mask].std(ddof=1)) if count > 1 else float(np.nan)
        stats[int(cls)] = {'mean': mean_val, 'std': std_val, 'count': count, 'share': (count / total_n) if total_n > 0 else 0.0}
    return stats


def analyze_labels_quality(
    returns_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    meta: Dict[str, Union[Dict[int, pd.Series], pd.Series]],
    *,
    overlap_trim_q: float = 0.01,
    extreme_q: float = 0.99,
    cohens_d_min_samples: int = 2,
    stability_use_mad: bool = True,
) -> Dict[int, Dict[str, Union[float, bool, Dict[str, float]]]]:
    """Produce per-horizon diagnostics that quantify label quality.

    This function runs a set of independent checks on the provided
    multi-horizon returns and corresponding 5-class labels and returns a
    dictionary of diagnostic metrics for each horizon. The checks are
    intentionally modular (separation, purity vs thresholds, overlap,
    monotonicity of class means, Cohen's d between adjacent classes,
    threshold stability, extremes precision/recall, noise in the neutral
    class, and per-class statistics) so they can be tested separately.

    Args:
        returns_df: DataFrame of forward returns indexed by timestamp with
            one column per horizon.
        labels_df: DataFrame of integer labels in {-2,-1,0,1,2} matching
            `returns_df` columns and index.
        meta: Auxiliary information produced by the labeling pipeline
            (expected keys include 't1_up', 't1_down', 't2_up', 't2_down',
            'imbalance', 'density'). Per-horizon entries may be scalars
            or Series aligned to timestamps.
        overlap_trim_q: Quantile used to trim tails when computing the
            symmetric overlap score (robustness to outliers).
        extreme_q: Quantile used to define extreme returns for precision
            / recall of ±2 labels.
        cohens_d_min_samples: Minimum samples required to compute Cohen's d
            for an adjacent-class pair.
        stability_use_mad: When True, compute threshold CV using MAD-based
            robust scale; otherwise use sample standard deviation.

    Returns:
        A mapping horizon -> diagnostics dictionary. The diagnostics
        dictionary contains the original/legacy fields for compatibility
        (for example: 'separation', 'purity_up', 'purity_dn',
        'overlap_score', 'monotonic', 'cohens_d', 'threshold_stability',
        'threshold_cv_up', 'threshold_cv_down', 'noise_ratio',
        'extreme_precision_up', 'extreme_precision_dn',
        'extreme_recall_up', 'extreme_recall_dn', 'class_means',
        'class_stds', 'present_classes', 'n_total'). Values that cannot
        be computed due to insufficient data are set to nan.

    Notes:
        - Threshold series in `meta` may be scalars or per-timestamp Series.
          When a per-timestamp Series is present it is reindexed to
          `returns_df.index`; if it contains only NaNs the scalar median
          is used as a fallback.
        - Separation for extremes uses pooled std between -2 and +2 class
          returns. Overlap is symmetric and computed only across actually
          present class pairs.
        - The function is deterministic and side-effect free; it is safe
          to call from higher-level pipelines and unit tests.
    """

    def _get_thresh_series_local(h: int, key: str) -> pd.Series:
        if key not in meta or h not in meta[key]:
            return pd.Series(np.nan, index=returns_df.index)
        val = meta[key][h]
        if isinstance(val, pd.Series):
            s = val.reindex(returns_df.index)
            if s.notna().sum() == 0:
                med = np.nanmedian(val.values) if len(val.values) else np.nan
                return pd.Series(med, index=returns_df.index)
            return s
        return pd.Series(float(val), index=returns_df.index)

    results: Dict[int, Dict[str, Union[float, bool, Dict[str, float]]]] = {}

    for h in labels_df.columns:
        r = returns_df[h]
        y = labels_df[h]

        classes = [-2, -1, 0, 1, 2]
        class_means = r.groupby(y).mean()
        class_stds = r.groupby(y).std(ddof=1)
        cm = class_means.reindex(classes).astype(float)
        cs = class_stds.reindex(classes).astype(float)

        sep_ratio = separation_ratio(r, y)

        t1_up_s = _get_thresh_series_local(h, 't1_up')
        t1_down_s = _get_thresh_series_local(h, 't1_down')

        purity_up, purity_down = purity_vs_thresholds(r, y, t1_up_s, t1_down_s)

        overlap = overlap_score(r, y, overlap_trim_q=overlap_trim_q)

        is_monotonic = monotonicity_test(r, y)

        cohens_map: Dict[str, float] = {}
        for (c1, c2) in [(-2, -1), (-1, 0), (0, 1), (1, 2)]:
            dval = cohens_d(r[y == c1], r[y == c2], min_samples=cohens_d_min_samples)
            if not np.isnan(dval):
                cohens_map[f'{c1}→{c2}'] = float(dval)

        cv_up = threshold_cv(t1_up_s, stability_use_mad=stability_use_mad) if not t1_up_s.isna().all() else np.nan
        cv_down = threshold_cv(t1_down_s, stability_use_mad=stability_use_mad) if not t1_down_s.isna().all() else np.nan
        threshold_stability = np.nanmax([cv_up, cv_down]) if not np.all(np.isnan([cv_up, cv_down])) else np.nan

        noise = noise_ratio(r, y)

        extremes = extremes_metrics(r, y, extreme_q=extreme_q)
        extreme_precision_up = extremes.get('extreme_precision_up', np.nan)
        extreme_precision_dn = extremes.get('extreme_precision_down', extremes.get('extreme_precision_dn', np.nan))
        extreme_recall_up = extremes.get('extreme_recall_up', np.nan)
        extreme_recall_dn = extremes.get('extreme_recall_down', extremes.get('extreme_recall_dn', np.nan))

        total_n = len(y)
        present = [c for c in classes if (y == c).any()]

        results[h] = {
            'separation': float(sep_ratio) if not np.isnan(sep_ratio) else np.nan,
            'purity_up': float(purity_up),
            'purity_dn': float(purity_down),
            'overlap_score': float(overlap) if not np.isnan(overlap) else np.nan,
            'monotonic': bool(is_monotonic),
            'cohens_d': {k: float(v) for k, v in cohens_map.items()},
            'threshold_stability': float(threshold_stability) if not np.isnan(threshold_stability) else np.nan,
            'threshold_cv_up': float(cv_up) if not np.isnan(cv_up) else np.nan,
            'threshold_cv_down': float(cv_down) if not np.isnan(cv_down) else np.nan,
            'noise_ratio': float(noise) if not np.isnan(noise) else np.nan,
            'extreme_precision_up': float(extreme_precision_up) if not np.isnan(extreme_precision_up) else np.nan,
            'extreme_precision_dn': float(extreme_precision_dn) if not np.isnan(extreme_precision_dn) else np.nan,
            'extreme_recall_up': float(extreme_recall_up) if not np.isnan(extreme_recall_up) else np.nan,
            'extreme_recall_dn': float(extreme_recall_dn) if not np.isnan(extreme_recall_dn) else np.nan,
            'class_means': {int(k): (float(v) if not np.isnan(v) else np.nan) for k, v in cm.to_dict().items()},
            'class_stds': {int(k): (float(v) if not np.isnan(v) else np.nan) for k, v in cs.to_dict().items()},
            'present_classes': present,
            'n_total': int(total_n),
        }

    return results
