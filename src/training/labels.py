import logging
import itertools
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import LabelGenerator
from analysis import analyze_labels_quality


def _evaluate_single_config(args):
    """
    Worker function to evaluate a single parameter configuration.
    Designed to be used with multiprocessing.
    """
    params, df_close, df_depth, horizons, band_cols = args
    
    logging.getLogger('src.data.label_generator').setLevel(logging.ERROR)
    
    try:
        gen = LabelGenerator(
            horizons=horizons,
            alpha=params.get('alpha', 0.9),
            k_imb=params.get('k_imb', 0.3),
            k_liq=params.get('k_liq', 0.1),
            gamma_strong=params.get('gamma_strong', 2.2),
            span_sig=params.get('span_sig', 200),
            band_cols=band_cols,
            floor_bp=params.get('floor_bp', 0.0002)
        )
        
        returns_df, labels_5, probs_5, sample_weights, meta = gen.build_labels(
            df_close, df_depth
        )
        
        metrics = analyze_labels_quality(returns_df, labels_5, meta)
        
        row = params.copy()
        
        quality_scores = []
        for h, h_metrics in metrics.items():
            sep = h_metrics.get('separation', np.nan)
            pur_up = h_metrics.get('purity_up', np.nan)
            pur_dn = h_metrics.get('purity_dn', np.nan)
            overlap = h_metrics.get('overlap_score', np.nan)
            monotonic = h_metrics.get('monotonic', False)
            thresh_stab = h_metrics.get('threshold_stability', np.nan)
            thresh_cv_up = h_metrics.get('threshold_cv_up', np.nan)
            thresh_cv_down = h_metrics.get('threshold_cv_down', np.nan)
            noise = h_metrics.get('noise_ratio', np.nan)
            ext_prec_up = h_metrics.get('extreme_precision_up', np.nan)
            ext_prec_dn = h_metrics.get('extreme_precision_dn', np.nan)
            ext_rec_up = h_metrics.get('extreme_recall_up', np.nan)
            ext_rec_dn = h_metrics.get('extreme_recall_dn', np.nan)
            
            cohens_d_dict = h_metrics.get('cohens_d', {})
            n_total = h_metrics.get('n_total', 0)
            
            row[f'h{h}_separation'] = sep
            row[f'h{h}_purity_up'] = pur_up
            row[f'h{h}_purity_dn'] = pur_dn
            row[f'h{h}_overlap'] = overlap
            row[f'h{h}_monotonic'] = int(monotonic)
            row[f'h{h}_threshold_stability'] = thresh_stab
            row[f'h{h}_threshold_cv_up'] = thresh_cv_up
            row[f'h{h}_threshold_cv_down'] = thresh_cv_down
            row[f'h{h}_noise_ratio'] = noise
            row[f'h{h}_extreme_precision_up'] = ext_prec_up
            row[f'h{h}_extreme_precision_dn'] = ext_prec_dn
            row[f'h{h}_extreme_recall_up'] = ext_rec_up
            row[f'h{h}_extreme_recall_dn'] = ext_rec_dn
            
            for pair_key, d_val in cohens_d_dict.items():
                clean_key = pair_key.replace('→', '_to_').replace('-', 'neg').replace('+', 'pos')
                row[f'h{h}_cohens_d_{clean_key}'] = d_val
            
            label_counts = labels_5[h].value_counts()
            for cls in [-2, -1, 0, 1, 2]:
                count = label_counts.get(cls, 0)
                share_pct = (count / n_total * 100) if n_total > 0 else 0.0
                row[f'h{h}_class_{cls}_share'] = share_pct
            
            q_score = (
                (sep > 2.0) * 20 if not np.isnan(sep) else 0
            ) + ((pur_up > 0.80) * 15) + ((pur_dn > 0.80) * 15) + (
                (overlap < 0.3) * 20 if not np.isnan(overlap) else 0
            ) + (monotonic * 15) + (
                (thresh_stab < 0.5) * 15 if not np.isnan(thresh_stab) else 0
            )
            
            row[f'h{h}_quality_score'] = q_score
            quality_scores.append(q_score)
        
        row['avg_quality_score'] = np.mean(quality_scores)
        return row
        
    except Exception as e:
        row = params.copy()
        row['error'] = str(e)
        row['avg_quality_score'] = np.nan
        return row


def tune_label_generator(
    df: pd.DataFrame,
    param_grid: Dict[str, List[Any]],
    horizons: List[int] = [1, 3, 5, 10, 15],
    band_cols: tuple = (1, 2, 3, 4, 5),
    output_path: Path = None,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Tunes LabelGenerator parameters using grid search with parallel processing.

    Args:
        df: DataFrame with price and depth data
        param_grid: Dictionary mapping parameter names to lists of values to try
        horizons: List of forecast horizons
        band_cols: Tuple of orderbook band columns to use
        output_path: Path to save results CSV file
        n_jobs: Number of parallel jobs. -1 means use all available CPU cores

    Returns:
        DataFrame with all configurations and their metrics
    """
    import os
    
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    df_close = df["close"]
    df_depth = df[[col for col in df.columns if col.startswith("depth_")]]
    
    param_combinations = list(itertools.product(*param_values))
    total_combinations = len(param_combinations)
    
    print(f"Testing {total_combinations} parameter combinations using {n_jobs} workers...\n")
    
    tasks = [
        (dict(zip(param_names, combo)), df_close, df_depth, horizons, band_cols)
        for combo in param_combinations
    ]
    
    results = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_evaluate_single_config, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=total_combinations, 
                          desc="Tuning parameters", unit="config"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                task = futures[future]
                params = task[0]
                row = params.copy()
                row['error'] = str(e)
                row['avg_quality_score'] = np.nan
                results.append(row)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('avg_quality_score', ascending=False, na_position='last')
    
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
    
    return results_df