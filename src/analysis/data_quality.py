from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pearson",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = "coolwarm",
    annot: bool = False,
    fmt: str = ".2f",
    vmin: float = -1,
    vmax: float = 1,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    mask_upper: bool = False
) -> plt.Figure:
    """Plot correlation heatmap for features in the dataframe.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        method: correlation method ('pearson', 'kendall', 'spearman')
        figsize: figure size as (width, height)
        cmap: colormap for the heatmap
        annot: whether to annotate cells with correlation values
        fmt: string formatting for annotations
        vmin: minimum value for colormap scale
        vmax: maximum value for colormap scale
        save_path: path to save the figure. If None, figure is not saved
        title: title for the plot
        mask_upper: if True, mask the upper triangle of the correlation matrix

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    data_subset = df[features]
    
    corr_matrix = data_subset.corr(method=method)
    
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        vmin=vmin,
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    if title is None:
        title = f"Feature Correlation Matrix ({method.capitalize()})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_top_correlations(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pearson",
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    exclude_diagonal: bool = True
) -> plt.Figure:
    """Plot top N highest absolute correlations as a bar chart.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        method: correlation method ('pearson', 'kendall', 'spearman')
        top_n: number of top correlations to display
        figsize: figure size as (width, height)
        save_path: path to save the figure. If None, figure is not saved
        exclude_diagonal: whether to exclude self-correlations (always 1.0)

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    data_subset = df[features]
    corr_matrix = data_subset.corr(method=method)
    
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if exclude_diagonal or i != j:
                corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    corr_df = pd.DataFrame(corr_pairs)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False).head(top_n)
    
    corr_df['pair'] = corr_df['feature_1'] + ' - ' + corr_df['feature_2']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red' if x < 0 else 'green' for x in corr_df['correlation']]
    
    ax.barh(range(len(corr_df)), corr_df['correlation'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df['pair'], fontsize=9)
    ax.set_xlabel('Correlation Coefficient', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Correlations ({method.capitalize()})', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_correlations(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "pearson",
    threshold: float = 0.7
) -> pd.DataFrame:
    """Analyze and return feature pairs with high correlations.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        method: correlation method ('pearson', 'kendall', 'spearman')
        threshold: minimum absolute correlation threshold

    Returns:
        DataFrame with highly correlated feature pairs
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    data_subset = df[features]
    corr_matrix = data_subset.corr(method=method)
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_value,
                    'abs_correlation': abs(corr_value)
                })
    
    result_df = pd.DataFrame(high_corr_pairs)
    if not result_df.empty:
        result_df = result_df.sort_values('abs_correlation', ascending=False)
    
    return result_df


def plot_boxplots(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    cols: int = 4,
    save_path: Optional[Path] = None,
    show_outliers: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot boxplots for multiple features to visualize distributions and outliers.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        figsize: figure size as (width, height)
        cols: number of columns in the subplot grid
        save_path: path to save the figure. If None, figure is not saved
        show_outliers: whether to show outlier points
        title: title for the plot

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_features = len(features)
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        bp = ax.boxplot(
            [data],
            vert=True,
            patch_artist=True,
            showfliers=show_outliers,
            widths=0.5
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_xticklabels([feature], rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = 'Feature Distribution Boxplots'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_outlier_detection(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
    figsize: Tuple[int, int] = (14, 8),
    cols: int = 4,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Detect and visualize outliers using IQR or Z-score method.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        method: outlier detection method ('iqr' or 'zscore')
        threshold: threshold for outlier detection (1.5 for IQR, 3 for Z-score typically)
        figsize: figure size as (width, height)
        cols: number of columns in the subplot grid
        save_path: path to save the figure. If None, figure is not saved
        title: title for the plot

    Returns:
        Tuple of (matplotlib Figure object, DataFrame with outlier statistics)
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_features = len(features)
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    outlier_stats = []
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        elif method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = z_scores > threshold
            lower_bound = data.mean() - threshold * data.std()
            upper_bound = data.mean() + threshold * data.std()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_count = outliers.sum()
        outlier_pct = (outlier_count / len(data)) * 100
        
        outlier_stats.append({
            'feature': feature,
            'total_count': len(data),
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_pct,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min': data.min(),
            'max': data.max()
        })
        
        ax.scatter(range(len(data)), data, c=outliers, cmap='coolwarm', 
                   alpha=0.5, s=1)
        ax.axhline(y=lower_bound, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(y=upper_bound, color='r', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f'{feature}\n{outlier_count} outliers ({outlier_pct:.2f}%)', 
                     fontsize=9)
        ax.set_xlabel('Index', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = f'Outlier Detection ({method.upper()}, threshold={threshold})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    stats_df = pd.DataFrame(outlier_stats)
    return fig, stats_df


def plot_distribution_comparison(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    cols: int = 4,
    bins: int = 50,
    save_path: Optional[Path] = None,
    show_kde: bool = True,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot histograms with KDE for multiple features to visualize distributions.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        figsize: figure size as (width, height)
        cols: number of columns in the subplot grid
        bins: number of bins for histograms
        save_path: path to save the figure. If None, figure is not saved
        show_kde: whether to overlay KDE plot
        title: title for the plot

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_features = len(features)
    rows = int(np.ceil(n_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data = df[feature].dropna()
        
        ax.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        
        if show_kde and len(data) > 1:
            ax2 = ax.twinx()
            data.plot.kde(ax=ax2, color='red', linewidth=2)
            ax2.set_ylabel('Density', fontsize=8)
            ax2.tick_params(labelsize=8)
        
        ax.set_title(f'{feature}', fontsize=9)
        ax.set_xlabel('Value', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = 'Feature Distribution Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_data_quality_metrics(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """Compute comprehensive data quality metrics for features.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns

    Returns:
        DataFrame with quality metrics for each feature
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    metrics = []
    
    for feature in features:
        data = df[feature]
        
        missing_count = data.isna().sum()
        missing_pct = (missing_count / len(data)) * 100
        
        if data.dtype in [np.number] and not data.dropna().empty:
            numeric_data = data.dropna()
            
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((numeric_data < lower_bound) | (numeric_data > upper_bound)).sum()
            outlier_pct = (outliers / len(numeric_data)) * 100
            
            unique_count = numeric_data.nunique()
            unique_pct = (unique_count / len(numeric_data)) * 100
            
            zero_count = (numeric_data == 0).sum()
            zero_pct = (zero_count / len(numeric_data)) * 100
            
            metrics.append({
                'feature': feature,
                'dtype': str(data.dtype),
                'count': len(data),
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'unique_count': unique_count,
                'unique_pct': unique_pct,
                'zero_count': zero_count,
                'zero_pct': zero_pct,
                'mean': numeric_data.mean(),
                'std': numeric_data.std(),
                'min': numeric_data.min(),
                'q25': Q1,
                'median': numeric_data.median(),
                'q75': Q3,
                'max': numeric_data.max(),
                'iqr': IQR,
                'outliers_count': outliers,
                'outliers_pct': outlier_pct,
                'skewness': numeric_data.skew(),
                'kurtosis': numeric_data.kurtosis()
            })
        else:
            metrics.append({
                'feature': feature,
                'dtype': str(data.dtype),
                'count': len(data),
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'unique_count': data.nunique(),
                'unique_pct': (data.nunique() / len(data)) * 100 if len(data) > 0 else 0
            })
    
    return pd.DataFrame(metrics)


def plot_missing_data(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Visualize missing data patterns in the dataframe.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all columns
        figsize: figure size as (width, height)
        save_path: path to save the figure. If None, figure is not saved
        title: title for the plot

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.columns.tolist()
    
    data_subset = df[features]
    
    missing_counts = data_subset.isna().sum()
    missing_pcts = (missing_counts / len(data_subset)) * 100
    
    missing_df = pd.DataFrame({
        'feature': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_pct': missing_pcts.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if missing_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No missing data found!', 
                ha='center', va='center', fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    ax1.barh(range(len(missing_df)), missing_df['missing_count'], color='coral', alpha=0.7)
    ax1.set_yticks(range(len(missing_df)))
    ax1.set_yticklabels(missing_df['feature'], fontsize=9)
    ax1.set_xlabel('Missing Count', fontsize=11)
    ax1.set_title('Missing Data Count', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    ax2.barh(range(len(missing_df)), missing_df['missing_pct'], color='salmon', alpha=0.7)
    ax2.set_yticks(range(len(missing_df)))
    ax2.set_yticklabels(missing_df['feature'], fontsize=9)
    ax2.set_xlabel('Missing Percentage (%)', fontsize=11)
    ax2.set_title('Missing Data Percentage', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    if title is None:
        title = 'Missing Data Analysis'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_statistics(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    cols: int = 3,
    save_path: Optional[Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot comprehensive statistics for features including mean, std, quartiles.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all numeric columns
        figsize: figure size as (width, height)
        cols: number of columns in the subplot grid
        save_path: path to save the figure. If None, figure is not saved
        title: title for the plot

    Returns:
        matplotlib Figure object
    """
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    stats_df = df[features].describe().T
    stats_df['skewness'] = df[features].skew()
    stats_df['kurtosis'] = df[features].kurtosis()
    
    n_stats = 6
    rows = int(np.ceil(n_stats / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    stat_names = ['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']
    colors = ['skyblue', 'lightgreen', 'coral', 'pink', 'gold', 'plum']
    
    for idx, (stat, color) in enumerate(zip(stat_names, colors)):
        ax = axes[idx]
        values = stats_df[stat].sort_values(ascending=False)
        
        ax.barh(range(len(values)), values, color=color, alpha=0.7)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(values.index, fontsize=8)
        ax.set_xlabel(stat.capitalize(), fontsize=10)
        ax.set_title(f'Feature {stat.capitalize()}', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    for idx in range(n_stats, len(axes)):
        axes[idx].axis('off')
    
    if title is None:
        title = 'Feature Statistics Overview'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def print_feature_stats(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    round_digits: int = 4
) -> pd.DataFrame:
    """Print a compact table of statistics for each column with features on the left.

    The function prints and returns a DataFrame where each row is a feature and
    columns include dtype, count, missing percentage, mean, std, 25%, 50%, 75%,
    min and max for numeric features. Non-numeric features will have dtype,
    count, missing percentage and unique count.

    Args:
        df: input dataframe
        features: list of feature names to include. If None, use all columns
        round_digits: number of decimal places to round numeric statistics

    Returns:
        DataFrame with statistics for each feature (features as index)
    """
    if features is None:
        features = df.columns.tolist()

    rows = []
    for feature in features:
        series = df[feature]
        count = len(series.dropna())
        missing = int(series.isna().sum())
        missing_pct = (missing / count) * 100 if count > 0 else 0.0
        dtype = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            numeric = series.dropna()
            mean = numeric.mean() if not numeric.empty else np.nan
            std = numeric.std() if not numeric.empty else np.nan
            q25 = numeric.quantile(0.25) if not numeric.empty else np.nan
            q50 = numeric.quantile(0.5) if not numeric.empty else np.nan
            q75 = numeric.quantile(0.75) if not numeric.empty else np.nan
            minimum = numeric.min() if not numeric.empty else np.nan
            maximum = numeric.max() if not numeric.empty else np.nan

            rows.append({
                'feature': feature,
                'dtype': dtype,
                'count': count,
                'missing_pct': missing_pct,
                'mean': mean,
                'std': std,
                '25%': q25,
                '50%': q50,
                '75%': q75,
                'min': minimum,
                'max': maximum,
            })
        else:
            unique = int(series.nunique(dropna=True))
            rows.append({
                'feature': feature,
                'dtype': dtype,
                'count': count,
                'missing_pct': missing_pct,
                'unique': unique
            })

    result = pd.DataFrame(rows).set_index('feature')

    def _format_val(x):
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return f"{x}"
        if isinstance(x, (float, np.floating)):
            return f"{x:.{round_digits}f}"
        return str(x)

    display_df = result.copy().applymap(_format_val)

    print(display_df.to_string())

    return result
