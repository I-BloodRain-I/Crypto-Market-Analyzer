import gc
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
pd.options.mode.copy_on_write = True

from data import (
    DataExtractor,
    DataProcessor,
    DataPipelineConfig,
    CustomTransformerConfig,
    LambdaFunctionConfig,
    MetricsNormalizer,
    OHLCVNormalizer,
    OrderbookNormalizer
)
from utils import convert_timeframe_to_seconds

logger = logging.getLogger(__name__)


class DataPipelineForTraining:

    COLS_TO_DROP = ["timestamp", "create_time", "open_time", "close_time", "quote_volume"]

    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline = pipeline or self._load_default_pipeline()

    @staticmethod
    def _load_default_pipeline() -> Pipeline:
        pipeline_cfg = DataPipelineConfig(
            custom_transformers=[
                CustomTransformerConfig(
                    features=["close"] + [f"depth_{i}" for i in range(-5, 6) if i != 0]\
                                    + [f"notional_{i}" for i in range(-5, 6) if i != 0],
                    instance=OrderbookNormalizer()
                ),
                CustomTransformerConfig(
                    features=["open", "high", "low", "close", "volume"],
                    instance=OHLCVNormalizer(lags=None)
                ),
                CustomTransformerConfig(
                    features=[
                        "count",
                        "taker_buy_volume",
                        "taker_buy_quote_volume",
                        "sum_open_interest",
                        "sum_open_interest_value",
                        "count_toptrader_long_short_ratio",
                        "sum_toptrader_long_short_ratio",
                        "count_long_short_ratio",
                        "sum_taker_long_short_vol_ratio"
                    ],
                    instance=MetricsNormalizer()
                ),
            ],
            lambda_funcs=[
                LambdaFunctionConfig(
                    feature="distance",
                    function="lambda x: np.clip(1 - x / 4, 0, 1)"
                )
            ]
        )
        return DataProcessor.build_pipeline(pipeline_cfg)

    @classmethod
    def build(cls, config: DataPipelineConfig) -> "DataPipelineForTraining":
        return cls(pipeline=DataProcessor.build_pipeline(config))

    @staticmethod
    def _resolve_raw_data(symbol: str, date_type: str) -> pd.DataFrame:
        # Check if raw data exists in the RAW_FOLDER
        if (DataExtractor.RAW_FOLDER / symbol / date_type).exists():
            if any((DataExtractor.RAW_FOLDER / symbol / date_type).glob("*.parquet")):
                raw_data_files = sorted((DataExtractor.RAW_FOLDER / symbol / date_type).glob("*.parquet"))
                raw_data = pd.concat([pd.read_parquet(f) for f in raw_data_files], ignore_index=True)
                return raw_data
        
        # Check if data exists but not yet extracted in the DOWNLOADED_FOLDER
        if (DataExtractor.DOWNLOADED_FOLDER).exists():
            if any((DataExtractor.DOWNLOADED_FOLDER).glob(f"{symbol}-*{date_type}.zip")):
                rows_per_part = 250_000 if date_type == "klines" else 1_000_000
                time_col = (
                    "open_time" if date_type == "klines" else 
                    "timestamp" if date_type == "bookDepth" else "create_time"
                )
                parts = DataExtractor.extract_and_chunk(
                    symbol, date_type, rows_per_part, time_col=time_col
                )
                raw_data = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
                return raw_data
            
        raise FileNotFoundError(f"No raw data found for symbol: {symbol}, date_type: {date_type}")

    @staticmethod
    def _extend_metrics_time(metrics_data: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
        timeframe_seconds = convert_timeframe_to_seconds(timeframe)
        if metrics_data is None or metrics_data.empty:
            return metrics_data

        res = DataProcessor.check_time_continuity(metrics_data, delta_ms=60_000*5, time_col="create_time")
        error_count = len([item for items in res.values() for item in items])
        logger.warning(f"Initial metrics time continuity check: {error_count} gaps found.")
        # Work in integer milliseconds only
        metrics_data = metrics_data.sort_values("create_time").reset_index(drop=True)
        metrics_data["create_time"] = metrics_data["create_time"].astype("int64")

        step_ms = int(timeframe_seconds * 1000)
        if step_ms <= 0:
            return metrics_data

        # Sometimes metrics have jittered timestamps; round to nearest timeframe to smooth
        rounded = ((metrics_data["create_time"] + step_ms // 2) // step_ms) * step_ms
        metrics_data = metrics_data.copy()
        metrics_data["create_time_rounded"] = rounded

        metrics_rounded = (
            metrics_data
            .sort_values(["create_time_rounded", "create_time"])
            .groupby("create_time_rounded", as_index=False)
            .last()
        )

        metrics_rounded = metrics_rounded.drop(columns=["create_time"]).rename(columns={"create_time_rounded": "create_time"})
        metrics_rounded["create_time"] = metrics_rounded["create_time"].astype("int64")

        existing_times = metrics_rounded["create_time"].tolist()
        if len(existing_times) <= 1:
            return metrics_rounded

        all_times = [existing_times[0]]
        for prev, curr in zip(existing_times, existing_times[1:]):
            next_expected = prev + step_ms
            if curr > next_expected:
                gap_count = (curr - next_expected) // step_ms
                if gap_count > 0:
                    extras = [next_expected + i * step_ms for i in range(int(gap_count))]
                    all_times.extend(extras)
            all_times.append(curr)

        full_times = pd.DataFrame({"create_time": all_times})
        expanded = full_times.merge(metrics_rounded, on="create_time", how="left", sort=True)

        res = DataProcessor.check_time_continuity(
            expanded, 
            delta_ms=timeframe_seconds*1000, 
            time_col="create_time"
        )
        error_count = len([item for items in res.values() for item in items])
        if error_count > 0:
            error_msg = (
                "Expanded metrics time has unexpected gaps after filling. "
                f"rows_expanded={expanded.shape[0]}, rows_input={metrics_rounded.shape[0]}, gaps_found={error_count}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        return expanded

    @staticmethod
    def _transpose_book_depth(df: pd.DataFrame) -> pd.DataFrame:
        depth = df.pivot(index="timestamp", columns="percentage", values="depth")
        notional = df.pivot(index="timestamp", columns="percentage", values="notional")

        depth.columns = [f"depth_{c}" for c in depth.columns]
        notional.columns = [f"notional_{c}" for c in notional.columns]

        return pd.concat([depth, notional], axis=1).reset_index()

    @staticmethod
    def _downgrade_dtype(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    @staticmethod
    def _align_by_time(
        main_df: pd.DataFrame,
        dfs: List[pd.DataFrame],
        main_time_col: str,
        time_cols: List[str]
    ) -> List[pd.DataFrame]:
        aligned_dfs = []
        timestamp_min = main_df[main_time_col].min()
        for df, time_col in zip(dfs, time_cols):
            aligned_dfs.append(df[df[time_col] >= timestamp_min])
        return aligned_dfs

    @staticmethod
    def _interpolate_book_depth(df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", cache=True)
        df = df.set_index("timestamp").sort_index()
        tf = timeframe if timeframe != "1m" else "1min"

        start = df.index.min().ceil(tf)
        end   = df.index.max().floor(tf)
        if start > end:
            raise ValueError(f"Not enough data to align to timeframe {timeframe}. start: {start}, end: {end}")

        target_idx = pd.date_range(start, end, freq=tf)
        union_idx = df.index.union(target_idx)
        num_cols = df.select_dtypes(include=[np.number]).columns
        out = df.reindex(union_idx)

        out[num_cols] = out[num_cols].interpolate(
            method="time",
            limit_direction="both", 
        )
        out = out.loc[target_idx]

        out = out.reset_index().rename(columns={"index": "timestamp"})
        out['timestamp'] = (out['timestamp'].view('int64') // 10**6).astype('int64')
        return out

    @staticmethod
    def _duplicate_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.drop(columns="create_time").ffill()
        main_indices = df.drop(columns="create_time").dropna(how='all').index

        last_main = pd.Series(np.nan, index=df.index)
        last_main.loc[main_indices] = main_indices
        last_main = last_main.ffill()
        
        df_filled['distance'] = (df.index.to_numpy() - last_main.to_numpy()).astype('int')
        df_filled["create_time"] = df["create_time"]

        return df_filled

    @classmethod
    def _combine_data_sources(cls, symbol: str, timeframe: str = "1m") -> pd.DataFrame:
        book_depth_data = cls._downgrade_dtype(cls._resolve_raw_data(symbol, "bookDepth"))
        klines_data     = cls._downgrade_dtype(cls._resolve_raw_data(symbol, "klines"))
        klines_data.pop("ignore")
        metrics_data    = cls._downgrade_dtype(cls._resolve_raw_data(symbol, "metrics"))
        metrics_data.pop("symbol")

        logger.debug(
            f"Book Depth Data: {book_depth_data.shape}, "
            f"Klines Data: {klines_data.shape}, "
            f"Metrics Data: {metrics_data.shape}"
        )

        metrics_data = cls._extend_metrics_time(metrics_data, timeframe=timeframe)
        metrics_data = cls._duplicate_metrics(metrics_data)
        book_depth_data = cls._transpose_book_depth(book_depth_data)
        book_depth_data = cls._interpolate_book_depth(book_depth_data, timeframe=timeframe)
        gc.collect()

        klines_data, metrics_data = cls._align_by_time(
            book_depth_data,
            [klines_data, metrics_data],
            "timestamp",
            ["open_time", "create_time"]
        )
        gc.collect()

        combined_data = book_depth_data.merge(
            klines_data, 
            left_on="timestamp", 
            right_on="open_time", 
            how="left",
            copy=False
        ).merge(
            metrics_data, 
            left_on="timestamp", 
            right_on="create_time", 
            how="left",
            copy=False
        )
        logger.debug(f"Combined Data Shape: {combined_data.shape}")

        del book_depth_data
        del klines_data
        del metrics_data
        gc.collect()

        return combined_data

    @staticmethod
    def _validate_df(df: pd.DataFrame, timeframe: str) -> None:
        ms = convert_timeframe_to_seconds(timeframe) * 1000
        duplicates = DataProcessor.check_no_duplicates(df, columns=["timestamp"])
        if duplicates.shape[0] > 0:
            logger.error(f"Duplicate timestamps found: \n%s", duplicates)
            raise ValueError("Combined data contains duplicate timestamps.")
        missing = DataProcessor.check_no_missing(df) 
        if missing:
            logger.error(f"Missing timestamps found: \n%s", missing)
            raise ValueError("Combined data contains missing timestamps.")
        time_continuity = DataProcessor.check_time_continuity(df, delta_ms=ms, time_col="timestamp")
        if [item for items in time_continuity.values() for item in items]:
            logger.error(f"Gaps in time continuity found: \n%s", time_continuity)
            raise ValueError("Combined data contains gaps in time.")

    @classmethod
    def load_df(cls, symbol: str, timeframe: str = "1m") -> pd.DataFrame:
        raw_data = cls._combine_data_sources(symbol)
        raw_data = raw_data.sort_values("timestamp").reset_index(drop=True)
        cls._validate_df(raw_data, timeframe=timeframe)
        return raw_data
    
    @staticmethod
    def add_time_features(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        timestamp = pd.to_datetime(df[time_col], unit="ms", utc=True, cache=True)
        df["time_sin_month"]  = np.sin(2 * np.pi * timestamp.dt.month / 12)
        df["time_cos_month"]  = np.cos(2 * np.pi * timestamp.dt.month / 12)
        df["time_sin_day"]    = np.sin(2 * np.pi * timestamp.dt.day_of_week / 7)
        df["time_cos_day"]    = np.cos(2 * np.pi * timestamp.dt.day_of_week / 7)
        df["time_sin_hour"]   = np.sin(2 * np.pi * timestamp.dt.hour / 24)
        df["time_cos_hour"]   = np.cos(2 * np.pi * timestamp.dt.hour / 24)
        df["time_sin_minute"] = np.sin(2 * np.pi * timestamp.dt.minute / 60)
        df["time_cos_minute"] = np.cos(2 * np.pi * timestamp.dt.minute / 60)
        return df

    @classmethod
    def process_data(
        cls, 
        symbol: str, 
        train: bool = False, 
        timeframe: str = "1m", 
        return_df: bool = False
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        raw_data = cls._combine_data_sources(symbol)
        raw_data = raw_data.sort_values("timestamp").reset_index(drop=True)
        cls._validate_df(raw_data, timeframe=timeframe)

        if train:
            past_features = cls.pipeline.fit_transform(raw_data.drop(columns=cls.COLS_TO_DROP))
        else:
            past_features = cls.pipeline.transform(raw_data.drop(columns=cls.COLS_TO_DROP))

        if not np.isfinite(past_features).all():
            raise ValueError("Processed data contains NaN or infinite values.")
        
        timestamp = pd.to_datetime(raw_data["timestamp"], unit="ms", utc=True, cache=True)
        future_features = pd.DataFrame(
            {
                "sin_month": np.sin(2 * np.pi * timestamp.dt.month / 12),
                "cos_month": np.cos(2 * np.pi * timestamp.dt.month / 12),
                "sin_day": np.sin(2 * np.pi * timestamp.dt.day_of_week / 7),
                "cos_day": np.cos(2 * np.pi * timestamp.dt.day_of_week / 7),
                "sin_hour": np.sin(2 * np.pi * timestamp.dt.hour / 24),
                "cos_hour": np.cos(2 * np.pi * timestamp.dt.hour / 24),
                "sin_minute": np.sin(2 * np.pi * timestamp.dt.minute / 60),
                "cos_minute": np.cos(2 * np.pi * timestamp.dt.minute / 60),
            }
        )
        
        if not return_df:
            return past_features, future_features.to_numpy()
        else:
            past_features = pd.DataFrame(
                past_features, 
                columns=cls.pipeline.get_feature_names_out(
                    raw_data.drop(columns=cls.COLS_TO_DROP).columns.tolist()
                )
            )
            return past_features, future_features


if __name__ == "__main__":
    from data import FeatureScalingConfig
    logging.basicConfig(level=logging.DEBUG)
    # cfg = DataPipelineConfig(
    #     numerical_std=FeatureScalingConfig(
    #         features=["high", "close"],
    #         impute_strategy="median",
    #         scaler="standard",
    #     )
    # )
    pipeline = DataPipelineForTraining(None)
    print("ok")
    exit(0)
    df = pipeline._combine_data_sources("BTCUSDT")
    df.iloc[-10_000:].to_csv("combined_data.csv", index=False)