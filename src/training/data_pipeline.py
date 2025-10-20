import gc
import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
pd.options.mode.copy_on_write = True

from src.data import DataExtractor, DataProcessor, DataPipelineConfig
from src.utils import convert_timeframe_to_seconds

logger = logging.getLogger(__name__)


class DataPipelineForTraining:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    @classmethod
    def build(cls, config: DataPipelineConfig) -> "DataPipelineForTraining":
        return cls(pipeline=DataProcessor.build_pipeline(config))

    def _resolve_raw_data(self, symbol: str, date_type: str) -> pd.DataFrame:
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

    def _extend_metrics_time(self, metrics_data: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
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

    def _transpose_book_depth(self, df: pd.DataFrame) -> pd.DataFrame:
        depth = df.pivot(index="timestamp", columns="percentage", values="depth")
        notional = df.pivot(index="timestamp", columns="percentage", values="notional")

        depth.columns = [f"depth_{c}" for c in depth.columns]
        notional.columns = [f"notional_{c}" for c in notional.columns]

        return pd.concat([depth, notional], axis=1).reset_index()

    def _downgrade_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
        return df

    def _align_by_time(
        self,
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

    def _interpolate_book_depth(self, df: pd.DataFrame, timeframe: str = "1m") -> pd.DataFrame:
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
        
    def _combine_data_sources(self, symbol: str, timeframe: str = "1m") -> pd.DataFrame:
        book_depth_data = self._downgrade_dtype(self._resolve_raw_data(symbol, "bookDepth"))
        klines_data     = self._downgrade_dtype(self._resolve_raw_data(symbol, "klines"))
        klines_data.pop("ignore")
        metrics_data    = self._downgrade_dtype(self._resolve_raw_data(symbol, "metrics"))
        metrics_data.pop("symbol")

        logger.debug(
            f"Book Depth Data: {book_depth_data.shape}, "
            f"Klines Data: {klines_data.shape}, "
            f"Metrics Data: {metrics_data.shape}"
        )

        metrics_data = self._extend_metrics_time(metrics_data, timeframe=timeframe)
        book_depth_data = self._transpose_book_depth(book_depth_data)
        book_depth_data = self._interpolate_book_depth(book_depth_data, timeframe=timeframe)
        gc.collect()

        klines_data, metrics_data = self._align_by_time(
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

    def process_data(self, symbol: str):
        raw_data = self._combine_data_sources(symbol)
        processed_data = self.pipeline.transform(raw_data)
        return processed_data

if __name__ == "__main__":
    from src.data import FeatureScalingConfig
    logging.basicConfig(level=logging.DEBUG)
    # cfg = DataPipelineConfig(
    #     numerical_std=FeatureScalingConfig(
    #         features=["high", "close"],
    #         impute_strategy="median",
    #         scaler="standard",
    #     )
    # )
    pipeline = DataPipelineForTraining(None)
    pipeline._combine_data_sources("BTCUSDT").iloc[-10_000:].to_csv("combined_data.csv", index=False)
    # metrics = pipeline._resolve_raw_data("BTCUSDT", "metrics")
    # metrics.pop("symbol")
    # metrics_expanded = pipeline._extend_metrics_time(metrics, timeframe="1m")
    # print(metrics.shape, metrics_expanded.shape)
    # print(metrics.head(100))
    # print(metrics_expanded.head(200))