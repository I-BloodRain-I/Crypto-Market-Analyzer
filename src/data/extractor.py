"""Helpers to extract, normalize and chunk downloaded market archives into parquet parts.

The module locates ZIP archives produced by the downloader, reads contained CSVs
in streaming fashion, coerces columns to expected dtypes, validates the result,
and writes out parquet parts of roughly `rows_per_part` rows. It is designed to
be resilient to occasional malformed rows and embedded CSV headers.
"""

import re
import logging
import zipfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataExtractor:
	"""Extracts downloaded market data archives and writes normalized parquet parts.

	This class provides a single high-level method `extract_and_chunk` which
	processes all matching ZIP archives for a given symbol and data type, and a
	set of helper methods used to resolve expected columns/dtypes, convert and
	validate chunk dtypes, and turn timestamp strings into integer open_time
	values (milliseconds since epoch).

	Constants:
		PROJECT_ROOT: Path to the project root inferred from this file's location.
		DATA_FOLDER: Top-level data directory inside the project.
		DOWNLOADED_FOLDER: Folder containing downloaded ZIP archives.
		RAW_FOLDER: Destination folder for normalized raw data (parquet parts).
	"""

	PROJECT_ROOT = Path(__file__).resolve().parents[2]
	DATA_FOLDER = PROJECT_ROOT / "data"
	DOWNLOADED_FOLDER = DATA_FOLDER / "downloaded"
	RAW_FOLDER = DATA_FOLDER / "raw"

	@classmethod
	def extract_and_chunk(
		cls,
		symbol: str,
		data_type: str,
		rows_per_part: int = 1_000_000,
		csv_chunksize: int = 10_000,
		remove_archives: bool = False,
		time_col = "open_time"
	) -> List[Path]:
		"""Scan downloaded ZIP archives for the symbol and write parquet parts.

		Args:
			 symbol: trading symbol folder name (e.g. "BTCUSDT").
			 data_type: type of data inside the archives ("klines" or "bookDepth").
			 rows_per_part: maximum number of rows to accumulate before writing a parquet part.
			 csv_chunksize: number of rows per read_csv chunk from each CSV.
			 remove_archives: if True attempt to delete processed zip archives to save disk space.
			 time_col: name of the time column to convert to milliseconds since epoch.

		Returns:
			List of written parquet part paths in sorted order.
		"""

		def _flush() -> None:
			nonlocal buf, buf_rows, part_idx
			if buf_rows == 0:
				return
			df = pd.concat(buf, ignore_index=True)
			table = pa.Table.from_pandas(df, preserve_index=False)
			out_path = extracted_folder / f"part{part_idx:04d}.parquet"
			# use a sensible default compression
			pq.write_table(table, str(out_path), compression="snappy")
			written.append(out_path)
			buf = []
			buf_rows = 0
			part_idx += 1

		buf: List[pd.DataFrame] = []
		buf_rows = 0
		part_idx = 1
		written: List[Path] = []

		extracted_folder = cls.RAW_FOLDER / symbol / data_type
		extracted_folder.mkdir(parents=True, exist_ok=True)

		archives = sorted(
			cls.DOWNLOADED_FOLDER.glob(f"{symbol}-*{data_type}.zip"),
			key=lambda p: datetime.strptime(m.group(), "%Y-%m-%d") if (m := re.search(r"\d{4}-\d{2}-\d{2}", p.stem)) else datetime.min,
		)
		logger.debug("Found %s archives for %s %s", len(archives), symbol, data_type)

		cols, dtypes = cls._resolve_cols_and_dtypes(data_type)

		for zip_file in archives:
			with zipfile.ZipFile(zip_file) as zf:
				csv_names = sorted([n for n in zf.namelist() if n.lower().endswith('.csv')])
				for name in csv_names:
					with zf.open(name) as f:
						reader = pd.read_csv(f, header=None, names=cols, chunksize=csv_chunksize)
						for chunk in reader:
							# remove potential header rows embedded in CSVs
							chunk = chunk[~chunk.iloc[:, 0].eq(chunk.columns[0])]
							chunk = cls._convert_dtypes(chunk, dtypes, time_col)
							if not chunk.empty:
								cls._validate_dtypes(chunk, dtypes)
							while len(chunk) > 0:
								need = rows_per_part - buf_rows
								take = chunk.iloc[:need]
								if len(take) > 0:
									buf.append(take)
									buf_rows += len(take)
								chunk = chunk.iloc[need:]
								if buf_rows == rows_per_part:
									_flush()
			if remove_archives:
				try:
					zip_file.unlink()
				except Exception:
					logger.exception("Failed to remove archive %s", zip_file)

		_flush()
		logger.info("Saved %s parts to %s", len(written), extracted_folder)
		return written

	@staticmethod
	def _resolve_cols_and_dtypes(data_type: str):
		"""Return column names and target dtypes for a supported data_type.

		Args:
			data_type: Identifier for the input data layout (e.g. 'klines' or 'bookDepth').

		Returns:
			A pair (cols, dtypes) where `cols` is an ordered list of column names to
			use when reading CSVs, and `dtypes` is a mapping of column name to the
			target pandas dtype string.

		Raises:
			ValueError: If an unknown data_type is provided.
		"""
		if data_type == "klines":
			cols = [
				"open_time", "open", "high", "low", "close",
				"volume", "close_time", "quote_volume", "count",
				"taker_buy_volume", "taker_buy_quote_volume",
				"ignore",
			]
			dtypes = {
				"open_time": "int64",
				"open": "float64",
				"high": "float64",
				"low": "float64",
				"close": "float64",
				"volume": "float64",
				"close_time": "int64",
				"quote_volume": "float64",
				"count": "int64",
				"taker_buy_volume": "float64",
				"taker_buy_quote_volume": "float64",
				"ignore": "int64",
			}
			return cols, dtypes
		elif data_type == "bookDepth":
			cols = ["timestamp", "percentage", "depth", "notional"]
			dtypes = {
				"timestamp": "object",
				"percentage": "int64",
				"depth": "float64",
				"notional": "float64",
			}
			return cols, dtypes
		elif data_type == "metrics":
			cols = [
				"create_time", "symbol", "sum_open_interest", 
				"sum_open_interest_value", "count_toptrader_long_short_ratio", 
				"sum_toptrader_long_short_ratio", "count_long_short_ratio", 
				"sum_taker_long_short_vol_ratio"
			]
			dtypes = {
				"create_time": "int64",
				"symbol": "object",
				"sum_open_interest": "int64",
				"sum_open_interest_value": "float64",
				"count_toptrader_long_short_ratio": "float64",
				"sum_toptrader_long_short_ratio": "float64",
				"count_long_short_ratio": "float64",
				"sum_taker_long_short_vol_ratio": "float64",
			}
			return cols, dtypes
		else:
			raise ValueError(f"Unknown data_type: {data_type}")

	@staticmethod
	def _convert_dtypes(df: pd.DataFrame, dtypes: Dict[str, str], time_col = str) -> pd.DataFrame:
		"""Apply the target dtypes to a chunk and return it.

		The function handles timestamp conversion separately.
		"""
		for col, dtype in dtypes.items():
			if col == time_col and col in df.columns:
				df[col] = DataExtractor._convert_timestamp_to_open_time(df[col])
			elif col in df.columns:
				try:
					df[col] = df[col].astype(dtype)
				except Exception:
					# fallback: coerce numeric conversions
					if dtype.startswith("int"):
						df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(dtype)
					else:
						df[col] = pd.to_numeric(df[col], errors="coerce").astype(dtype)
		return df

	@staticmethod
	def _validate_dtypes(df: pd.DataFrame, dtypes: Dict[str, str]) -> None:
		"""Ensure dataframe columns match expected dtypes or exit the program.

		Raises RuntimeError if any column is missing or has an unexpected dtype.
		"""
		mismatches: List[str] = []
		for col, expected in dtypes.items():
			if col not in df.columns:
				mismatches.append(f"{col}: missing")
				continue
			# timestamp columns are converted to int64 by _convert_timestamp_to_open_time
			expected_actual = "int64" if col == "timestamp" else expected
			actual = str(df[col].dtype)
			if actual != expected_actual:
				mismatches.append(f"{col}: expected {expected_actual}, got {actual}")
		if mismatches:
			logger.error("Dtype mismatch detected:\n%s", "\n".join(mismatches))
			raise RuntimeError("Dtype mismatch detected:\n%s", "\n".join(mismatches))

	@staticmethod
	def _convert_timestamp_to_open_time(ts: pd.Series) -> pd.Series:
		"""Convert timestamp string to integer open_time in milliseconds.

		Accepts strings with format "%Y-%m-%d %H:%M:%S" and coerces invalid values.
		"""
		dt = pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=True)
		try:
			ints = dt.astype('int64') // 10**6
		except Exception:
			# handle tz-aware Series by converting to UTC and removing tzinfo, then astype
			try:
				naive = dt.dt.tz_convert('UTC').dt.tz_localize(None)
				ints = naive.astype('int64') // 10**6
			except Exception:
				# final fallback: operate on numpy values
				ints = dt.to_numpy(dtype='int64') // 10**6
		return pd.Series(ints).fillna(0).astype('int64')