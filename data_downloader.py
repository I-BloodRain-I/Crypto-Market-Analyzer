"""
Download Binance futures data .zip files by date range.

Usage examples:
    python data_downloader.py --start 01-01-2023 --end 05-01-2023 --endpoint klines
    python data_downloader.py --symbol BTCUSDT --interval 1m --start 01-01-2023 --end 05-01-2023 \
        --workers 4 --endpoint klines --retries 5 --timeout 10 --dest data/downloaded --overwrite

Arguments:
    --symbol    Trading pair symbol (default: BTCUSDT)
    --interval  Kline interval, e.g. 1m, 1h, 1d (default: 1m)
    --start     Start date in DD-MM-YYYY (required)
    --end       End date in DD-MM-YYYY (required)
    --dest      Destination folder to save zips (default: data/downloaded)
    --retries   Number of request retries (default: 5)
    --timeout   Per-request timeout in seconds (default: 10.0)
    --endpoint  API endpoint to download from, e.g. klines or bookDepth (required)
    --overwrite Overwrite existing files (flag)
    --workers   Number of download threads to use (default: 4; 1 = sequential)
"""

import sys
import logging
import argparse
import threading
from pathlib import Path
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter

DATA_FOLDER = Path.cwd() / "data"
DOWNLOADED_FOLDER = DATA_FOLDER / "downloaded"
DOWNLOADED_FOLDER.mkdir(exist_ok=True, parents=True)

BASIC_URL = "https://data.binance.vision/data/futures/um/daily"
ENDPOINTS = ("klines", "bookDepth", "bookTicker", "metrics")

logger = logging.getLogger(__name__)


def _create_session(retries: int = 5, backoff_factor: float = 0.5):
    """Create a requests Session with retry logic.

    Uses a compatible parameter name for urllib3 versions that differ on
    the Retry constructor keyword (allowed_methods vs method_whitelist).
    """
    session = requests.Session()
    retry_kwargs = dict(total=retries, backoff_factor=backoff_factor, status_forcelist=(429, 500, 502, 503, 504))
    try:
        # newer urllib3
        retry_kwargs["allowed_methods"] = frozenset(["GET"])
        retry = Retry(**retry_kwargs)
    except TypeError:
        # older urllib3
        retry_kwargs["method_whitelist"] = frozenset(["GET"])  # type: ignore[attr-defined]
        retry = Retry(**retry_kwargs)

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "crypto-downloader/1.0 (+https://example)"})
    return session


class BinanceDataDownloader:
    """Downloader for Binance compressed files.

    This class encapsulates the logic to download daily compressed files from
    Binance (for example `klines` or `bookDepth`) for a given symbol,
    interval and date range. It handles session creation with retries,
    streaming downloads to temporary files, and optional multithreaded
    execution.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        start_str: str,
        end_str: str,
        save_path: Path,
        endpoint: str = "klines",
        session: Optional[requests.Session] = None,
        timeout: float = 10.0,
        chunk_size: int = 1024 * 64,
        overwrite: bool = False,
        max_workers: int = 4,
        retries: int = 5,
        backoff_factor: float = 0.5,
    ) -> None:
        """Initialize the downloader with the requested configuration.

        Args:
            symbol: Trading pair symbol (e.g. BTCUSDT).
            interval: Kline interval (e.g. 1m, 1h, 1d).
            start_str: Start date in DD-MM-YYYY format.
            end_str: End date in DD-MM-YYYY format.
            save_path: Directory where downloaded .zip files will be saved.
            endpoint: API endpoint path segment to download from (e.g. klines or bookDepth).
            session: Optional requests.Session to reuse for sequential runs; if None
                and multithreading is used, per-thread sessions are created.
            timeout: Per-request timeout in seconds.
            chunk_size: Bytes to read per iteration when streaming responses.
            overwrite: If True, overwrite existing files; otherwise skip non-empty files.
            max_workers: Number of worker threads to use (1 = sequential).
            retries: Number of request retries handled by urllib3 Retry.
            backoff_factor: Backoff factor for retries.

        Raises:
            ValueError: If start_date is after end_date.
        """
        self.symbol = symbol
        self.interval = interval
        self.start_str = start_str
        self.end_str = end_str
        self.save_path = save_path
        self.endpoint = endpoint
        self.session = session
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.overwrite = overwrite
        self.max_workers = max_workers
        self.retries = retries
        self.backoff_factor = backoff_factor

        # prepare dates
        start_date = datetime.strptime(start_str, "%d-%m-%Y")
        end_date = datetime.strptime(end_str, "%d-%m-%Y")
        if start_date > end_date:
            raise ValueError("start date must be <= end date")
        self.dates = []
        cur = start_date
        while cur <= end_date:
            self.dates.append(cur)
            cur += timedelta(days=1)

        # clamp workers
        if self.max_workers is None:
            self.max_workers = 1
        self.max_workers = max(1, min(self.max_workers, len(self.dates)))

        # thread local for sessions used in multithreaded mode
        self._thread_local = threading.local()

    def _get_thread_session(self) -> requests.Session:
        if getattr(self._thread_local, "session", None) is None:
            self._thread_local.session = _create_session(retries=self.retries, backoff_factor=self.backoff_factor)
        return self._thread_local.session

    def _download_for_date(self, cur_date: datetime) -> Tuple[Path, str]:
        file_name = self._resolve_file_name(cur_date)
        out_file = self.save_path / file_name.replace(".zip", f"_{self.endpoint}.zip")
        url = self._resolve_url(file_name)

        if out_file.exists() and not self.overwrite and out_file.stat().st_size > 0:
            logger.info("Skipping existing file %s", file_name)
            return out_file, "skipped"

        if self.max_workers > 1:
            sess = self._get_thread_session()
        else:
            sess = self.session or _create_session(retries=self.retries, backoff_factor=self.backoff_factor)

        tmp_file = out_file.with_suffix(".part")
        try:
            with sess.get(url, timeout=self.timeout, stream=True) as response:
                response.raise_for_status()
                self.save_path.mkdir(parents=True, exist_ok=True)
                with open(tmp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                tmp_file.replace(out_file)
                logger.info("Downloaded %s", file_name)
                return out_file, "downloaded"
        except requests.HTTPError as e:
            status = e.response.status_code if getattr(e, 'response', None) is not None else None
            logger.warning("Failed to download %s: HTTP %s", file_name, status)
            try:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
            except Exception:
                pass
            return out_file, "http_error"
        except requests.RequestException as e:
            logger.warning("Network error when downloading %s: %s", file_name, e)
            try:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
            except Exception:
                pass
            return out_file, "network_error"

    def _resolve_file_name(self, cur_date: datetime) -> str:
        date_str = cur_date.strftime("%Y-%m-%d")
        if self.endpoint == "klines":
            file_name = f"{self.symbol}-{self.interval}-{date_str}.zip"
        elif self.endpoint in ["bookDepth", "bookTicker", "metrics"]:
            file_name = f"{self.symbol}-{self.endpoint}-{date_str}.zip"
        else:
            raise ValueError(f"Unknown endpoint: {self.endpoint}")
        return file_name

    def _resolve_url(self, file_name: str) -> str:
        if self.endpoint == "klines":
            url = f"{BASIC_URL}/{self.endpoint}/{self.symbol}/{self.interval}/{file_name}"
        elif self.endpoint in ["bookDepth", "bookTicker", "metrics"]:
            url = f"{BASIC_URL}/{self.endpoint}/{self.symbol}/{file_name}"
        else:
            raise ValueError(f"Unknown endpoint: {self.endpoint}")
        return url

    def download(self) -> List[Tuple[Path, str]]:
        """Download files for the configured date range.

        The method will perform the downloads sequentially or concurrently
        depending on the configuration provided to the initializer.

        Returns:
            A list of tuples (path, status) where status is one of:
            'downloaded', 'skipped', 'http_error', 'network_error'.
        """
        results: List[Tuple[Path, str]] = []
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._download_for_date, d): d for d in self.dates}
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        results.append(res)
                    except Exception as exc:
                        logger.exception("Download task raised: %s", exc)
        else:
            for d in self.dates:
                results.append(self._download_for_date(d))
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance futures kline .zip files by date range")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval, e.g. 1m, 1h, 1d")
    parser.add_argument("--start", required=True, help="Start date DD-MM-YYYY")
    parser.add_argument("--end", required=True, help="End date DD-MM-YYYY")
    parser.add_argument("--dest", default=str(DOWNLOADED_FOLDER), help="Destination folder to save zips")
    parser.add_argument("--retries", type=int, default=5, help="Number of request retries")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds")
    parser.add_argument("--endpoint", required=True, help="API endpoint to download from, e.g. klines or bookDepth")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--workers", type=int, default=4, help="Number of download threads to use (1 = sequential)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.endpoint not in ENDPOINTS:
        logger.error(f"Invalid endpoint specified. Supported endpoints: {ENDPOINTS}.")
        sys.exit(1)

    try:
        dest = Path(args.dest)
        session: Optional[requests.Session]
        if args.workers and args.workers > 1:
            session = None
        else:
            session = _create_session(retries=args.retries)

        # Second run just to ensure all files are downloaded
        for i in range(2):
            if i == 1:
                logger.setLevel(logging.WARNING)
            downloader = BinanceDataDownloader(
                args.symbol,
                args.interval,
                args.start,
                args.end,
                dest,
                endpoint=args.endpoint,
                session=session,
                timeout=args.timeout,
                chunk_size=1024 * 64,
                overwrite=args.overwrite,
                max_workers=args.workers,
                retries=args.retries,
                backoff_factor=0.5,
            )
            downloader.download()
    except Exception as exc:
        logger.exception("Error: %s", exc)
        sys.exit(1)