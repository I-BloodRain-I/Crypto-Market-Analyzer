"""
Download Binance futures kline .zip files by date range.

Usage examples:
    python data_downloader.py --start 01-01-2023 --end 05-01-2023
    python data_downloader.py --symbol BTCUSDT --interval 1m --start 01-01-2023 --end 05-01-2023 \
        --workers 4 --retries 5 --timeout 10 --dest data/raw --overwrite

Arguments:
    --symbol    Trading pair symbol (default: BTCUSDT)
    --interval  Kline interval, e.g. 1m, 1h, 1d (default: 1m)
    --start     Start date in DD-MM-YYYY (required)
    --end       End date in DD-MM-YYYY (required)
    --dest      Destination folder to save zips (default: data/raw)
    --retries   Number of request retries (default: 5)
    --timeout   Per-request timeout in seconds (default: 10.0)
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
RAW_FOLDER = DATA_FOLDER / "raw"
RAW_FOLDER.mkdir(exist_ok=True, parents=True)
PROCESSED_FOLDER = DATA_FOLDER / "processed"
PROCESSED_FOLDER.mkdir(exist_ok=True, parents=True)

BASIC_URL = "https://data.binance.vision/data/futures/um/daily"

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


def download_klines(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str,
    save_path: Path,
    session: Optional[requests.Session] = None,
    timeout: float = 10.0,
    chunk_size: int = 1024 * 64,
    overwrite: bool = False,
    max_workers: int = 4,
    retries: int = 5,
    backoff_factor: float = 0.5,
) -> List[Tuple[Path, str]]:
    """
    Download historical kline data from Binance.

    Args:
        symbol: Trading pair symbol, e.g. 'BTCUSDT'.
        interval: Kline interval, e.g. '1m', '1h', '1d'.
        start_str: Start date in 'DD-MM-YYYY' format.
        end_str: End date in 'DD-MM-YYYY' format.
        save_path: Directory where .zip files will be saved; created if missing.
        session: Optional requests.Session to reuse for sequential runs. If None and
            max_workers > 1, a per-thread session is created for each worker.
        timeout: Per-request timeout in seconds.
        chunk_size: Bytes to read per iteration when streaming responses.
        overwrite: If True, overwrite existing files; otherwise skip non-empty files.
        max_workers: Number of worker threads to use (1 = sequential). Clamped to
            the number of dates being downloaded.
        retries: Total number of request retries handled by urllib3 Retry.
        backoff_factor: Backoff factor for retries.

    Returns:
        A list of tuples (path, status) where status is one of:
        'downloaded', 'skipped', 'http_error', 'network_error'.

    Raises:
        ValueError: If start_date is after end_date.

    Notes:
        - Downloads stream to a temporary '.part' file and are atomically replaced on
          success to avoid partial files.
        - For multithreaded downloads, thread-local requests.Session objects are used
          to avoid sharing sessions across threads.
        - Existing non-empty files are skipped unless overwrite=True.
    """
    start_date = datetime.strptime(start_str, "%d-%m-%Y")
    end_date = datetime.strptime(end_str, "%d-%m-%Y")
    if start_date > end_date:
        raise ValueError("start date must be <= end date")

    save_path.mkdir(parents=True, exist_ok=True)

    dates = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur)
        cur += timedelta(days=1)

    if max_workers is None:
        max_workers = 1
    max_workers = max(1, min(max_workers, len(dates)))

    thread_local = threading.local()

    def _get_thread_session() -> requests.Session:
        if getattr(thread_local, "session", None) is None:
            thread_local.session = _create_session(retries=retries, backoff_factor=backoff_factor)
        return thread_local.session

    def _download_for_date(cur_date: datetime):
        date_str = cur_date.strftime("%Y-%m-%d")
        file_name = f"{symbol}-{interval}-{date_str}.zip"
        out_file = save_path / file_name
        url = f"{BASIC_URL}/klines/{symbol}/{interval}/{file_name}"

        if out_file.exists() and not overwrite and out_file.stat().st_size > 0:
            logger.info("Skipping existing file %s", file_name)
            return out_file, "skipped"

        if max_workers > 1:
            sess = _get_thread_session()
        else:
            sess = session or _create_session(retries=retries, backoff_factor=backoff_factor)

        tmp_file = out_file.with_suffix(".part")
        try:
            with sess.get(url, timeout=timeout, stream=True) as response:
                response.raise_for_status()
                with open(tmp_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
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

    results: List[Tuple[Path, str]] = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_download_for_date, d): d for d in dates}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as exc:
                    logger.exception("Download task raised: %s", exc)
    else:
        for d in dates:
            results.append(_download_for_date(d))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Binance futures kline .zip files by date range")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", default="1m", help="Kline interval, e.g. 1m, 1h, 1d")
    parser.add_argument("--start", required=True, help="Start date DD-MM-YYYY")
    parser.add_argument("--end", required=True, help="End date DD-MM-YYYY")
    parser.add_argument("--dest", default=str(RAW_FOLDER), help="Destination folder to save zips")
    parser.add_argument("--retries", type=int, default=5, help="Number of request retries")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--workers", type=int, default=4, help="Number of download threads to use (1 = sequential)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        dest = Path(args.dest)
        # For multithreaded runs we let each worker create its own session
        session: Optional[requests.Session]
        if args.workers and args.workers > 1:
            session = None
        else:
            session = _create_session(retries=args.retries)

        download_klines(
            args.symbol,
            args.interval,
            args.start,
            args.end,
            dest,
            session=session,
            timeout=args.timeout,
            overwrite=args.overwrite,
            max_workers=args.workers,
            retries=args.retries,
            backoff_factor=0.5,
        )
    except Exception as exc:
        logger.exception("Error: %s", exc)
        sys.exit(1)