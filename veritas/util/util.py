import asyncio
import logging
import sys
from collections.abc import Awaitable
from datetime import date, datetime
from hashlib import blake2s
from typing import Iterable

import tqdm
from langcodes import Language
from pydantic import ValidationError

logger = logging.getLogger("VeriTaS")

QUARTERS = {
    1: ((1, 1), (3, 31)),
    2: ((4, 1), (6, 30)),
    3: ((7, 1), (9, 30)),
    4: ((10, 1), (12, 31)),
}


def get_quarter_date_range(year: int, quarter: int) -> tuple[date, date]:
    (start_month, start_day), (end_month, end_day) = QUARTERS[quarter]
    return date(year, start_month, start_day), date(year, end_month, end_day)


def get_quarters(start: tuple[int, int], end: tuple[int, int] | None = None) -> list[tuple[int, int]]:
    """Returns a list of (year, quarter) tuples from start to end (inclusive).
    If no end is specified, it returns quarters up to the current date."""
    if end is None:
        now = datetime.now()
        end = (now.year, (now.month - 1) // 3 + 1)

    quarters = []
    y, q = start
    while (y, q) <= end:
        quarters.append((y, q))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return quarters


class FileCache:
    """Remembers previously opened files.
    
    This is required to avoid re-reading files from disk which causes a "too many open files" error."""

    def __init__(self, ttl: int = 120):
        self.ttl = ttl
        self.cache = {}
        self._last_cleanup = datetime.now()
        self._cleanup_interval = 10  # Clean every 10 seconds

    def get(self, file_path: str) -> str:
        if (datetime.now() - self._last_cleanup).total_seconds() > self._cleanup_interval:
            self._clean_cache()
            self._last_cleanup = datetime.now()

        if file_path in self.cache:
            cached_file, timestamp = self.cache[file_path]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return cached_file
            else:
                # Lazy cleanup - remove just this expired entry
                del self.cache[file_path]

        content = self._load(file_path)
        self.cache[file_path] = (content, datetime.now())
        return content

    def _clean_cache(self):
        """Cleans up the cache by removing expired entries."""
        current_time = datetime.now()
        self.cache = {
            k: v for k, v in self.cache.items() if (current_time - v[1]).total_seconds() < self.ttl
        }

    def _load(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()


file_cache = FileCache()


def load_file(file_path: str) -> str:
    return file_cache.get(file_path)


def get_lang_iso_code(lang_name: str) -> str | None:
    """Turns a language name like 'German' into its ISO 639-1 code like 'de'."""
    return Language.find(lang_name).language


async def run_with_semaphore(
        tasks: Iterable[Awaitable], limit: int, show_progress: bool = False, progress_description: str | None = None
) -> list:
    """
    Runs asynchronous tasks with a concurrency limit.

    Args:
        tasks: The tasks to execute concurrently.
        limit: The maximum number of coroutines to run concurrently.
        show_progress: Whether to show a progress bar while executing tasks.
        progress_description: The message to display in the progress bar.

    Returns:
        list: A list of results returned by the tasks, order-preserved.
    """
    semaphore = asyncio.Semaphore(limit)  # Limit concurrent executions

    async def limited_coroutine(t: Awaitable):
        async with semaphore:
            return await t

    tasks = [asyncio.create_task(limited_coroutine(task)) for task in tasks]

    # Report completion status of tasks (if more than one task)
    if show_progress:
        # Send progress to stderr so normal output/errors on stdout aren't overwritten
        progress = tqdm.tqdm(total=len(tasks), desc=progress_description, file=sys.stderr)
        while progress.n < len(tasks):
            progress.n = sum(task.done() for task in tasks)
            progress.refresh()
            await asyncio.sleep(0.1)
        progress.close()

    return await asyncio.gather(*tasks)


def hash_int32(obj) -> int:
    """Deterministic hash returning a 32-bit signed integer."""
    digest = blake2s(str(obj).encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def date_or_datetime_to_str(date_or_datetime) -> str:
    if isinstance(date_or_datetime, date):
        return date_or_datetime.strftime("%B %d, %Y")
    elif isinstance(date_or_datetime, datetime):
        return date_or_datetime.strftime("%B %d, %Y, %H:%M:%S")
    elif date_or_datetime is None:
        return ""
    else:
        raise ValueError(f"Unsupported type: {type(date_or_datetime)}")


def validate(python_object, pydantic_model) -> bool:
    """Validates a Python object against a Pydantic model. Returns True if
    the object is valid, False otherwise."""
    try:
        pydantic_model(python_object)
        return True
    except ValidationError:
        return False


def get_quarter_start_date(start: tuple[int, int]) -> datetime:
    """Returns the start date for the given quarter."""
    start_date, _ = get_quarter_date_range(*start)
    return datetime.combine(start_date, datetime.min.time())
