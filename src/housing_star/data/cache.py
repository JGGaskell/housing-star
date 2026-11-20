"""On-disk Parquet cache of Land Registry transactions.

Layout (one directory per postcode prefix, one file per fetched date interval)::

    <root>/GL5/transactions_20000101_20260101.parquet
    <root>/GL6/8/transactions_20100101_20110101.parquet

Existing ``v2`` files can still be read. Extracts saved without a transaction id are skipped
because they may have collapsed distinct sales.

Only date intervals that are not yet covered are fetched. Every frame this cache returns has
passed :class:`housing_star.data.schema.TransactionSchema`, so series built from it are aggregated
from checked sales. Copies of one transaction id are merged with
:func:`housing_star.data.land_registry.resolve_transactions`.

How downloads are planned
    :meth:`TransactionCache.prefetch` takes every prefix a dataset needs at once. It rounds the
    window out to whole years (only past years, so a recent month is never marked as covered
    early), works out what each prefix is missing, and groups prefixes with the same gap into
    requests of about ``rows_per_request`` sales: several prefixes in one request, split into
    year ranges where the gap is long. Up to ``max_workers`` requests run at once. Results are
    split back to each prefix and written to that prefix's own files, so the layout above does not
    change.

Sectors from districts
    A sector (``"GL6 8"``) is covered wherever its district (``"GL6"``) is, and is read from the
    district's files, filtered to the sector. It is downloaded only for intervals no district file
    covers.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import platformdirs
import polars as pl

from housing_star._dates import parse_date
from housing_star.data.land_registry import (
    DEFAULT_ROW_LIMIT,
    PPDQuerySpec,
    fetch_transactions,
    normalise_transactions,
    prefix_filter,
    property_type_label,
    record_status_label,
    resolve_transactions,
    split_prefix,
)
from housing_star.data.schema import TRANSACTION_SCHEMA, validate_transactions

logger = logging.getLogger(__name__)

CACHE_ENV_VAR = "HOUSING_STAR_CACHE"
_FILE_PATTERN = re.compile(r"transactions_(?:v2_)?(\d{8})_(\d{8})\.parquet")

Fetcher = Callable[[PPDQuerySpec], pl.DataFrame]
Interval = tuple[date, date]

# Planning estimates of sales a year, by prefix depth (district, sector, deeper). A district averages
# under 400 non-new-build sales a year; these allow for busier ones. A request that turns out larger
# than its limit (twice the planned size) is split by date, not lost.
_SALES_PER_YEAR = {0: 500, 1: 80, 2: 15}
_MAX_PREFIXES_PER_REQUEST = 40


@dataclass
class DownloadStats:
    """What a cache has downloaded since it was created.

    ``requests`` counts calls to the fetcher. The default fetcher sends one SPARQL request per
    call, and more only when a date range has to be split.
    """

    requests: int = 0
    sales: int = 0
    seconds: float = 0.0


@dataclass(frozen=True)
class _Job:
    prefixes: tuple[str, ...]
    start: date
    end: date


def default_cache_dir() -> Path:
    """``$HOUSING_STAR_CACHE`` if set, otherwise the platform user-cache directory.

    That is ``~/Library/Caches/housing_star`` on macOS, ``~/.cache/housing_star`` on Linux and
    ``%LOCALAPPDATA%\\housing_star\\Cache`` on Windows.
    """
    override = os.environ.get(CACHE_ENV_VAR)
    if override:
        return Path(override)
    # Without appauthor=False, Windows repeats the name: housing_star\housing_star\Cache.
    return Path(platformdirs.user_cache_dir("housing_star", appauthor=False))


def normalise_prefix(prefix: str) -> str:
    """Upper-case and collapse whitespace so equivalent prefixes share a directory."""
    return " ".join(prefix.upper().split())


def prefix_dir(root: Path, prefix: str) -> Path:
    """``'GL5 2B'`` -> ``root/GL5/2/B``: the outward code, then one folder per remaining character."""
    outward, _, rest = normalise_prefix(prefix).partition(" ")
    if not outward:
        raise ValueError("postcode prefix is empty")
    return root.joinpath(outward, *rest.replace(" ", ""))


def merge_intervals(intervals: Sequence[Interval]) -> list[Interval]:
    """Merge overlapping or adjacent half-open intervals."""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def cache_filename(start: date, end: date) -> str:
    """Parquet name for one fetched interval."""
    return f"transactions_{start:%Y%m%d}_{end:%Y%m%d}.parquet"


def missing_intervals(request: Interval, covered: Sequence[Interval]) -> list[Interval]:
    """The parts of ``request`` not covered by the union of ``covered``."""
    cursor, stop = request
    gaps: list[Interval] = []
    for start, end in merge_intervals(covered):
        if end <= cursor or start >= stop:
            continue
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < stop:
        gaps.append((cursor, stop))
    return gaps


def ancestors(prefix: str) -> list[str]:
    """Shorter prefixes that contain ``prefix``, nearest first: ``"GL6 8A"`` -> ``["GL6 8", "GL6"]``."""
    outward, inward = split_prefix(prefix)
    return [f"{outward} {inward[:k]}".strip() for k in range(len(inward) - 1, -1, -1)]


def round_window(window: Interval, today: date) -> Interval:
    """Round ``window`` out to whole calendar years, but never past the start of this year."""
    start, end = window
    this_year = date(today.year, 1, 1)
    whole = date(end.year + (end != date(end.year, 1, 1)), 1, 1)
    if whole <= this_year:
        end = whole
    return date(start.year, 1, 1), end


def year_pieces(start: date, end: date) -> list[Interval]:
    """``[start, end)`` cut at every 1 January inside it."""
    cuts = [start, *(date(y, 1, 1) for y in range(start.year + 1, end.year + 1)), end]
    return [(a, b) for a, b in zip(cuts, cuts[1:], strict=False) if a < b]


def plan_requests(gaps: dict[str, list[Interval]], rows_per_request: int) -> list[_Job]:
    """Group the missing intervals of many prefixes into requests of about ``rows_per_request``.

    Prefixes missing the same interval share requests, up to a planned size; a long interval is cut
    into whole-year ranges. Estimates come from :data:`_SALES_PER_YEAR` by prefix depth.
    """
    by_gap: dict[Interval, list[str]] = {}
    for prefix, intervals in gaps.items():
        for interval in intervals:
            by_gap.setdefault(interval, []).append(prefix)
    jobs: list[_Job] = []
    for (start, end), prefixes in sorted(by_gap.items()):
        rate = {p: _SALES_PER_YEAR[min(len(split_prefix(p)[1]), 2)] for p in sorted(prefixes)}
        batches: list[list[str]] = []
        for prefix, per_year in rate.items():
            current = batches[-1] if batches else None
            if (
                current is None
                or len(current) >= _MAX_PREFIXES_PER_REQUEST
                or sum(rate[p] for p in current) + per_year > rows_per_request
            ):
                batches.append([prefix])
            else:
                current.append(prefix)
        pieces = year_pieces(start, end)
        for batch in batches:
            per_year = sum(rate[p] for p in batch)
            years_per_request = max(1, math.floor(rows_per_request / per_year))
            for k in range(0, len(pieces), years_per_request):
                chunk = pieces[k : k + years_per_request]
                jobs.append(_Job(tuple(batch), chunk[0][0], chunk[-1][1]))
    return jobs


@dataclass(frozen=True)
class TransactionCache:
    """Cache-first access to Land Registry transactions.

    Each ``(postcode prefix, date interval)`` is downloaded from HM Land Registry **once** and then
    read from local Parquet files. The first request for an area is slow because it downloads;
    repeat requests are near-instant, and widening the date window downloads only the new part.

    Parameters
    ----------
    root : cache directory; defaults to ``$HOUSING_STAR_CACHE`` or the user cache directory.
    fetcher : ``PPDQuerySpec -> DataFrame``; replace it to test offline or use another source. A
        spec can name several prefixes, and the frame must hold the sales of all of them.
    verbose : print a notice to stderr for every download (never for cache hits).
    max_workers : requests sent at once. HM Land Registry's endpoint answers 429 when pushed
        harder; every request then waits.
    rows_per_request : planned sales per request. A request is allowed twice this before its date
        range is split.
    """

    root: str | Path | None = None
    fetcher: Fetcher = fetch_transactions
    verbose: bool = True
    max_workers: int = 4
    rows_per_request: int = DEFAULT_ROW_LIMIT // 2
    stats: DownloadStats = field(default_factory=DownloadStats, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_workers < 1 or self.rows_per_request < 1:
            raise ValueError("max_workers and rows_per_request must be at least 1.")

    @property
    def directory(self) -> Path:
        """The cache root actually in use."""
        return Path(self.root) if self.root is not None else default_cache_dir()

    def cached_ranges(self, prefix: str) -> list[tuple[date, date, Path]]:
        """``(start, end, path)`` for each usable cached interval of ``prefix``."""
        folder = prefix_dir(self.directory, prefix)
        found: dict[Interval, Path] = {}
        # Unversioned names sort first and take precedence when both names cover one interval.
        for path in sorted(folder.glob("transactions_*.parquet")):
            match = _FILE_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            start, end = (date.fromisoformat(_iso(g)) for g in match.groups())
            interval = (start, end)
            if interval not in found and "transactionId" in pl.read_parquet_schema(path):
                found[interval] = path
        return [(start, end, path) for (start, end), path in sorted(found.items())]

    def missing(self, prefix: str, start: str | date, end: str | date) -> list[Interval]:
        """The date intervals of ``[start, end)`` that would have to be downloaded.

        A sector is covered wherever it or its district is cached.
        """
        window = (parse_date(start), parse_date(end))
        covered = [(s, e) for s, e, _, _ in self._sources(normalise_prefix(prefix))]
        return missing_intervals(window, covered)

    def _sources(self, prefix: str) -> list[tuple[date, date, Path, str]]:
        """Cached intervals usable for ``prefix``: its own, then each enclosing prefix's."""
        return [
            (start, end, path, owner)
            for owner in (prefix, *ancestors(prefix))
            for start, end, path in self.cached_ranges(owner)
        ]

    def transactions(self, prefix: str, start: str | date, end: str | date) -> pl.DataFrame:
        """Transactions for ``prefix`` in ``[start, end)``, fetching only what is missing.

        The returned frame has passed :func:`housing_star.data.schema.validate_transactions`.
        To fetch many prefixes, call :meth:`prefetch` with all of them first: it batches them.
        """
        window = (parse_date(start), parse_date(end))
        if window[1] <= window[0]:
            raise ValueError(f"end ({window[1]}) must be after start ({window[0]})")
        prefix = normalise_prefix(prefix)
        self.prefetch([prefix], *window)
        return self._read(prefix, window)

    def prefetch(self, prefixes: Iterable[str], start: str | date, end: str | date) -> None:
        """Download whatever ``prefixes`` are missing in ``[start, end)``, in batched requests.

        The window is rounded out to whole past years. A sector is skipped when its district is in
        the same call, since the district's files will cover it. Afterwards each prefix's adjacent
        cache files are merged into one, so a history downloaded in pieces is read in one go.
        """
        window = (parse_date(start), parse_date(end))
        if window[1] <= window[0]:
            raise ValueError(f"end ({window[1]}) must be after start ({window[0]})")
        wanted = list(dict.fromkeys(normalise_prefix(p) for p in prefixes))
        for prefix in wanted:
            split_prefix(prefix)
        wanted = [p for p in wanted if not set(ancestors(p)) & set(wanted)]
        window = round_window(window, date.today())
        gaps = {p: gap for p in wanted if (gap := self.missing(p, *window))}
        jobs = plan_requests(gaps, self.rows_per_request)
        if jobs:
            self._download(jobs, len(gaps))
        for prefix in wanted:
            self._compact(prefix)

    def _download(self, jobs: list[_Job], areas: int) -> None:
        """Run ``jobs`` (several at once when allowed) and write each result as it arrives."""
        self._announce(
            f"housing_star: downloading {areas} postcode area(s) from HM Land Registry in "
            f"{len(jobs)} request(s), up to {self.max_workers} at a time "
            f"(slow the first time; cached in {self.directory})"
        )
        started = time.perf_counter()
        if len(jobs) == 1 or self.max_workers == 1:
            for k, job in enumerate(jobs, 1):
                self._store(job, self._fetch(job), k, len(jobs))
        else:
            with ThreadPoolExecutor(self.max_workers) as pool:
                for k, (job, frame) in enumerate(
                    pool.map(lambda job: (job, self._fetch(job)), jobs), 1
                ):
                    self._store(job, frame, k, len(jobs))
        self.stats.seconds += time.perf_counter() - started

    def _fetch(self, job: _Job) -> pl.DataFrame:
        spec = PPDQuerySpec(
            job.prefixes,
            job.start.isoformat(),
            job.end.isoformat(),
            limit=2 * self.rows_per_request,
        )
        return self.fetcher(spec)

    def _store(self, job: _Job, frame: pl.DataFrame, k: int, total: int) -> None:
        """Split one request's sales by prefix and write each prefix's file for the job's dates."""
        frame = normalise_transactions(frame)
        self.stats.requests += 1
        self.stats.sales += frame.height
        for prefix in job.prefixes:
            self._write(prefix, job.start, job.end, frame.filter(prefix_filter(prefix)))
        self._announce(
            f"housing_star:   {k}/{total}: {', '.join(job.prefixes)} {job.start} to {job.end}, "
            f"{frame.height} sales"
        )

    def _announce(self, message: str) -> None:
        logger.info(message)
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def _write(self, prefix: str, start: date, end: date, frame: pl.DataFrame) -> None:
        folder = prefix_dir(self.directory, prefix)
        folder.mkdir(parents=True, exist_ok=True)
        # An empty frame is still written: it marks the interval as fetched.
        path = folder / cache_filename(start, end)
        with TemporaryDirectory(dir=folder) as staging_dir:
            staged = Path(staging_dir) / path.name
            frame.write_parquet(staged)  # already normalised and validated by _store
            staged.replace(path)

    def _read(self, prefix: str, window: Interval) -> pl.DataFrame:
        paths = [
            path
            for start, end, path, _ in self._sources(prefix)
            if start < window[1] and end > window[0]
        ]
        return _load(paths, prefix, window)

    def _compact(self, prefix: str) -> None:
        """Merge each run of adjacent or overlapping files of ``prefix`` into one file.

        Downloads write one file per request, so a long history can arrive in several pieces.
        Reading one file per run is faster. The merged file is written before the pieces are
        removed, so an interruption leaves the cache readable.
        """
        ranges = self.cached_ranges(prefix)
        for start, end in merge_intervals([(s, e) for s, e, _ in ranges]):
            pieces = [path for s, e, path in ranges if start <= s and e <= end]
            if len(pieces) < 2:
                continue
            self._write(prefix, start, end, _load(pieces, prefix, (start, end)))
            for path in pieces:
                if path.name != cache_filename(start, end):
                    path.unlink()


def _load(paths: Sequence[Path], prefix: str, window: Interval) -> pl.DataFrame:
    """The validated sales of ``prefix`` in ``window`` from cache files, one row per transaction."""
    columns = list(TRANSACTION_SCHEMA)
    # Scan files one by one: older cache files carry extra columns, which would break a
    # single multi-file scan.
    frames = [pl.scan_parquet(path).select(columns).filter(prefix_filter(prefix)) for path in paths]
    # Labels first, so a type URI and its short name are the same sale before ids are merged.
    # The date filter runs before that merge: only copies inside the requested window compete.
    return validate_transactions(
        resolve_transactions(
            pl.concat(frames)
            .with_columns(property_type_label(), record_status_label())
            .filter(pl.col("transactionDate").is_between(window[0], window[1], closed="left"))
            .collect()
        )
    )


def _iso(compact: str) -> str:
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"
