"""Client for the HM Land Registry Price Paid Data (PPD) SPARQL endpoint.

How the data are extracted
--------------------------
One request covers one or more postcode prefixes (districts such as ``"GL5"``, sectors such as
``"GL6 8"``) over one half-open date interval. :func:`fetch_transactions` sends it to
``landregistry.data.gov.uk`` and returns every matching sale. Each sale keeps its transaction id,
price, date, unit postcode, property type and record status, and nothing else. By default the
query keeps residential sales only (detached, semi-detached, terraced, flat/maisonette) and
**excludes new builds**. :func:`normalise_transactions` cleans the table, drops and logs any sale
that would fail :class:`housing_star.data.schema.TransactionSchema`, collapses the rest to one row
per transaction id and validates the result, so only checked sales are cached and one malformed
record cannot block a district. Two flats sold on the same day for the same price stay two sales,
because they have different transaction ids. A ``change`` replaces an ``add`` for that id; a
``delete`` removes the sale.

Why requests are split by date, not paged
    The endpoint answers a ``LIMIT``/``OFFSET`` page by re-running the whole postcode search and
    sorting every match, so each page costs as much as the whole answer and long histories ran
    into its 90-second gateway limit. A request here asks for ``limit + 1`` rows with no ordering.
    If more than ``limit`` come back, the date interval is halved and each half requested on its
    own, so no request repeats another's work. :class:`housing_star.data.cache.TransactionCache`
    plans requests of a size that rarely needs splitting.

Several prefixes in one request
    The postcode search is a text index. ``postcode:GL5`` matches the district exactly (``GL50`` is
    a different token), and ``(+postcode:GL6 +postcode:8*)`` matches sector ``GL6 8`` exactly. One
    request ORs several of these together, and :func:`prefix_filter` assigns each returned sale to
    the prefixes it belongs to.

Timeouts, connection errors and HTTP 429/503 responses are retried with exponential backoff. A 429 pauses every
thread that is about to send a request, not only the one that received it.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

import polars as pl
import requests

from housing_star.data.schema import (
    PROPERTY_TYPES,
    TRANSACTION_ID_PATTERN,
    TRANSACTION_SCHEMA,
    UNIT_POSTCODE_PATTERN,
    validate_transactions,
)

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://landregistry.data.gov.uk/landregistry/sparql"
_HEADERS = {
    "Accept": "application/sparql-results+json",
    "Content-Type": "application/sparql-query",
    "User-Agent": "housing-star/0.1",
}
_RETRYABLE_STATUS = {429, 503}


def property_type_label() -> pl.Expr:
    """The readable type label: the last segment of a Land Registry type URI, upper-cased.

    ``"http://landregistry.data.gov.uk/def/common/terraced"`` and a label that is already
    ``"TERRACED"`` both become ``"TERRACED"``. Older cache files stored the whole URI in upper
    case; the same expression reduces those on read.
    """
    return pl.col("propertyType").str.strip_chars().str.split("/").list.last().str.to_uppercase()


def transaction_id_label() -> pl.Expr:
    """Upper-case the transaction id and strip one surrounding pair of braces."""
    return (
        pl.col("transactionId")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace(r"^\{(.+)\}$", "$1")
        .str.to_uppercase()
    )


def record_status_label() -> pl.Expr:
    """Map a record-status URI or notation (``A``/``C``/``D``) to ``ADD``, ``CHANGE`` or ``DELETE``.

    The vocabulary is the Price Paid one (``ppi:add``, ``ppi:change``, ``ppi:delete``). Anything
    else becomes null and :func:`resolve_transactions` rejects the frame.
    """
    token = (
        pl.col("recordStatus")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .str.split("/")
        .list.last()
        .str.split("#")
        .list.last()
        .str.split(":")
        .list.last()
    )
    return token.replace_strict(
        {
            "A": "ADD",
            "ADD": "ADD",
            "C": "CHANGE",
            "CHANGE": "CHANGE",
            "D": "DELETE",
            "DELETE": "DELETE",
        },
        default=None,
    )


def resolve_transactions(frame: pl.DataFrame) -> pl.DataFrame:
    """One row per ``transactionId``, applying Price Paid record-status instructions.

    The vocabulary defines ``change`` as a replacement of the stored record with that id, and a
    delete as a removal of it. When several copies are combined (overlapping pages or cache
    files):

    * a ``DELETE`` removes the sale, whichever other status is present;
    * otherwise a ``CHANGE`` replaces an ``ADD``;
    * several rows with the same status keep the last in input order.

    Raises
    ------
    ValueError
        If ``recordStatus`` is missing or not one of ``ADD``, ``CHANGE`` and ``DELETE``.
    """
    if frame.is_empty():
        return frame
    known = pl.col("recordStatus").is_in(["ADD", "CHANGE", "DELETE"])
    unknown = frame.filter(pl.col("recordStatus").is_null() | ~known).get_column("recordStatus")
    if unknown.len():
        found = sorted({str(value) for value in unknown.unique().to_list()})
        raise ValueError(f"recordStatus must be ADD, CHANGE or DELETE; got {found}.")
    return (
        frame.with_row_index("_order")
        .with_columns(
            pl.col("recordStatus")
            .replace_strict({"ADD": 0, "CHANGE": 1, "DELETE": 2})
            .alias("_rank")
        )
        .sort(["transactionId", "_rank", "_order"])
        .unique(subset=["transactionId"], keep="last")
        .filter(pl.col("recordStatus") != "DELETE")
        .sort("_order")
        .drop("_order", "_rank")
    )


DEFAULT_ROW_LIMIT = 30_000
"""Rows one request may return before its date interval is split. About 15 seconds of transfer,
well inside the endpoint's 90-second gateway limit."""

_PREFIX = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)(?: (\d[A-Z]{0,2}))?$")


def split_prefix(prefix: str) -> tuple[str, str]:
    """``"GL6 8"`` -> ``("GL6", "8")``; ``"GL5"`` -> ``("GL5", "")``.

    Raises
    ------
    ValueError
        If ``prefix`` is not an outward code, optionally followed by the start of an inward code.
        The check also keeps anything but a postcode out of the query text.
    """
    cleaned = " ".join(prefix.upper().split())
    match = _PREFIX.fullmatch(cleaned)
    if match is None:
        raise ValueError(
            f"Unsupported postcode prefix {prefix!r}: expected a district ('GL5'), a sector "
            "('GL6 8') or a longer start of a postcode ('GL6 8A')."
        )
    return match.group(1), match.group(2) or ""


@dataclass(frozen=True)
class PPDQuerySpec:
    """One request: sales for ``prefixes`` in the half-open window ``[start_date, end_date)``.

    ``prefixes`` may be a single string. ``limit`` is the most rows one request returns; above it
    :func:`fetch_transactions` splits the date window.
    """

    prefixes: tuple[str, ...]
    start_date: str
    end_date: str
    non_new_build_only: bool = True
    limit: int = DEFAULT_ROW_LIMIT

    def __post_init__(self) -> None:
        raw = (self.prefixes,) if isinstance(self.prefixes, str) else tuple(self.prefixes)
        if not raw:
            raise ValueError("prefixes is empty")
        cleaned = tuple(dict.fromkeys(" ".join(p.upper().split()) for p in raw))
        for prefix in cleaned:
            split_prefix(prefix)
        if self.limit < 1:
            raise ValueError("limit must be at least 1")
        object.__setattr__(self, "prefixes", cleaned)


def text_query(prefixes: tuple[str, ...]) -> str:
    """The text-index query matching every sale in ``prefixes``, exactly."""
    parts = []
    for prefix in prefixes:
        outward, inward = split_prefix(prefix)
        if inward:
            parts.append(f"(+postcode:{outward} +postcode:{inward.lower()}*)")
        else:
            parts.append(f"postcode:{outward}")
    return " OR ".join(parts)


def prefix_filter(prefix: str, column: str = "propertyAddress_postcode") -> pl.Expr:
    """True for unit postcodes (``"GL6 8AU"``) inside ``prefix`` (``"GL6"``, ``"GL6 8"``...)."""
    outward, inward = split_prefix(prefix)
    unit = pl.col(column)
    return (unit.str.split(" ").list.get(0, null_on_oob=True) == outward) & unit.str.split(
        " "
    ).list.get(1, null_on_oob=True).str.starts_with(inward)


def build_ppd_query(spec: PPDQuerySpec) -> str:
    """Render the SPARQL query for ``spec`` (residential sales, optionally excluding new builds).

    It returns only the fields the cache keeps, asks for ``limit + 1`` rows so that a full answer
    can be recognised, and has no ``ORDER BY``: sorting is what made paged requests expensive.
    """
    new_build = "FILTER ( ?newBuild = false )" if spec.non_new_build_only else ""
    types = "lrcommon:detached, lrcommon:semi-detached, lrcommon:terraced, lrcommon:flat-maisonette"
    return f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ppd: <http://landregistry.data.gov.uk/def/ppi/>
PREFIX lrcommon: <http://landregistry.data.gov.uk/def/common/>

SELECT DISTINCT ?transactionId ?pricePaid ?transactionDate ?propertyType ?recordStatus
       ?propertyAddress_postcode
WHERE {{
  ?propertyAddress <http://jena.apache.org/text#query> ( "{text_query(spec.prefixes)}" 3000000 ) .

  ?id rdf:type ppd:TransactionRecord ;
     ppd:propertyAddress ?propertyAddress ;
     ppd:transactionDate ?transactionDate ;
     ppd:pricePaid ?pricePaid ;
     ppd:newBuild ?newBuild ;
     ppd:propertyType ?propertyType ;
     ppd:transactionCategory ?transactionCategory ;
     ppd:hasTransaction ?hasTransaction ;
     ppd:transactionId ?transactionId .

  FILTER ( ?transactionDate >= "{spec.start_date}"^^xsd:date )
  FILTER ( ?transactionDate < "{spec.end_date}"^^xsd:date )
  {new_build}
  FILTER ( ?propertyType IN ({types}) )

  OPTIONAL {{ ?propertyAddress lrcommon:postcode ?propertyAddress_postcode }}
  OPTIONAL {{ ?id ppd:recordStatus ?recordStatus }}
}}
LIMIT {spec.limit + 1}
""".strip()


class _RequestGate:
    """Holds every thread back after a 429, until the endpoint's requested pause is over."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open_at = 0.0

    def wait(self) -> None:
        with self._lock:
            delay = self._open_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def close_for(self, seconds: float) -> None:
        with self._lock:
            self._open_at = max(self._open_at, time.monotonic() + seconds)

    def reset(self) -> None:
        with self._lock:
            self._open_at = 0.0


_GATE = _RequestGate()


def run_sparql(
    query: str,
    *,
    session: requests.Session,
    timeout_s: int = 180,
    max_retries: int = 6,
    backoff_base: float = 1.8,
) -> dict[str, Any]:
    """POST ``query`` and return the SPARQL JSON, retrying timeouts, connection errors and HTTP 429/503.

    A 429 (too many requests) closes a gate shared by every thread for the backoff delay, so all
    requests in flight from this process slow down together. Timeouts and 503s back off only the
    thread that saw them.
    """
    last_error: Exception = RuntimeError("max_retries must be >= 1")
    for attempt in range(1, max_retries + 1):
        retry_after: str | None = None
        throttled = False
        _GATE.wait()
        try:
            response = session.post(
                SPARQL_ENDPOINT, data=query.encode(), headers=_HEADERS, timeout=timeout_s
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return payload
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc  # a timeout, or a dropped connection or failed DNS lookup
        except requests.exceptions.HTTPError as exc:
            if exc.response is None or exc.response.status_code not in _RETRYABLE_STATUS:
                raise
            last_error = exc
            retry_after = exc.response.headers.get("Retry-After")
            throttled = exc.response.status_code == 429
        if attempt < max_retries:
            delay = (
                float(retry_after)
                if retry_after is not None and retry_after.isdigit()
                else backoff_base**attempt + random.uniform(0.0, 1.0)
            )
            logger.warning(
                "SPARQL attempt %d/%d failed; retrying in %.1fs", attempt, max_retries, delay
            )
            if throttled:  # every thread waits at the gate, this one included
                _GATE.close_for(delay)
            else:
                time.sleep(delay)
    raise last_error


def sparql_json_to_frame(payload: dict[str, Any]) -> pl.DataFrame:
    """Convert SPARQL JSON bindings to a frame of string columns."""
    variables = payload["head"]["vars"]
    rows = [
        {name: binding.get(name, {}).get("value") for name in variables}
        for binding in payload["results"]["bindings"]
    ]
    return pl.DataFrame(rows, schema=dict.fromkeys(variables, pl.Utf8))


def _row_checks() -> dict[str, pl.Expr]:
    """Per-row versions of the :class:`~housing_star.data.schema.TransactionSchema` value checks."""
    return {
        "pricePaid": pl.col("pricePaid").is_not_null() & (pl.col("pricePaid") > 0),
        "transactionDate": pl.col("transactionDate").is_not_null(),
        "propertyAddress_postcode": pl.col("propertyAddress_postcode")
        .str.contains(UNIT_POSTCODE_PATTERN)
        .fill_null(False),
        "propertyType": pl.col("propertyType").is_in(list(PROPERTY_TYPES)).fill_null(False),
        "transactionId": pl.col("transactionId")
        .str.contains(TRANSACTION_ID_PATTERN)
        .fill_null(False),
        "recordStatus": pl.col("recordStatus").is_not_null(),
    }


def drop_invalid_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove sales that would fail the schema's value checks, logging how many and why.

    One malformed record (a missing price, a postcode that is not a full unit code, an
    unrecognised record status...) is dropped on its own, so it cannot stop a whole district
    from being downloaded and cached. Each dropped row is counted once per failed field in a
    ``WARNING`` on the ``housing_star.data.land_registry`` logger.
    """
    checks = _row_checks()
    flagged = frame.with_columns(
        **{f"_ok_{name}": expr for name, expr in checks.items()}
    ).with_columns(pl.all_horizontal(*(f"_ok_{name}" for name in checks)).alias("_ok"))
    bad = flagged.filter(~pl.col("_ok"))
    if bad.height:
        reasons = {
            name: int((~bad[f"_ok_{name}"]).sum()) for name in checks if (~bad[f"_ok_{name}"]).any()
        }
        logger.warning(
            "Dropped %d of %d Land Registry sale(s) that failed validation (failures by field: %s)",
            bad.height,
            frame.height,
            reasons,
        )
    return flagged.filter(pl.col("_ok")).select(frame.columns)


def normalise_transactions(frame: pl.DataFrame) -> pl.DataFrame:
    """Reduce an API frame to :class:`~housing_star.data.schema.TransactionSchema`.

    Values are cleaned and typed. Rows that would break the schema (no price, an unparseable
    date, a postcode that is not a full unit code, an unknown property type, transaction id or
    record status) are dropped and logged by :func:`drop_invalid_rows`, so one bad record cannot
    block a district. :func:`resolve_transactions` then keeps one row per transaction id, and
    the result is validated before it is returned, so the cache only stores checked sales.
    An empty frame is valid.

    Raises
    ------
    ValueError
        If a required column is missing.
    pandera.errors.SchemaError
        If the cleaned frame still breaks
        :class:`~housing_star.data.schema.TransactionSchema` (for example a wrong column type).
    """
    if frame.is_empty():
        cleaned = pl.DataFrame(schema=TRANSACTION_SCHEMA)
    else:
        missing = [name for name in TRANSACTION_SCHEMA if name not in frame.columns]
        if missing:
            raise ValueError(f"PPD frame missing required columns: {missing}")
        cleaned = frame.select(list(TRANSACTION_SCHEMA)).with_columns(
            pl.col("pricePaid").cast(pl.Int64, strict=False),
            pl.col("transactionDate").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            pl.col("propertyAddress_postcode")
            .str.to_uppercase()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars(),
            property_type_label(),
            transaction_id_label(),
            record_status_label(),
        )
        cleaned = drop_invalid_rows(cleaned)
    return validate_transactions(resolve_transactions(cleaned))


def fetch_transactions(spec: PPDQuerySpec, *, timeout_s: int = 180) -> pl.DataFrame:
    """Every sale for ``spec``, normalised. A window with more than ``spec.limit`` sales is halved
    and each half requested on its own, as often as needed.

    Raises
    ------
    RuntimeError
        If a single day holds more than ``spec.limit`` sales, so the window cannot be split further.
    """
    with requests.Session() as session:
        raw = _fetch_window(spec, session=session, timeout_s=timeout_s)
    return normalise_transactions(raw)


def _fetch_window(spec: PPDQuerySpec, *, session: requests.Session, timeout_s: int) -> pl.DataFrame:
    rows = sparql_json_to_frame(
        run_sparql(build_ppd_query(spec), session=session, timeout_s=timeout_s)
    )
    if rows.height <= spec.limit:
        return rows
    start, end = date.fromisoformat(spec.start_date), date.fromisoformat(spec.end_date)
    if (end - start).days <= 1:
        raise RuntimeError(
            f"More than {spec.limit} sales on {start} for {', '.join(spec.prefixes)}; "
            "raise PPDQuerySpec.limit."
        )
    middle = start + (end - start) // 2
    logger.info(
        "%d+ sales for %s %s to %s; splitting at %s",
        spec.limit,
        ", ".join(spec.prefixes),
        start,
        end,
        middle,
    )
    return pl.concat(
        [
            _fetch_window(
                replace(spec, end_date=middle.isoformat()), session=session, timeout_s=timeout_s
            ),
            _fetch_window(
                replace(spec, start_date=middle.isoformat()), session=session, timeout_s=timeout_s
            ),
        ]
    )
