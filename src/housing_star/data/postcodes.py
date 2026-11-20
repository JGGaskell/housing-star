"""UK postcode parsing and the bundled district/sector centroid table.

Nodes of the spatial graph are postcode *districts* (``"GL5"``) or *sectors* (``"GL6 8"``).
Centroids are the mean Ordnance Survey easting/northing of the unit postcodes inside each area and
are shipped with the package (see ``ATTRIBUTION.md``).
"""

from __future__ import annotations

import re
from functools import cache
from importlib import resources
from typing import Literal

import polars as pl

Level = Literal["district", "sector"]

_DISTRICT = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]?$")


def clean_code(code: str) -> str:
    """Upper-case a user-supplied code and collapse internal whitespace."""
    return " ".join(code.upper().split())


def infer_level(code: str) -> Level:
    """Return ``"district"`` for ``"GL6"`` and ``"sector"`` for ``"GL6 8"``.

    Raises
    ------
    ValueError
        If ``code`` is neither (unit postcodes and bare areas are not supported).
    """
    parts = clean_code(code).split(" ")
    if len(parts) == 1 and _DISTRICT.match(parts[0]):
        return "district"
    if len(parts) == 2 and _DISTRICT.match(parts[0]) and re.fullmatch(r"\d", parts[1]):
        return "sector"
    raise ValueError(
        f"Unsupported postcode {code!r}: expected a district ('GL5') or sector ('GL6 8')."
    )


def node_key_expr(column: str, level: Level) -> pl.Expr:
    """Polars expression mapping unit postcodes (``"GL5 2AD"``) to their node key at ``level``."""
    unit = pl.col(column).str.to_uppercase().str.replace_all(r"\s+", " ").str.strip_chars()
    outward = unit.str.split(" ").list.get(0)
    if level == "district":
        return outward
    return outward + pl.lit(" ") + unit.str.split(" ").list.get(1).str.slice(0, 1)


@cache
def load_centroids() -> pl.DataFrame:
    """Return the bundled centroid table: ``level, key, easting, northing, n_units``."""
    path = resources.files("housing_star.data").joinpath("_resources/centroids.parquet")
    with resources.as_file(path) as file:
        return pl.read_parquet(file)
