"""Build the bundled district/sector centroid table from Ordnance Survey Code-Point Open.

Usage::

    python scripts/build_centroids.py path/to/codepo.parquet

The input must have the columns ``Postcode``, ``Eastings`` and ``Northings`` (one row per unit
postcode). The output is written to ``src/housing_star/data/_resources/centroids.parquet``.
``ATTRIBUTION.md`` gives the download, the release used and how to turn its CSV files into that
input.

Code-Point Open gives postcodes it cannot place (positional quality 90, mostly non-geographic
postcodes such as PO boxes and large-user codes) the coordinates (0, 0), a point in the sea south
west of the Isles of Scilly. Those rows are dropped before averaging, so they neither drag a
district's centroid towards (0, 0) nor create a node there. An area with no placed postcode is
left out of the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

OUTPUT = Path(__file__).resolve().parents[1] / "src/housing_star/data/_resources/centroids.parquet"


def build(source: Path) -> pl.DataFrame:
    """Aggregate unit postcodes to district and sector centroids."""
    units = (
        pl.scan_parquet(source)
        .select(
            pl.col("Postcode").str.to_uppercase().str.replace_all(r"\s+", " ").str.strip_chars(),
            pl.col("Eastings").cast(pl.Float64).alias("easting"),
            pl.col("Northings").cast(pl.Float64).alias("northing"),
        )
        .filter((pl.col("easting") != 0) | (pl.col("northing") != 0))
        .with_columns(
            pl.col("Postcode").str.split(" ").list.get(0).alias("outward"),
            pl.col("Postcode").str.split(" ").list.get(1).str.slice(0, 1).alias("sector_digit"),
        )
        .collect()
    )
    keys = {
        "district": pl.col("outward"),
        "sector": pl.col("outward") + pl.lit(" ") + pl.col("sector_digit"),
    }
    parts = [
        units.group_by(key.alias("key"))
        .agg(
            pl.col("easting").mean(),
            pl.col("northing").mean(),
            pl.len().cast(pl.UInt32).alias("n_units"),
        )
        .with_columns(pl.lit(level).alias("level"))
        for level, key in keys.items()
    ]
    return (
        pl.concat(parts)
        .select("level", "key", "easting", "northing", "n_units")
        .sort("level", "key")
    )


if __name__ == "__main__":
    build(Path(sys.argv[1])).write_parquet(OUTPUT, compression="zstd")
    print(f"Wrote {OUTPUT}")
