"""Neighbourhood selection around a focal postcode."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from housing_star._types import FloatArray
from housing_star.data.postcodes import clean_code, infer_level, load_centroids


@dataclass(frozen=True)
class Neighbourhood:
    """Nodes (nearest first, the focal node at index 0) and their centroid coordinates."""

    keys: list[str]
    eastings: FloatArray
    northings: FloatArray


def select_neighbourhood(
    query: str,
    *,
    radius_m: float,
    max_nodes: int,
    centroids: pl.DataFrame | None = None,
) -> Neighbourhood:
    """Select the ``max_nodes`` nodes nearest to ``query`` within ``radius_m`` metres.

    Nodes live at the same level (district or sector) as ``query``.

    Raises
    ------
    ValueError
        If ``query`` is not in the centroid table.
    """
    level = infer_level(query)
    key = clean_code(query)
    table = (load_centroids() if centroids is None else centroids).filter(pl.col("level") == level)
    focal = table.filter(pl.col("key") == key)
    if focal.is_empty():
        raise ValueError(f"Postcode {query!r} not found among {level} centroids.")

    e = table["easting"].to_numpy()
    n = table["northing"].to_numpy()
    dist = np.hypot(e - focal["easting"][0], n - focal["northing"][0])
    order = np.argsort(dist, kind="stable")
    order = order[dist[order] <= radius_m][:max_nodes]
    return Neighbourhood(
        keys=table["key"].gather(order).to_list(),
        eastings=e[order],
        northings=n[order],
    )
