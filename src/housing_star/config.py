"""Configuration for graph construction, panel cleaning and the adjusted price index."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    """How the spatial neighbourhood around the focal postcode is built.

    Parameters
    ----------
    radius_m : include nodes whose centroid is within this many metres of the focal node.
    max_nodes : keep at most this many nodes (nearest first, focal node included).
    eps_m : distance offset in the inverse-distance weights; keeps weights finite.
    """

    radius_m: float = 5_000.0
    max_nodes: int = 20
    eps_m: float = 10.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or self.eps_m <= 0:
            raise ValueError("radius_m and eps_m must be positive.")
        if self.max_nodes < 2:
            raise ValueError("max_nodes must be at least 2: STAR needs neighbours.")


@dataclass(frozen=True)
class CleaningConfig:
    """Quality thresholds applied to the training window only (never to the hold-out).

    Parameters
    ----------
    min_coverage : minimum fraction of training periods with an observed price.
    min_mean_n : minimum mean number of sales per observed period.
    min_nodes : thin nodes are merged into their nearest neighbour until this few remain.
    max_gap : drop nodes with more than this many consecutive missing training periods.
    keep_extrapolated : keep nodes whose training edges had to be extrapolated.
    """

    min_coverage: float = 0.6
    min_mean_n: float = 5.0
    min_nodes: int = 3
    max_gap: int = 2
    keep_extrapolated: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1].")
        if self.min_nodes < 2 or self.max_gap < 0:
            raise ValueError("min_nodes must be >= 2 and max_gap >= 0.")


@dataclass(frozen=True)
class AdjustmentConfig:
    """Settings for the composition-adjusted price index (``prices="adjusted"``).

    Parameters
    ----------
    trim : a sale more than this many log points from its fitted value is dropped. The default,
        0.7, is about a factor of two either way.
    """

    trim: float = 0.7

    def __post_init__(self) -> None:
        if self.trim <= 0:
            raise ValueError("trim must be positive.")
