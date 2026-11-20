"""Drawing for :meth:`PriceSeriesDataset.plot_neighbourhood`.

Kept here, and imported lazily by that method, so constructing a dataset does not import
matplotlib. Forecasting charts stay in the notebooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from housing_star.data.dataset import PriceSeriesDataset


def _matplotlib() -> Any:
    """Import matplotlib, or raise an actionable error when it is not installed."""
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise ImportError(
            "plot_neighbourhood needs matplotlib: pip install 'housing-star[plot]'."
        ) from exc
    return matplotlib


def plot_neighbourhood(ds: PriceSeriesDataset, ax: Axes) -> Axes:
    """Draw ``ds``'s nodes and spatial weights on ``ax``, and return ``ax``.

    The focal postcode is red and every other node is blue, at their centroids (kilometres).
    A grey line joins each pair; its opacity is ``min(1, 4 * max(w_ij, w_ji))``, so a closer
    neighbour, which has the larger weight, is drawn darker. A dataset with no focal postcode
    is drawn entirely in blue.

    Parameters
    ----------
    ax : a matplotlib axes. The caller creates the figure, so this can share a layout with
        other charts.

    Raises
    ------
    ValueError
        If ``ds`` has no centroid coordinates.
    ImportError
        If matplotlib is not installed.
    """
    matplotlib = _matplotlib()
    if ds.eastings is None or ds.northings is None:
        raise ValueError("plot_neighbourhood needs centroid coordinates (eastings/northings).")
    x, y = ds.eastings / 1_000, ds.northings / 1_000
    for i in range(ds.N):
        for j in range(i + 1, ds.N):
            ax.plot(
                [x[i], x[j]],
                [y[i], y[j]],
                color="grey",
                lw=1,
                alpha=min(1.0, 4 * max(ds.W[i, j], ds.W[j, i])),
            )
    focal = ds.target_idx if ds.target is not None else None
    red, blue = (matplotlib.colors.to_rgba(name) for name in ("tab:red", "tab:blue"))
    ax.scatter(x, y, c=[red if i == focal else blue for i in range(ds.N)], zorder=3)
    for i, label in enumerate(ds.postcodes):
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set(
        xlabel="Easting (km)",
        ylabel="Northing (km)",
        title="Spatial neighbourhood",
        aspect="equal",
    )
    return ax
