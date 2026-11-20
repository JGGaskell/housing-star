"""``PriceSeriesDataset``: the leak-free, model-ready view of a postcode neighbourhood."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import polars as pl

from housing_star._output import export
from housing_star._types import FloatArray, IntArray
from housing_star.config import AdjustmentConfig, CleaningConfig, GraphConfig
from housing_star.data.cache import TransactionCache
from housing_star.data.prep import panel_from_long, prepare_panel
from housing_star.data.schema import validate_panel
from housing_star.data.series import PriceMeasure, panel_series
from housing_star.spatial.graph import Neighbourhood, select_neighbourhood
from housing_star.spatial.weights import inverse_distance_weights

if TYPE_CHECKING:
    from matplotlib.axes import Axes

GrowthMethod = Literal["mean", "slope"]


def estimate_growth(log_prices: FloatArray, train_end: int, method: GrowthMethod) -> FloatArray:
    """Each node's average growth per period over the training columns ``[:train_end]``.

    ``"mean"`` is the mean period-to-period change in log price, which depends only on the first
    and last training value. ``"slope"`` is the least-squares slope of log price against time, which
    uses every training value. Fewer than two training periods give zero growth.
    """
    if train_end < 2:
        return np.zeros(log_prices.shape[0])
    window = log_prices[:, :train_end]
    if method == "mean":
        return np.asarray((window[:, -1] - window[:, 0]) / (train_end - 1), dtype=np.float64)
    t = np.arange(train_end, dtype=float)
    t = t - t.mean()
    return np.asarray((window - window.mean(axis=1, keepdims=True)) @ t / (t @ t), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class PriceSeriesDataset:
    r"""Prices for ``N`` neighbouring postcodes over ``T`` periods, split into train and hold-out.

    Models see log prices with each node's *training-window* mean removed:

    .. math:: y_{i,t} = \log P_{i,t} - \frac{1}{T_{\mathrm{train}}}\sum_{s < T_{\mathrm{train}}} \log P_{i,s}

    Parameters
    ----------
    prices : ``(N, T)`` strictly positive prices in pounds.
    W : ``(N, N)`` row-standardised spatial weights.
    postcodes : node labels; a merged node is labelled ``"A+B"``.
    times : ``T`` ISO bucket-start dates.
    train_end : columns ``[:train_end]`` are training, ``[train_end:]`` hold-out.
    target : the focal postcode, if any.
    eastings, northings : optional ``(N,)`` centroid coordinates in metres.

    growth_method : how a node's average growth is measured, see :func:`estimate_growth`.

    Attributes
    ----------
    y : ``(N, T)`` log-demeaned prices (derived).
    log_node_means : ``(N,)`` training-window mean log price per node (derived).
    growth_means : ``(N,)`` training-window average growth of log price per period (derived).
        Models describe growth *relative to* this average, so a forecast always starts from the
        latest observed price.
    """

    prices: FloatArray
    W: FloatArray
    postcodes: list[str]
    times: list[str]
    train_end: int
    target: str | None = None
    eastings: FloatArray | None = None
    northings: FloatArray | None = None
    growth_method: GrowthMethod = "mean"
    y: FloatArray = field(init=False, repr=False)
    log_node_means: FloatArray = field(init=False, repr=False)
    growth_means: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        N, T = self.prices.shape
        if self.W.shape != (N, N) or len(self.postcodes) != N or len(self.times) != T:
            raise ValueError("prices, W, postcodes and times have inconsistent shapes.")
        if not 1 <= self.train_end <= T:
            raise ValueError(f"train_end must be in [1, {T}]; got {self.train_end}.")
        if not (np.isfinite(self.prices).all() and (self.prices > 0).all()):
            raise ValueError("prices must be finite and strictly positive.")
        log_prices = np.log(self.prices)
        means = log_prices[:, : self.train_end].mean(axis=1)
        object.__setattr__(self, "log_node_means", means)
        object.__setattr__(self, "y", log_prices - means[:, None])
        object.__setattr__(
            self, "growth_means", estimate_growth(log_prices, self.train_end, self.growth_method)
        )

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_panel(
        cls,
        prices: pl.DataFrame,
        neighbourhood: Neighbourhood,
        *,
        target: str,
        holdout: int = 1,
        cleaning: CleaningConfig | None = None,
        eps_m: float = 10.0,
    ) -> PriceSeriesDataset:
        """Build a dataset from long-format prices without touching the network or disk.

        Parameters
        ----------
        prices : columns ``postcode, bucket_start (Date), price, n``. Checked with
            :func:`housing_star.data.schema.validate_panel` before cleaning. The bucket dates
            must be evenly spaced. Columns are the observed dates, and a lag of one column is
            one period, so a hole that makes those dates uneven is rejected.
            ``from_postcode`` passes the frame
            :func:`~housing_star.data.series.panel_series` already checked.
        neighbourhood : the nodes and their centroids.
        target : focal postcode; must survive cleaning (possibly inside a merged node).
        holdout : number of trailing periods reserved for evaluation.
        """
        prices = validate_panel(prices)
        panel = panel_from_long(
            prices, neighbourhood.keys, neighbourhood.eastings, neighbourhood.northings
        )
        panel = prepare_panel(panel, cleaning or CleaningConfig(), holdout)
        ds = cls(
            prices=panel.prices,
            W=inverse_distance_weights(panel.eastings, panel.northings, eps_m=eps_m),
            postcodes=panel.postcodes,
            times=panel.times,
            train_end=panel.prices.shape[1] - holdout,
            target=target,
            eastings=panel.eastings,
            northings=panel.northings,
        )
        if ds.N < 2:
            raise ValueError(f"Only {ds.N} node survived cleaning; STAR needs at least 2 nodes.")
        if ds.index_of(target) is None:
            raise ValueError(
                f"Target {target!r} was dropped during cleaning; nodes left: {ds.postcodes}."
            )
        return ds

    @classmethod
    def from_postcode(
        cls,
        postcode: str,
        *,
        start: str | date,
        end: str | date,
        freq: str = "1y",
        holdout: int = 1,
        graph: GraphConfig | None = None,
        cleaning: CleaningConfig | None = None,
        cache: TransactionCache | None = None,
        prices: PriceMeasure = "median",
        adjustment: AdjustmentConfig | None = None,
    ) -> PriceSeriesDataset:
        """Fetch (or read from cache) Land Registry data for ``postcode`` and its neighbours.

        Cached and newly downloaded sales are validated against
        :class:`~housing_star.data.schema.TransactionSchema` before they are aggregated.

        Parameters
        ----------
        postcode : district (``"GL5"``) or sector (``"GL6 8"``).
        start, end : half-open window ``[start, end)``.
        freq : bucket size, e.g. ``"1y"``. The observed buckets must be evenly spaced. A
            bucket with no sale in any node leaves a hole, and an uneven calendar is rejected.
        holdout : trailing periods reserved for evaluation; all cleaning ignores them.
        prices : ``"median"`` (default) or ``"adjusted"``, a composition-adjusted index that holds
            the mix of streets and property types fixed. Its postcode and type effects are
            estimated on the training periods only. See :mod:`housing_star.data.adjusted`.
        adjustment : trimming and solver settings for ``prices="adjusted"``.
        """
        graph = graph or GraphConfig()
        hood = select_neighbourhood(postcode, radius_m=graph.radius_m, max_nodes=graph.max_nodes)
        panel = panel_series(
            hood.keys,
            start=start,
            end=end,
            freq=freq,
            cache=cache,
            prices=prices,
            holdout=holdout,
            adjustment=adjustment,
        )
        return cls.from_panel(
            panel, hood, target=postcode, holdout=holdout, cleaning=cleaning, eps_m=graph.eps_m
        )

    # ------------------------------------------------------------------ shape and views

    @property
    def N(self) -> int:
        """Number of nodes."""
        return int(self.prices.shape[0])

    @property
    def T(self) -> int:
        """Number of periods."""
        return int(self.prices.shape[1])

    @property
    def n_eval(self) -> int:
        """Number of hold-out periods."""
        return self.T - self.train_end

    @overload
    def to_frame(self, as_pandas: Literal[False] = False) -> pl.DataFrame: ...
    @overload
    def to_frame(self, as_pandas: bool) -> Any: ...
    def to_frame(self, as_pandas: bool = False) -> Any:
        """The dataset as a tidy table, one row per ``(postcode, time)``.

        Columns: ``postcode, time, price`` (pounds), ``y`` (log, demeaned: what models see) and
        ``is_holdout``. Returns a polars DataFrame, or a pandas one if ``as_pandas`` is true.
        """
        n, t = self.N, self.T
        frame = pl.DataFrame(
            {
                "postcode": np.repeat(self.postcodes, t),
                "time": np.tile(self.times, n),
                "price": self.prices.ravel(),
                "y": self.y.ravel(),
                "is_holdout": np.tile(np.arange(t) >= self.train_end, n),
            }
        ).with_columns(pl.col("time").str.to_date())
        return export(frame, as_pandas)

    def plot_neighbourhood(self, ax: Axes) -> Axes:
        """Draw this neighbourhood on ``ax`` and return ``ax``.

        The focal postcode is red and every other node is blue, at their centroids
        (kilometres). A grey line joins each pair; its opacity is
        ``min(1, 4 * max(w_ij, w_ji))``, so a closer neighbour is drawn darker. A dataset
        with no focal postcode is drawn entirely in blue.

        Matplotlib is imported only when this is called (``pip install housing-star[plot]``).
        The caller creates the figure, so the map can sit beside other charts.

        Raises
        ------
        ValueError
            If this dataset has no centroid coordinates.
        ImportError
            If matplotlib is not installed.
        """
        from housing_star.plot import plot_neighbourhood

        return plot_neighbourhood(self, ax)

    def y_train(self) -> FloatArray:
        """Training columns of ``y``."""
        return self.y[:, : self.train_end]

    def y_eval(self) -> FloatArray:
        """Hold-out columns of ``y``."""
        return self.y[:, self.train_end :]

    def to_abs(self, y: FloatArray) -> FloatArray:
        """Invert the transform: log-demeaned values -> prices in pounds."""
        return np.exp(y + self.log_node_means[:, None])

    def window(self, train_end: int, end: int) -> PriceSeriesDataset:
        """Keep periods ``[:end]`` with a new split; the node means and average growth are
        recomputed on ``[:train_end]``.

        This is how cross-validation folds are made without leaking later periods.
        """
        if not 1 <= train_end <= end <= self.T:
            raise ValueError(f"Need 1 <= train_end <= end <= {self.T}; got {train_end}, {end}.")
        return replace(
            self, prices=self.prices[:, :end], times=self.times[:end], train_end=train_end
        )

    def with_holdout(self, n: int) -> PriceSeriesDataset:
        """Same periods, with the last ``n`` reserved for evaluation."""
        return self.window(self.T - n, self.T)

    def train_only(self) -> PriceSeriesDataset:
        """Drop the hold-out columns entirely."""
        return self.window(self.train_end, self.train_end)

    # ------------------------------------------------------------------ node lookup

    def index_of(self, postcode: str) -> int | None:
        """Row of ``postcode``, matching merged labels (``"A+B"``) by component; else ``None``."""
        for i, label in enumerate(self.postcodes):
            if postcode == label or postcode in label.split("+"):
                return i
        return None

    @property
    def target_idx(self) -> int:
        """Row of the focal postcode."""
        if self.target is None:
            raise ValueError("This dataset has no target postcode.")
        idx = self.index_of(self.target)
        if idx is None:
            raise ValueError(f"Target {self.target!r} is not among the nodes {self.postcodes}.")
        return idx

    def neighbour_order(self, idx: int) -> IntArray:
        """Indices of the other nodes, nearest centroid first."""
        if self.eastings is None or self.northings is None:
            raise ValueError("neighbour_order needs centroid coordinates (eastings/northings).")
        dist = np.hypot(self.eastings - self.eastings[idx], self.northings - self.northings[idx])
        dist[idx] = np.inf
        return np.argsort(dist, kind="stable")[: self.N - 1]
