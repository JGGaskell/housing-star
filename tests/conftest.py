"""Shared synthetic fixtures. No test touches the network or the user's cache."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import polars as pl
import pytest

from housing_star.data.dataset import PriceSeriesDataset
from housing_star.data.land_registry import PPDQuerySpec
from housing_star.data.schema import TRANSACTION_SCHEMA
from housing_star.models.star.estimation import StarFit
from housing_star.models.star.simulation import simulate_log_prices
from housing_star.spatial.weights import inverse_distance_weights

DatasetFactory = Callable[..., PriceSeriesDataset]


@pytest.fixture
def make_dataset() -> DatasetFactory:
    """Build a synthetic STAR dataset: nodes on a line, 1 km apart, node 0 is the target."""

    def build(
        n_nodes: int = 6,
        n_periods: int = 24,
        holdout: int = 3,
        rho: float = 0.3,
        phi: float = 0.4,
        theta: float = 0.2,
        seed: int = 0,
    ) -> PriceSeriesDataset:
        eastings = np.arange(n_nodes) * 1_000.0
        northings = np.zeros(n_nodes)
        W = inverse_distance_weights(eastings, northings)
        growth = 0.02 + 0.005 * np.arange(n_nodes)  # each node trends at its own rate
        y = simulate_log_prices(
            W,
            n_periods,
            growth=growth,
            rho=rho,
            phi=np.array([phi]),
            theta=np.array([theta]),
            seed=seed,
        )
        level = 12.0 + 0.1 * np.arange(n_nodes)[:, None]
        return PriceSeriesDataset(
            prices=np.exp(y + level),
            W=W,
            postcodes=[f"PC{i}" for i in range(n_nodes)],
            times=[f"{2000 + t}-01-01" for t in range(n_periods)],
            train_end=n_periods - holdout,
            target="PC0",
            eastings=eastings,
            northings=northings,
        )

    return build


@pytest.fixture
def ds(make_dataset: DatasetFactory) -> PriceSeriesDataset:
    return make_dataset()


@pytest.fixture
def star_fit() -> StarFit:
    return StarFit(
        rho=0.3,
        phi=np.array([0.4]),
        theta=np.array([0.2]),
        sigma2=0.01,
        loglik=0.0,
    )


def fake_transaction_id(*parts: object) -> str:
    """A stable UUID-shaped id, unique for ``parts``."""
    digest = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest().upper()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def synthetic_transactions(spec: PPDQuerySpec) -> pl.DataFrame:
    """Fake Land Registry response: 8 sales a year for one unit postcode inside each prefix."""
    return pl.concat([_synthetic_prefix(prefix, spec) for prefix in spec.prefixes])


def _synthetic_prefix(prefix: str, spec: PPDQuerySpec) -> pl.DataFrame:
    start, end = date.fromisoformat(spec.start_date), date.fromisoformat(spec.end_date)
    outward, _, rest = prefix.partition(" ")
    unit = f"{outward} {rest or '1'}AA"
    seed = sum(map(ord, prefix))
    rows = []
    for year in range(start.year, end.year + 1):
        for k in range(8):
            when = date(year, 1 + k % 12, 1 + k)
            if start <= when < end:
                price = int(200_000 * (1.06 ** (year - 2000)) * (1 + 0.01 * ((seed + k) % 5)))
                rows.append(
                    (
                        price,
                        when,
                        unit,
                        "TERRACED",
                        fake_transaction_id(prefix, year, k),
                        "ADD",
                    )
                )
    return pl.DataFrame(rows, schema=TRANSACTION_SCHEMA, orient="row")


def dataset_from_log_prices(
    log_prices: np.ndarray, train_end: int, **kwargs: Any
) -> PriceSeriesDataset:
    """A dataset with exactly these log prices (equal weights), for hand-checkable tests."""
    n, t = log_prices.shape
    return PriceSeriesDataset(
        prices=np.exp(log_prices),
        W=np.full((n, n), 1 / n),
        postcodes=[f"P{i}" for i in range(n)],
        times=[str(k) for k in range(t)],
        train_end=train_end,
        **kwargs,
    )
