"""Run a scenario: baseline forecast versus a shocked forecast."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal, overload

import numpy as np
import polars as pl

from housing_star._output import export
from housing_star._types import FloatArray
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models.star.model import StarModel, warn_if_unstable
from housing_star.scenario.config import Scenario


@dataclass(frozen=True)
class ScenarioResult:
    """Baseline and perturbed forecasts, both ``(N, horizon)`` in log-demeaned levels."""

    scenario: Scenario
    postcodes: list[str]
    target_idx: int
    log_node_means: FloatArray
    baseline: FloatArray
    perturbed: FloatArray

    @property
    def horizon(self) -> int:
        """Number of forecast periods."""
        return int(self.baseline.shape[1])

    def baseline_abs(self) -> FloatArray:
        """Baseline forecast in pounds."""
        return np.exp(self.baseline + self.log_node_means[:, None])

    def perturbed_abs(self) -> FloatArray:
        """Perturbed forecast in pounds."""
        return np.exp(self.perturbed + self.log_node_means[:, None])

    def effect_pct(self) -> FloatArray:
        """``(N, horizon)`` percentage price change versus baseline."""
        return (self.perturbed_abs() / self.baseline_abs() - 1.0) * 100.0

    def target_effect(self) -> FloatArray:
        """``(horizon,)`` percentage price change of the target node versus baseline."""
        return np.asarray(self.effect_pct()[self.target_idx], dtype=np.float64)

    @overload
    def to_frame(self, as_pandas: Literal[False] = False) -> pl.DataFrame: ...
    @overload
    def to_frame(self, as_pandas: bool) -> Any: ...
    def to_frame(self, as_pandas: bool = False) -> Any:
        """Tidy frame with one row per ``(node, step)`` (polars, or pandas if ``as_pandas``)."""
        N, H = self.baseline.shape
        frame = pl.DataFrame(
            {
                "postcode": np.repeat(self.postcodes, H),
                "is_target": np.repeat(np.arange(N) == self.target_idx, H),
                "step": np.tile(np.arange(1, H + 1), N),
                "baseline_abs": self.baseline_abs().ravel(),
                "perturbed_abs": self.perturbed_abs().ravel(),
                "pct_effect": self.effect_pct().ravel(),
            }
        )
        return export(frame, as_pandas)


def simulate(model: StarModel, ds: PriceSeriesDataset, scenario: Scenario) -> ScenarioResult:
    """Forecast ``scenario.horizon`` periods ahead with and without the scenario's shocks.

    Parameters
    ----------
    model : a fitted :class:`~housing_star.models.star.model.StarModel`.
    ds : the dataset the forecast starts from (its last column is the last observation). Use a
        dataset without hold-out, or ``ds.train_only()``, to forecast from the end of training.

    Raises
    ------
    TypeError
        If ``model`` is not a ``StarModel``: spillover needs a spatial model.

    Warns
    -----
    UnstableModelWarning
        If the model's growth shocks do not fade (persistence of 1 or more).
    ValueError
        If the target or a selected node is not in ``ds``.
    """
    if not isinstance(model, StarModel):
        raise TypeError(
            f"simulate() needs a StarModel (spillover requires a spatial model); got {type(model).__name__}."
        )
    warn_if_unstable(model, ds.W)
    target_idx = ds.index_of(scenario.target)
    if target_idx is None:
        raise ValueError(f"Target {scenario.target!r} not found among nodes {ds.postcodes}.")

    baseline = model.forecast(ds, scenario.horizon)
    cumulative: defaultdict[int, FloatArray] = defaultdict(lambda: np.zeros(scenario.horizon))
    impulses: defaultdict[int, FloatArray] = defaultdict(lambda: np.zeros(scenario.horizon))
    for pert in scenario.perturbations:
        node = pert.select.resolve(ds, target_idx)
        shock = np.zeros(scenario.horizon)
        for step in pert.steps:
            shock[step.step - 1] = step.log_offset
        if pert.mode == "clamp":
            cumulative[node] += np.cumsum(shock)
        else:
            impulses[node] += shock

    perturbed = model.forecast(
        ds,
        scenario.horizon,
        clamp={node: baseline[node] + offset for node, offset in cumulative.items()},
        impulse=dict(impulses),
    )
    return ScenarioResult(
        scenario, list(ds.postcodes), target_idx, ds.log_node_means, baseline, perturbed
    )
