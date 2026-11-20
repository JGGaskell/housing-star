"""housing_star: spatial-temporal autoregression (STAR) for UK house prices.

Quick start::

    import housing_star as hs

    ds = hs.PriceSeriesDataset.from_postcode("GL5", start="2005-01-01", end="2023-01-01", holdout=3)
    tuned = hs.tune(hs.StarModel, ds, {"p": [1, 2, 3]})
    models = {"STAR": hs.StarModel(**tuned.best_params), "AR(1)": hs.ArModel(p=1)}
    print(hs.evaluate(ds, models))
"""

import logging
from importlib.metadata import PackageNotFoundError, version

from housing_star.config import AdjustmentConfig, CleaningConfig, GraphConfig
from housing_star.data.cache import TransactionCache
from housing_star.data.dataset import PriceSeriesDataset
from housing_star.data.series import download_plan
from housing_star.evaluation.backtest import backtest
from housing_star.evaluation.cv import WalkForwardCV
from housing_star.evaluation.evaluate import cross_validate, evaluate, target_predictions
from housing_star.evaluation.tuning import TuningResult, tune
from housing_star.models.ar import ArModel
from housing_star.models.drift import DriftModel, NoChangeModel
from housing_star.models.star.model import StarModel, UnstableModelWarning
from housing_star.scenario.config import NodeSelector, Perturbation, PerturbationStep, Scenario
from housing_star.scenario.simulate import ScenarioResult, simulate

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    __version__ = version("housing-star")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled checkout
    __version__ = "0.0.0"

__all__ = [
    "AdjustmentConfig",
    "ArModel",
    "CleaningConfig",
    "DriftModel",
    "GraphConfig",
    "NoChangeModel",
    "NodeSelector",
    "Perturbation",
    "PerturbationStep",
    "PriceSeriesDataset",
    "Scenario",
    "ScenarioResult",
    "StarModel",
    "TransactionCache",
    "TuningResult",
    "UnstableModelWarning",
    "WalkForwardCV",
    "backtest",
    "cross_validate",
    "download_plan",
    "evaluate",
    "simulate",
    "target_predictions",
    "tune",
]
