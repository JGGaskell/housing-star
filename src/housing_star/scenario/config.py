"""Scenario definition: which nodes to shock, by how much, over which future periods.

YAML example::

    target: GL5
    horizon: 5
    perturbations:
      - name: nearby_boom
        select: { nth_closest: 1 }     # or { postcode: GL6 }
        mode: clamp                     # clamp | impulse (default: clamp)
        steps:
          - { step: 1, change_pct: 5 }
          - { step: 2, change_pct: 15 }

Shock modes
-----------
``clamp``
    The node is *forced* onto ``baseline * cumulative_multiplier`` from each listed step on
    (unlisted steps keep the last multiplier). Steps compound: +5% then +15% means
    ``1.05 * 1.15 = 1.2075`` times the baseline price.
``impulse``
    At each listed step the node's price is moved by ``change_pct`` relative to the value the
    dynamics would give it that period. Afterwards the node grows on from its new price, so the
    shift **persists** (the model's growth dynamics may add to it or partly reverse it) rather
    than reverting to the old price path. Several impulses compound. Unlisted steps get no shock.

In both modes neighbours react in the *same* period the shock occurs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from housing_star.data.dataset import PriceSeriesDataset

Mode = Literal["clamp", "impulse"]


@dataclass(frozen=True)
class NodeSelector:
    """Pick a node by exact ``postcode`` or by proximity rank ``nth_closest`` (1 = nearest).

    Exactly one must be given. Proximity is measured from the scenario's target node.
    """

    postcode: str | None = None
    nth_closest: int | None = None

    def __post_init__(self) -> None:
        if (self.postcode is None) == (self.nth_closest is None):
            raise ValueError("NodeSelector needs exactly one of 'postcode' or 'nth_closest'.")
        if self.nth_closest is not None and self.nth_closest < 1:
            raise ValueError(f"nth_closest must be >= 1; got {self.nth_closest}.")

    def resolve(self, ds: PriceSeriesDataset, target_idx: int) -> int:
        """Row index of the selected node in ``ds``."""
        if self.postcode is not None:
            idx = ds.index_of(self.postcode)
            if idx is None:
                raise ValueError(
                    f"Postcode {self.postcode!r} not found among nodes {ds.postcodes}."
                )
            return idx
        assert self.nth_closest is not None
        order = ds.neighbour_order(target_idx)
        if self.nth_closest > len(order):
            raise ValueError(
                f"nth_closest={self.nth_closest} but the dataset has only {len(order)} neighbours."
            )
        return int(order[self.nth_closest - 1])


@dataclass(frozen=True)
class PerturbationStep:
    """A price change of ``change_pct`` percent at future ``step`` (1 = first forecast period)."""

    step: int
    change_pct: float

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError(f"step must be >= 1; got {self.step}.")
        if self.change_pct <= -100.0:
            raise ValueError(f"change_pct must be > -100; got {self.change_pct}.")

    @property
    def log_offset(self) -> float:
        """The change as an additive offset in log space."""
        return math.log1p(self.change_pct / 100.0)


@dataclass(frozen=True)
class Perturbation:
    """A named shock to one node over one or more steps."""

    name: str
    select: NodeSelector
    steps: tuple[PerturbationStep, ...]
    mode: Mode = "clamp"

    def __post_init__(self) -> None:
        if self.mode not in ("clamp", "impulse"):
            raise ValueError(f"mode must be 'clamp' or 'impulse'; got {self.mode!r}.")
        if not self.steps:
            raise ValueError(f"Perturbation {self.name!r} has no steps.")
        numbers = [s.step for s in self.steps]
        if len(numbers) != len(set(numbers)):
            raise ValueError(f"Perturbation {self.name!r} repeats a step: {numbers}.")


@dataclass(frozen=True)
class Scenario:
    """Target postcode, forecast horizon and the perturbations to apply."""

    target: str
    horizon: int
    perturbations: tuple[Perturbation, ...]

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1; got {self.horizon}.")
        for pert in self.perturbations:
            last = max(s.step for s in pert.steps)
            if last > self.horizon:
                raise ValueError(
                    f"Perturbation {pert.name!r} has step {last} > horizon {self.horizon}."
                )

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form; ``Scenario.from_dict(s.to_dict()) == s``."""
        return {
            "target": self.target,
            "horizon": self.horizon,
            "perturbations": [
                {
                    "name": p.name,
                    "select": (
                        {"postcode": p.select.postcode}
                        if p.select.postcode is not None
                        else {"nth_closest": p.select.nth_closest}
                    ),
                    "mode": p.mode,
                    "steps": [{"step": s.step, "change_pct": s.change_pct} for s in p.steps],
                }
                for p in self.perturbations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        """Build a scenario from parsed YAML/JSON."""
        perturbations = tuple(
            Perturbation(
                name=str(p["name"]),
                select=NodeSelector(**p["select"]),
                steps=tuple(
                    PerturbationStep(int(s["step"]), float(s["change_pct"])) for s in p["steps"]
                ),
                mode=p.get("mode", "clamp"),
            )
            for p in data.get("perturbations", [])
        )
        return cls(str(data["target"]), int(data["horizon"]), perturbations)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        """Load a scenario from a YAML file."""
        with Path(path).open() as file:
            return cls.from_dict(yaml.safe_load(file))

    def to_yaml(self, path: str | Path) -> None:
        """Write the scenario to a YAML file."""
        with Path(path).open("w") as file:
            yaml.safe_dump(self.to_dict(), file, sort_keys=False)
