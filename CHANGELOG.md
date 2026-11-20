# Changelog

Notable changes to `housing-star`. Versions follow [semantic versioning](https://semver.org/); before
1.0, a minor version may include breaking changes.

## 0.1.0 - 2026-11-20

First public release. Supports Python 3.11 to 3.15.

### Added

- **Data.** `PriceSeriesDataset.from_postcode` builds a postcode district or sector and its
  neighbourhood from HM Land Registry Price Paid Data. Sales are downloaded on demand from the Land
  Registry SPARQL endpoint, validated with pandera schemas, and cached locally
  (`TransactionCache`, `download_plan`). Downloads are batched, run in parallel and retried.
  `PriceSeriesDataset.from_panel` builds a dataset from a price panel you already have.
- **Prices.** Median sale prices, or a composition-adjusted index (`prices="adjusted"`,
  `AdjustmentConfig`) that holds the mix of unit postcodes and property types fixed, fitted on the
  training periods only.
- **Neighbourhoods and weights.** Nodes are chosen from bundled district and sector centroids (OS
  Code-Point Open) within a radius (`GraphConfig`), thin nodes are dropped or merged using the
  training window only (`CleaningConfig`), and neighbours are weighted by inverse distance.
  `plot_neighbourhood` draws the network (optional `plot` extra).
- **Models.** `StarModel`, a spatial-temporal autoregression on growth, and the benchmarks
  `ArModel`, `DriftModel` and `NoChangeModel`, all with `fit`, `predict` and `forecast`.
  `UnstableModelWarning` flags a fitted model whose shocks do not fade.
- **Evaluation.** `evaluate` and `target_predictions` with one-step and recursive protocols,
  walk-forward `cross_validate` and `tune` (`WalkForwardCV`), and a rolling-window `backtest`.
- **Scenarios.** `Scenario`, `Perturbation`, `NodeSelector` and `simulate`, with `clamp` and
  `impulse` shocks and YAML round-tripping.
- **Tables.** Every result is a polars DataFrame, or a pandas one with `as_pandas=True` (optional
  `pandas` extra).
- **Notebooks.** Forecasting (01), scenarios (02), and an out-of-sample evaluation on 27 randomly
  drawn districts at district and sector scale (03).
