"""Every table-returning function returns polars by default and pandas on request."""

import polars as pl
import pytest

import housing_star as hs
from housing_star import _output
from housing_star.data.series import download_plan
from housing_star.scenario.config import NodeSelector, Perturbation, PerturbationStep, Scenario
from housing_star.scenario.simulate import simulate


def is_pandas(obj) -> bool:
    return type(obj).__module__.startswith("pandas")


def test_export_keeps_polars_by_default():
    frame = pl.DataFrame({"a": [1]})
    assert _output.export(frame, False) is frame


def test_export_converts_with_polars_to_pandas():
    out = _output.export(pl.DataFrame({"a": [1, 2]}), True)
    assert is_pandas(out) and out["a"].tolist() == [1, 2]


def test_a_missing_pandas_installation_gives_an_actionable_error(monkeypatch):
    def broken(self):
        raise ModuleNotFoundError("No module named 'pandas'")

    monkeypatch.setattr(pl.DataFrame, "to_pandas", broken)
    with pytest.raises(ImportError, match=r"housing-star\[pandas\]"):
        _output.export(pl.DataFrame({"a": [1]}), True)


@pytest.fixture
def models():
    return {"STAR": hs.StarModel(), "AR(1)": hs.ArModel()}


CV = hs.WalkForwardCV(n_folds=2, min_train=8)


def test_dataset_table_is_tidy(ds):
    frame = ds.to_frame()
    assert isinstance(frame, pl.DataFrame) and frame.height == ds.N * ds.T
    assert frame.columns == ["postcode", "time", "price", "y", "is_holdout"]
    assert frame["time"].dtype == pl.Date
    assert frame["is_holdout"].sum() == ds.N * ds.n_eval
    assert is_pandas(ds.to_frame(as_pandas=True))


def test_predict_frame_puts_predictions_next_to_actuals(ds):
    frame = hs.ArModel().fit(ds).predict_frame(ds)
    assert frame.columns == ["postcode", "time", "actual", "predicted", "is_holdout"]
    assert frame["predicted"].null_count() == 2 * ds.N  # needs the previous price and one lag
    assert is_pandas(hs.ArModel().fit(ds).predict_frame(ds, as_pandas=True))


def test_forecast_frame_is_in_pounds_with_one_row_per_node_and_step(ds):
    model = hs.ArModel().fit(ds)
    frame = model.forecast_frame(ds.train_only(), 3)
    assert frame.columns == ["postcode", "step", "forecast"] and frame.height == ds.N * 3
    assert frame["forecast"].min() > 1_000  # pounds, not log values
    assert is_pandas(model.forecast_frame(ds.train_only(), 3, as_pandas=True))


def test_evaluation_tables(ds, models):
    assert isinstance(hs.evaluate(ds, models), pl.DataFrame)
    assert is_pandas(hs.evaluate(ds, models, as_pandas=True))
    assert is_pandas(hs.target_predictions(ds, models, as_pandas=True))
    assert is_pandas(hs.cross_validate(models, ds, CV, as_pandas=True))


def test_tuning_tables(ds):
    polars_result = hs.tune(hs.ArModel, ds, {"p": [1, 2]}, CV)
    pandas_result = hs.tune(hs.ArModel, ds, {"p": [1, 2]}, CV, as_pandas=True)
    assert isinstance(polars_result.summary, pl.DataFrame) and isinstance(
        polars_result.scores, pl.DataFrame
    )
    assert is_pandas(pandas_result.summary) and is_pandas(pandas_result.scores)
    assert pandas_result.best_params == polars_result.best_params


def test_scenario_table(ds, star_fit):
    pert = Perturbation("p", NodeSelector(nth_closest=1), (PerturbationStep(1, 5),))
    result = simulate(
        hs.StarModel.from_fit(star_fit, ds.growth_means), ds, Scenario("PC0", 2, (pert,))
    )
    assert isinstance(result.to_frame(), pl.DataFrame)
    assert is_pandas(result.to_frame(as_pandas=True))


def test_backtest_table(make_dataset):
    kwargs = dict(
        start="2000-01-01",
        end="2010-01-01",
        window_years=10,
        cv=CV,
        build_dataset=lambda pc, s, e: make_dataset(n_periods=14, holdout=2),
    )
    assert isinstance(hs.backtest(["PC0"], **kwargs), pl.DataFrame)
    assert is_pandas(hs.backtest(["PC0"], as_pandas=True, **kwargs))


def test_download_plan_table(tmp_path):
    cache = hs.TransactionCache(root=tmp_path, verbose=False)
    plan = download_plan("GL5", start="2000-01-01", end="2001-01-01", cache=cache, as_pandas=True)
    assert is_pandas(plan)
