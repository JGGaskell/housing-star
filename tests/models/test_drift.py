import numpy as np
import pytest

from housing_star.data.dataset import PriceSeriesDataset
from housing_star.models.drift import DriftModel, NoChangeModel


def kinked_dataset():
    # log prices rise 0.1 a period, dip 0.3 in the last training period, then rise again
    z = np.array([[0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    return PriceSeriesDataset(
        prices=np.exp(z),
        W=np.full((2, 2), 0.5),
        postcodes=["A", "B"],
        times=[str(i) for i in range(7)],
        train_end=5,
    )


def test_growth_is_the_mean_training_change():
    np.testing.assert_allclose(DriftModel().fit(kinked_dataset()).growth_, [0.0, 0.0], atol=1e-12)


def test_forecasts_start_from_the_latest_value_not_a_fitted_line():
    ds = kinked_dataset().window(4, 7)  # train before the dip: growth 0.1 for A, 0 for B
    model = DriftModel().fit(ds)
    np.testing.assert_allclose(model.growth_, [0.1, 0.0])
    one_step = model.predict(ds)
    assert np.isnan(one_step[:, 0]).all()
    np.testing.assert_allclose(one_step[:, 1:], ds.y[:, :-1] + [[0.1], [0.0]])
    to_dip = ds.window(5, 5)  # forecast from the dip, not from the pre-dip line
    recursive = model.forecast(to_dip, 2)
    np.testing.assert_allclose(recursive, to_dip.y[:, [4]] + [[0.1, 0.2], [0.0, 0.0]])


def test_params_expose_the_growth_rates():
    assert len(DriftModel().fit(kinked_dataset()).params_["growth"]) == 2


def test_fit_needs_two_training_periods(ds):
    with pytest.raises(ValueError, match="at least 2"):
        DriftModel().fit(ds.window(1, 1))


def test_forecast_horizon_must_be_positive():
    model = DriftModel().fit(kinked_dataset())
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(kinked_dataset(), 0)


def test_dataset_must_have_the_fitted_nodes(make_dataset):
    model = DriftModel().fit(make_dataset(n_nodes=4))
    with pytest.raises(ValueError, match="fitted on 4 nodes"):
        model.predict(make_dataset(n_nodes=5))


def test_unfitted_params_are_unavailable():
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = DriftModel().params_


def test_no_change_carries_the_latest_value_forward():
    ds = kinked_dataset().window(4, 7)
    model = NoChangeModel().fit(ds)
    one_step = model.predict(ds)
    assert np.isnan(one_step[:, 0]).all()
    np.testing.assert_allclose(one_step[:, 1:], ds.y[:, :-1])
    to_dip = ds.window(5, 5)
    np.testing.assert_allclose(model.forecast(to_dip, 3), np.repeat(to_dip.y[:, [4]], 3, axis=1))


def test_no_change_fits_on_a_single_training_period(ds):
    assert NoChangeModel().fit(ds.window(1, 1)).params_ == {}


def test_no_change_checks_horizon_nodes_and_fitting(make_dataset):
    model = NoChangeModel().fit(make_dataset(n_nodes=4))
    with pytest.raises(ValueError, match="horizon"):
        model.forecast(make_dataset(n_nodes=4), 0)
    with pytest.raises(ValueError, match="fitted on 4 nodes"):
        model.predict(make_dataset(n_nodes=5))
    with pytest.raises(RuntimeError, match="not fitted"):
        NoChangeModel().predict(make_dataset(n_nodes=4))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = NoChangeModel().params_
