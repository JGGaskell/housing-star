"""The neighbourhood map: the one drawing the library does."""

import builtins

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_rgba


def _axes():
    return plt.subplots()


def test_focal_node_is_red_and_closer_pairs_are_darker(ds):
    _fig, ax = _axes()
    assert ds.plot_neighbourhood(ax) is ax

    n = ds.N
    assert len(ax.lines) == n * (n - 1) // 2
    # Pairs are drawn with i < j, so line 0 joins nodes 0 and 1 and line 4 joins 0 and 5.
    assert ax.lines[0].get_alpha() > ax.lines[4].get_alpha()

    colors = ax.collections[0].get_facecolors()
    assert len(colors) == n
    assert colors[0] == pytest.approx(to_rgba("tab:red"))
    assert all(point == pytest.approx(to_rgba("tab:blue")) for point in colors[1:])
    assert [text.get_text() for text in ax.texts] == ds.postcodes
    assert ax.get_title() == "Spatial neighbourhood"
    assert ax.get_xlabel() == "Easting (km)"
    assert ax.get_aspect() == 1.0
    plt.close("all")


def test_a_dataset_with_no_target_is_drawn_all_blue(ds):
    import dataclasses

    plain = dataclasses.replace(ds, target=None)
    _fig, ax = _axes()
    plain.plot_neighbourhood(ax)
    colors = ax.collections[0].get_facecolors()
    assert all(point == pytest.approx(to_rgba("tab:blue")) for point in colors)
    plt.close("all")


def test_missing_coordinates_raise(ds):
    import dataclasses

    unlocated = dataclasses.replace(ds, eastings=None, northings=None)
    _fig, ax = _axes()
    with pytest.raises(ValueError, match="centroid coordinates"):
        unlocated.plot_neighbourhood(ax)
    plt.close("all")


def test_missing_matplotlib_explains_the_extra(monkeypatch, ds):
    _fig, ax = _axes()
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(ImportError, match=r"housing-star\[plot\]"):
        ds.plot_neighbourhood(ax)
    monkeypatch.undo()  # matplotlib >= 3.11 imports while closing figures
    plt.close("all")
