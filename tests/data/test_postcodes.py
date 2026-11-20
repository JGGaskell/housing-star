import polars as pl
import pytest

from housing_star.data.postcodes import clean_code, infer_level, load_centroids, node_key_expr


def test_clean_code_uppercases_and_collapses_whitespace():
    assert clean_code("  gl5   2 ") == "GL5 2"


@pytest.mark.parametrize(
    ("code", "level"),
    [("GL5", "district"), ("e1", "district"), ("GL6 8", "sector"), ("cf15  7", "sector")],
)
def test_infer_level(code, level):
    assert infer_level(code) == level


@pytest.mark.parametrize("code", ["SW", "GL6 8AU", "GL6 88", "", "1AB", "A B C"])
def test_infer_level_rejects_unsupported_codes(code):
    with pytest.raises(ValueError, match="Unsupported postcode"):
        infer_level(code)


@pytest.mark.parametrize(
    ("level", "expected"),
    [("district", ["GL5", "E14", "E1"]), ("sector", ["GL5 2", "E14 9", "E1 6"])],
)
def test_node_key_expr_maps_unit_postcodes(level, expected):
    frame = pl.DataFrame({"pc": ["gl5  2AD", "E14 9XY", "E1 6AN"]})
    assert frame.select(node_key_expr("pc", level))["pc"].to_list() == expected


def test_bundled_centroids_cover_both_levels():
    table = load_centroids()
    assert set(table["level"]) == {"district", "sector"}
    assert {"GL5", "E1", "E14"} <= set(table.filter(pl.col("level") == "district")["key"])
    assert load_centroids() is table  # cached


def test_bundled_centroids_exclude_unplaced_postcodes():
    """Code-Point Open puts postcodes it cannot place at (0, 0); none may reach the table."""
    table = load_centroids()
    # The nearest real centroid, in the Isles of Scilly, is about 90 km from the origin.
    assert (table["easting"].pow(2) + table["northing"].pow(2)).sqrt().min() > 50_000
    # NW1W is a non-geographic London district; with its unplaced postcodes averaged in it sat in
    # Berkshire, inside RG45's neighbourhood.
    nw1w = table.filter(pl.col("level") == "district", pl.col("key") == "NW1W")
    assert 525_000 < nw1w["easting"].item() < 535_000
    assert 180_000 < nw1w["northing"].item() < 186_000


def test_build_centroids_drops_origin_rows(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "scripts" / "build_centroids.py"
    spec = importlib.util.spec_from_file_location("build_centroids", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "codepo.parquet"
    pl.DataFrame(
        {
            "Postcode": ["GL5 1AA", "GL5 1AB", "GL5 9ZZ", "PO24 1AA"],
            "Eastings": [385_000, 385_200, 0, 0],
            "Northings": [204_000, 204_200, 0, 0],
        }
    ).write_parquet(source)
    table = module.build(source)
    assert table.filter(pl.col("key") == "GL5").row(0, named=True) == {
        "level": "district",
        "key": "GL5",
        "easting": 385_100.0,
        "northing": 204_100.0,
        "n_units": 2,
    }
    assert set(table["key"]) == {"GL5", "GL5 1"}  # GL5 9 and PO24 had no placed postcode
