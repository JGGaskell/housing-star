"""Input schemas: one for Land Registry sales, one for the median panel."""

from datetime import date

import polars as pl
import pytest
from pandera.errors import SchemaError

from housing_star.data.cache import TransactionCache, cache_filename
from housing_star.data.land_registry import normalise_transactions
from housing_star.data.schema import (
    PANEL_SCHEMA,
    TRANSACTION_SCHEMA,
    PanelSchema,
    TransactionSchema,
    validate_panel,
    validate_transactions,
)


def _sale(**overrides: object) -> pl.DataFrame:
    row: dict[str, object] = {
        "pricePaid": 250_000,
        "transactionDate": date(2020, 5, 17),
        "propertyAddress_postcode": "GL5 2AD",
        "propertyType": "TERRACED",
        "transactionId": "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "recordStatus": "ADD",
    }
    row.update(overrides)
    return pl.DataFrame([row], schema=TRANSACTION_SCHEMA)


def test_polars_dtypes_follow_the_model():
    assert dict(TransactionSchema.empty().schema) == TRANSACTION_SCHEMA


def test_a_canonical_sale_passes():
    out = validate_transactions(_sale())
    assert out["propertyType"].to_list() == ["TERRACED"]


def test_address_detail_beyond_the_postcode_is_not_kept():
    assert not any("paon" in c or "street" in c for c in TRANSACTION_SCHEMA)


def test_other_property_type_from_older_extracts_passes():
    assert validate_transactions(_sale(propertyType="OTHERPROPERTYTYPE")).height == 1


def test_empty_frame_is_valid():
    assert validate_transactions(pl.DataFrame(schema=TRANSACTION_SCHEMA)).is_empty()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("pricePaid", 0),
        ("pricePaid", -1),
        ("pricePaid", None),
        ("transactionDate", None),
        ("propertyAddress_postcode", None),
        ("propertyAddress_postcode", "GL5"),
        ("propertyAddress_postcode", "gl5 2ad"),
        ("propertyType", None),
        ("propertyType", "FLAT"),
        ("propertyType", "HTTP://LANDREGISTRY.DATA.GOV.UK/DEF/COMMON/TERRACED"),
        ("transactionId", None),
        ("transactionId", "not-a-uuid"),
        ("recordStatus", None),
        ("recordStatus", "DELETE"),
    ],
)
def test_schema_rejects_values_that_cannot_be_modelled(column, value):
    frame = _sale().with_columns(pl.lit(value, dtype=_sale().schema[column]).alias(column))
    with pytest.raises(SchemaError, match=column):
        validate_transactions(frame)


def test_schema_rejects_an_unexpected_column():
    with pytest.raises(SchemaError, match="extra"):
        validate_transactions(_sale().with_columns(extra=pl.lit(1)))


def test_schema_rejects_a_repeated_transaction_id():
    with pytest.raises(SchemaError, match="transactionId"):
        validate_transactions(pl.concat([_sale(), _sale()]))


def test_schema_rejects_a_missing_column():
    with pytest.raises(SchemaError, match="pricePaid"):
        validate_transactions(_sale().drop("pricePaid"))


def test_normalise_drops_a_price_that_is_not_a_positive_integer():
    raw = pl.DataFrame(
        {
            "pricePaid": ["0"],
            "transactionDate": ["2020-05-17"],
            "propertyAddress_postcode": ["gl5 2ad"],
            "propertyType": ["http://landregistry.data.gov.uk/def/common/terraced"],
            "propertyAddress_paon": ["1"],
            "propertyAddress_street": ["HIGH ST"],
            "transactionId": ["{aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa}"],
            "recordStatus": ["A"],
        }
    )
    out = normalise_transactions(raw)
    assert out.is_empty()
    assert validate_transactions(out).is_empty()


def _write(
    folder,
    name: str,
    property_type: str,
    transaction_id: str = "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "pricePaid": [180_000],
            "transactionDate": [date(2020, 3, 1)],
            "propertyAddress_postcode": ["GL5 2AD"],
            "propertyType": [property_type],
            "propertyAddress_paon": ["4"],
            "propertyAddress_street": [None],
            "transactionId": [transaction_id],
            "recordStatus": ["ADD"],
            "uniqueId": pl.Series([1], dtype=pl.UInt64),
        }
    ).write_parquet(folder / name)


def test_legacy_type_uris_are_reduced_before_validation(tmp_path):
    _write(
        tmp_path / "GL5",
        cache_filename(date(2020, 1, 1), date(2021, 1, 1)),
        "HTTP://LANDREGISTRY.DATA.GOV.UK/DEF/COMMON/SEMI-DETACHED",
    )
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    out = cache.transactions("GL5", "2020-01-01", "2021-01-01")
    assert out["propertyType"].to_list() == ["SEMI-DETACHED"]
    assert out.columns == list(TRANSACTION_SCHEMA)  # older files' address columns are dropped


def test_two_encodings_of_one_sale_count_once(tmp_path):
    folder = tmp_path / "GL5"
    _write(folder, cache_filename(date(2020, 1, 1), date(2020, 7, 1)), "TERRACED")
    _write(
        folder,
        cache_filename(date(2020, 6, 1), date(2021, 1, 1)),
        "HTTP://LANDREGISTRY.DATA.GOV.UK/DEF/COMMON/TERRACED",
    )
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    out = cache.transactions("GL5", "2020-01-01", "2021-01-01")
    assert out.height == 1
    assert out["propertyType"].to_list() == ["TERRACED"]


def test_a_malformed_sale_is_dropped_and_the_rest_cached(tmp_path):
    bad = _sale(pricePaid=0, transactionId="BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB")
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pl.concat([_sale(), bad]), verbose=False
    )
    out = cache.transactions("GL5", "2020-01-01", "2021-01-01")
    assert out["transactionId"].to_list() == ["AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"]
    assert len(list(tmp_path.rglob("*.parquet"))) == 1


def test_a_download_missing_a_column_is_not_cached(tmp_path):
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: _sale().drop("pricePaid"), verbose=False
    )
    with pytest.raises(ValueError, match="missing required columns"):
        cache.transactions("GL5", "2020-01-01", "2021-01-01")
    assert list(tmp_path.rglob("*.parquet")) == []


def _panel_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "postcode": "GL5",
        "bucket_start": date(2020, 1, 1),
        "price": 250_000.0,
        "n": 8,
    }
    row.update(overrides)
    return row


def _panel(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows) or [_panel_row()])


def test_panel_dtypes_follow_the_model():
    assert dict(PanelSchema.empty().schema) == {
        "postcode": pl.Utf8,
        "bucket_start": pl.Date,
        "price": pl.Float64,
        "n": pl.Int64,
    }


def test_a_median_row_passes_and_an_empty_panel_is_valid():
    out = validate_panel(
        _panel(_panel_row(), _panel_row(postcode="GL5 1", bucket_start=date(2021, 1, 1)))
    )
    assert out.height == 2
    assert validate_panel(pl.DataFrame(schema=PANEL_SCHEMA)).is_empty()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("postcode", "GL5 2AD"),
        ("postcode", "gl5"),
        ("postcode", None),
        ("bucket_start", None),
        ("price", 0.0),
        ("price", -1.0),
        ("price", None),
        ("n", 0),
        ("n", None),
    ],
)
def test_panel_schema_rejects_a_row_that_cannot_be_a_median(column, value):
    frame = _panel().with_columns(pl.lit(value, dtype=_panel().schema[column]).alias(column))
    with pytest.raises(SchemaError, match=column):
        validate_panel(frame)


def test_panel_schema_rejects_a_repeated_node_period_and_an_extra_column():
    with pytest.raises(SchemaError, match="postcode"):
        validate_panel(_panel(_panel_row(), _panel_row()))
    with pytest.raises(SchemaError, match="extra"):
        validate_panel(_panel().with_columns(extra=pl.lit(1)))


@pytest.mark.parametrize("column", ["price", "n", "postcode"])
def test_panel_schema_rejects_a_missing_column(column):
    with pytest.raises(SchemaError, match=column):
        validate_panel(_panel().drop(column))


def test_a_cached_file_that_breaks_the_schema_is_rejected(tmp_path):
    folder = tmp_path / "GL5"
    folder.mkdir()
    _sale(pricePaid=0).write_parquet(folder / cache_filename(date(2020, 1, 1), date(2021, 1, 1)))
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    with pytest.raises(SchemaError, match="pricePaid"):
        cache.transactions("GL5", "2020-01-01", "2021-01-01")
