"""Pandera contracts for the two ways a price reaches a model.

:class:`TransactionSchema` is one Land Registry sale. The SPARQL client parses a response into
this shape and checks it before anything is cached. :class:`housing_star.data.cache.TransactionCache`
checks again on read, after reducing a legacy property-type URI to its label.

:class:`PanelSchema` is the long price frame a model is built from: one median per node and
period. :meth:`housing_star.data.dataset.PriceSeriesDataset.from_panel` checks a frame the caller
supplied. :func:`housing_star.data.series.panel_series` checks the frame it aggregated from
validated sales, and ``from_postcode`` passes that frame through ``from_panel``, so both inputs
are checked again there.

Only what the models use is kept: no house number, street or other address detail beyond the
unit postcode. A row of :class:`TransactionSchema` is one sale: ``transactionId`` is the Price
Paid identifier, so two flats in one building stay two sales. ``recordStatus`` is ``ADD`` or
``CHANGE`` after :func:`housing_star.data.land_registry.resolve_transactions` has applied
deletions. The checks that protect a model are the price, the date and the postcode.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandera.polars as pa
import polars as pl

# ``otherPropertyType`` is no longer requested, but extracts cached before that filter still
# contain it and are read back into the same models.
PROPERTY_TYPES: tuple[str, ...] = (
    "DETACHED",
    "SEMI-DETACHED",
    "TERRACED",
    "FLAT-MAISONETTE",
    "OTHERPROPERTYTYPE",
)

# A full unit postcode, already upper-cased with the outward and inward codes separated by one
# space (``"GL5 2AD"``). Districts and sectors are node keys, not values in this column.
UNIT_POSTCODE_PATTERN = r"^[A-Z]{1,2}\d[A-Z\d]? \d[A-Z]{2}$"

# Price Paid transaction identifier: a UUID, upper-cased, without braces.
TRANSACTION_ID_PATTERN = r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$"

# A node key: a district (``GL5``, ``GL51``) or a sector (``GL5 1``). Merged labels are produced
# by cleaning, after this check, and are not valid input.
NODE_KEY_PATTERN = r"^[A-Z]{1,2}\d[A-Z\d]?(?: \d)?$"

# Statuses that survive resolution. ``DELETE`` removes the sale and is not stored.
RECORD_STATUSES: tuple[str, ...] = ("ADD", "CHANGE")


class TransactionSchema(pa.DataFrameModel):
    """One row per Price Paid sale in the columns the cache stores."""

    pricePaid: int = pa.Field(gt=0, nullable=False)
    transactionDate: date = pa.Field(nullable=False)
    propertyAddress_postcode: str = pa.Field(str_matches=UNIT_POSTCODE_PATTERN, nullable=False)
    propertyType: str = pa.Field(isin=PROPERTY_TYPES, nullable=False)
    transactionId: str = pa.Field(str_matches=TRANSACTION_ID_PATTERN, nullable=False, unique=True)
    recordStatus: str = pa.Field(isin=list(RECORD_STATUSES), nullable=False)

    class Config:
        """Reject a column the schema does not name."""

        strict = True
        name = "TransactionSchema"


TRANSACTION_SCHEMA: dict[str, Any] = {
    "pricePaid": pl.Int64,
    "transactionDate": pl.Date,
    "propertyAddress_postcode": pl.Utf8,
    "propertyType": pl.Utf8,
    "transactionId": pl.Utf8,
    "recordStatus": pl.Utf8,
}
"""Polars dtypes for :class:`TransactionSchema`, for frames built before validation."""


def validate_transactions(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` when it satisfies :class:`TransactionSchema`.

    An empty frame with these columns is valid: it is a fetched interval that contained no sales.

    Raises
    ------
    pandera.errors.SchemaError
        If a column is missing or unexpected, a dtype differs, or a value fails a check.
    """
    return TransactionSchema.validate(frame)


class PanelSchema(pa.DataFrameModel):
    """One price per node and period (a median or an adjusted index), the frame ``from_panel`` pivots."""

    postcode: str = pa.Field(str_matches=NODE_KEY_PATTERN, nullable=False)
    bucket_start: date = pa.Field(nullable=False)
    price: float = pa.Field(gt=0, lt=float("inf"), nullable=False)
    n: int = pa.Field(ge=1, nullable=False)

    class Config:
        """Reject an extra column, and one node-period pair must not appear twice."""

        strict = True
        unique = ["postcode", "bucket_start"]
        name = "PanelSchema"


PANEL_SCHEMA: dict[str, Any] = {
    "postcode": pl.Utf8,
    "bucket_start": pl.Date,
    "price": pl.Float64,
    "n": pl.UInt32,
}
"""Polars dtypes for :class:`PanelSchema`. ``n`` may arrive as any integer width."""


def validate_panel(frame: pl.DataFrame) -> pl.DataFrame:
    """Return ``frame`` when it satisfies :class:`PanelSchema`.

    An empty frame with these columns is valid. ``from_panel`` then reports that no node had a
    price, which is the same message as an empty Land Registry extract. A sale count may arrive
    as any integer width (``pl.len()`` is ``UInt32``); it is cast to ``Int64`` before the check.
    Spacing of ``bucket_start`` is checked by :func:`housing_star.data.prep.panel_from_long`.
    A period missing for every node is an absent row, and an uneven calendar is rejected there.

    Raises
    ------
    pandera.errors.SchemaError
        If a column is missing or unexpected, a node key is not a district or sector, a price is
        not positive, a sale count is below 1, or a node and period are repeated.
    """
    if "n" in frame.columns:
        frame = frame.with_columns(pl.col("n").cast(pl.Int64))
    if "price" in frame.columns:
        frame = frame.with_columns(pl.col("price").cast(pl.Float64, strict=False))
    return PanelSchema.validate(frame)
