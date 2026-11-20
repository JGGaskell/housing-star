"""Result export: polars by default, pandas on request."""

from __future__ import annotations

from typing import Any

import polars as pl


def export(frame: pl.DataFrame, as_pandas: bool) -> Any:
    """Return ``frame`` unchanged, or as a pandas ``DataFrame`` if ``as_pandas`` is set.

    pandas is never imported by this package: the conversion is ``polars.DataFrame.to_pandas``,
    which needs ``pandas`` and ``pyarrow`` installed (``pip install housing-star[pandas]``).

    Raises
    ------
    ImportError
        If ``as_pandas`` is true and pandas or pyarrow is missing.
    """
    if not as_pandas:
        return frame
    try:
        return frame.to_pandas()
    except ModuleNotFoundError as exc:
        raise ImportError(
            "as_pandas=True needs pandas and pyarrow: pip install 'housing-star[pandas]'."
        ) from exc
