from datetime import date

import pytest

from housing_star._dates import add_months, parse_date


def test_parse_date_accepts_strings_and_dates():
    assert parse_date(" 2020-03-01 ") == date(2020, 3, 1)
    assert parse_date(date(2020, 3, 1)) == date(2020, 3, 1)


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (date(2020, 1, 15), 1, date(2020, 2, 1)),
        (date(2020, 11, 30), 3, date(2021, 2, 1)),
        (date(2020, 1, 1), 24, date(2022, 1, 1)),
        (date(2020, 1, 1), -1, date(2019, 12, 1)),
    ],
)
def test_add_months_returns_first_of_month(start, months, expected):
    assert add_months(start, months) == expected
