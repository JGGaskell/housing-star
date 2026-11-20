"""Small calendar helpers shared by the data and evaluation layers."""

from __future__ import annotations

from datetime import date


def parse_date(value: str | date) -> date:
    """Parse an ISO ``YYYY-MM-DD`` string (dates pass through unchanged)."""
    return value if isinstance(value, date) else date.fromisoformat(value.strip())


def add_months(day: date, months: int) -> date:
    """Return the first day of the month ``months`` after the month of ``day``."""
    index = day.year * 12 + (day.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)
