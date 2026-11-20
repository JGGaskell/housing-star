import re
from typing import Any

import polars as pl
import pytest
import requests

from housing_star.data import land_registry as lr
from housing_star.data.land_registry import (
    PPDQuerySpec,
    build_ppd_query,
    fetch_transactions,
    normalise_transactions,
    prefix_filter,
    run_sparql,
    sparql_json_to_frame,
    split_prefix,
    text_query,
)
from housing_star.data.schema import TRANSACTION_SCHEMA


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, headers: dict | None = None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)  # type: ignore[arg-type]

    def json(self) -> Any:
        return self._payload


class FakeSession:
    """Replays scripted outcomes: a response to return or an exception to raise."""

    def __init__(self, *outcomes: Any):
        self.outcomes, self.calls = list(outcomes), 0

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(lr.time, "sleep", slept.append)
    lr._GATE.reset()
    yield slept
    lr._GATE.reset()


def test_query_contains_prefix_dates_and_new_build_filter():
    query = build_ppd_query(PPDQuerySpec(" gl5 ", "2020-01-01", "2021-01-01"))
    assert '"postcode:GL5"' in query
    assert '"2020-01-01"^^xsd:date' in query
    assert "?newBuild = false" in query


def test_query_can_include_new_builds():
    spec = PPDQuerySpec("GL5", "2020-01-01", "2021-01-01", non_new_build_only=False)
    assert "?newBuild = false" not in build_ppd_query(spec)


def test_query_asks_for_one_row_more_than_the_limit_and_does_not_sort():
    # Sorting with LIMIT/OFFSET made the endpoint re-run the whole search for every page.
    query = build_ppd_query(PPDQuerySpec("GL5", "2020-01-01", "2021-01-01", limit=5_000))
    assert query.endswith("LIMIT 5001")
    assert "ORDER BY" not in query and "OFFSET" not in query


def test_query_returns_only_the_cached_fields():
    query = build_ppd_query(PPDQuerySpec("GL5", "2020-01-01", "2021-01-01"))
    selected = query.split("SELECT DISTINCT", 1)[1].split("WHERE", 1)[0].split()
    assert sorted(name.lstrip("?") for name in selected) == sorted(TRANSACTION_SCHEMA)


def test_several_prefixes_share_one_text_query():
    assert text_query(("GL5", "GL6 8", "GL6 8A")) == (
        "postcode:GL5 OR (+postcode:GL6 +postcode:8*) OR (+postcode:GL6 +postcode:8a*)"
    )
    query = build_ppd_query(PPDQuerySpec(("GL5", "GL10"), "2020-01-01", "2021-01-01"))
    assert '"postcode:GL5 OR postcode:GL10"' in query


def test_spec_cleans_and_deduplicates_prefixes():
    spec = PPDQuerySpec(("gl5", " GL5 ", "gl6  8"), "2020-01-01", "2021-01-01")
    assert spec.prefixes == ("GL5", "GL6 8")


@pytest.mark.parametrize("bad", ["  ", "GL5 2AD Z", "5GL", 'GL5" } DROP', "GL5 A"])
def test_anything_but_a_postcode_prefix_is_rejected(bad):
    with pytest.raises(ValueError, match="Unsupported postcode prefix"):
        PPDQuerySpec(bad, "2020-01-01", "2021-01-01")


def test_spec_needs_a_prefix_and_a_positive_limit():
    with pytest.raises(ValueError, match="prefixes is empty"):
        PPDQuerySpec((), "2020-01-01", "2021-01-01")
    with pytest.raises(ValueError, match="limit"):
        PPDQuerySpec("GL5", "2020-01-01", "2021-01-01", limit=0)


def test_split_prefix():
    assert split_prefix("gl6  8") == ("GL6", "8")
    assert split_prefix("EC1A") == ("EC1A", "")


def test_prefix_filter_matches_whole_outward_codes_only():
    frame = pl.DataFrame({"propertyAddress_postcode": ["GL5 2AD", "GL50 1AA", "GL5 3BB", None]})
    assert frame.filter(prefix_filter("GL5"))["propertyAddress_postcode"].to_list() == [
        "GL5 2AD",
        "GL5 3BB",
    ]
    assert frame.filter(prefix_filter("GL5 2"))["propertyAddress_postcode"].to_list() == ["GL5 2AD"]


def test_run_sparql_returns_json():
    session = FakeSession(FakeResponse(payload={"ok": 1}))
    assert run_sparql("q", session=session) == {"ok": 1}


def test_run_sparql_retries_timeouts_with_backoff(no_sleep):
    session = FakeSession(requests.exceptions.ReadTimeout(), FakeResponse(payload={"ok": 1}))
    assert run_sparql("q", session=session) == {"ok": 1}
    assert session.calls == 2
    assert len(no_sleep) == 1


def test_run_sparql_retries_a_dropped_connection(no_sleep):
    lookup_failed = requests.exceptions.ConnectionError(
        "Failed to resolve 'landregistry.data.gov.uk'"
    )
    session = FakeSession(lookup_failed, FakeResponse(payload={"ok": 1}))
    assert run_sparql("q", session=session) == {"ok": 1}
    assert session.calls == 2 and len(no_sleep) == 1


def test_run_sparql_honours_retry_after_header(no_sleep):
    busy = FakeResponse(status=429, headers={"Retry-After": "7"})
    session = FakeSession(busy, FakeResponse(payload={}))
    run_sparql("q", session=session)
    assert no_sleep == [pytest.approx(7.0, abs=0.5)]  # waited once, at the shared gate


def test_a_429_holds_back_every_thread(no_sleep):
    session = FakeSession(FakeResponse(status=429), FakeResponse(payload={}))
    run_sparql("q", session=session)
    assert len(no_sleep) == 1
    lr._GATE.close_for(30.0)  # as another thread's 429 would
    run_sparql("q", session=FakeSession(FakeResponse(payload={})))
    assert len(no_sleep) == 2 and no_sleep[-1] == pytest.approx(30.0, abs=0.5)


def test_a_503_backs_off_only_its_own_thread(no_sleep):
    session = FakeSession(FakeResponse(status=503), FakeResponse(payload={}))
    run_sparql("q", session=session)
    assert len(no_sleep) == 1
    run_sparql("q", session=FakeSession(FakeResponse(payload={})))
    assert len(no_sleep) == 1  # nothing else waits


def test_run_sparql_does_not_retry_client_errors():
    session = FakeSession(FakeResponse(status=404))
    with pytest.raises(requests.exceptions.HTTPError):
        run_sparql("q", session=session)
    assert session.calls == 1


def test_run_sparql_raises_http_errors_without_response():
    session = FakeSession(requests.exceptions.HTTPError())
    with pytest.raises(requests.exceptions.HTTPError):
        run_sparql("q", session=session)


def test_run_sparql_gives_up_after_max_retries():
    session = FakeSession(requests.exceptions.ReadTimeout(), requests.exceptions.ReadTimeout())
    with pytest.raises(requests.exceptions.ReadTimeout):
        run_sparql("q", session=session, max_retries=2)
    assert session.calls == 2


def test_run_sparql_without_attempts_reports_misconfiguration():
    with pytest.raises(RuntimeError, match="max_retries"):
        run_sparql("q", session=FakeSession(), max_retries=0)


def test_sparql_json_to_frame_fills_missing_bindings_with_null():
    payload = {
        "head": {"vars": ["a", "b"]},
        "results": {
            "bindings": [{"a": {"value": "1"}}, {"a": {"value": "2"}, "b": {"value": "x"}}]
        },
    }
    frame = sparql_json_to_frame(payload)
    assert frame.to_dict(as_series=False) == {"a": ["1", "2"], "b": [None, "x"]}


def _raw(**overrides: str) -> pl.DataFrame:
    row = {
        "pricePaid": "250000",
        "transactionDate": "2020-05-17",
        "propertyAddress_postcode": " gl5  2ad ",
        "propertyType": " terraced ",
        "propertyAddress_paon": "1",
        "propertyAddress_street": "HIGH ST",
        "transactionId": "{aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa}",
        "recordStatus": "http://landregistry.data.gov.uk/def/ppi/add",
        "extra": "ignored",
    }
    return pl.DataFrame([{**row, **overrides}])


def test_normalise_transactions_types_and_cleans_values():
    out = normalise_transactions(_raw())
    assert dict(out.schema) == TRANSACTION_SCHEMA
    assert out.row(0) == (
        250000,
        pl.Series(["2020-05-17"]).str.to_date()[0],
        "GL5 2AD",
        "TERRACED",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA",
        "ADD",
    )


def test_property_type_uris_are_reduced_to_their_label():
    uri = "http://landregistry.data.gov.uk/def/common/flat-maisonette"
    assert normalise_transactions(_raw(propertyType=uri))["propertyType"].to_list() == [
        "FLAT-MAISONETTE"
    ]


def test_normalise_transactions_drops_duplicate_copies_of_one_id():
    assert normalise_transactions(pl.concat([_raw(), _raw()])).height == 1


def test_simultaneous_flat_sales_stay_distinct():
    # Same retained address fields; the secondary address differs, and so does the transaction id.
    shared = {
        "pricePaid": "250000",
        "transactionDate": "2020-05-17",
        "propertyAddress_postcode": "GL5 2AD",
        "propertyType": "flat-maisonette",
        "propertyAddress_paon": "12",
        "propertyAddress_street": "HIGH ST",
        "recordStatus": "A",
    }
    first = _raw(
        **shared,
        propertyAddress_saon="FLAT 1",
        transactionId="{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAA1}",
    )
    second = _raw(
        **shared,
        propertyAddress_saon="FLAT 2",
        transactionId="{AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAA2}",
    )
    out = normalise_transactions(pl.concat([first, second]))
    assert out.height == 2
    assert "propertyAddress_saon" not in out.columns
    assert out.sort("transactionId")["transactionId"].to_list() == [
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAA1",
        "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAA2",
    ]


def test_a_change_replaces_an_add_even_when_the_add_is_later():
    ident = "{BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB}"
    change = _raw(
        transactionId=ident,
        pricePaid="180000",
        recordStatus="http://landregistry.data.gov.uk/def/ppi/change",
    )
    add = _raw(transactionId=ident, pricePaid="100000", recordStatus="A")
    out = normalise_transactions(pl.concat([change, add]))
    assert out.height == 1
    assert out["pricePaid"].to_list() == [180_000]
    assert out["recordStatus"].to_list() == ["CHANGE"]


def test_repeated_adds_keep_the_last_copy():
    ident = "DDDDDDDD-DDDD-DDDD-DDDD-DDDDDDDDDDDD"
    first = _raw(transactionId=ident, pricePaid="100000", recordStatus="ADD")
    second = _raw(transactionId=ident, pricePaid="120000", recordStatus="ADD")
    assert normalise_transactions(pl.concat([first, second]))["pricePaid"].to_list() == [120_000]


def test_a_delete_removes_the_sale():
    ident = "CCCCCCCC-CCCC-CCCC-CCCC-CCCCCCCCCCCC"
    add = _raw(transactionId=ident, recordStatus="add")
    delete = _raw(transactionId=ident, recordStatus="D")
    out = normalise_transactions(pl.concat([add, delete]))
    assert out.is_empty()
    assert dict(out.schema) == TRANSACTION_SCHEMA


def test_an_unknown_record_status_is_dropped_not_fatal(caplog):
    good = _raw()
    bad = _raw(recordStatus="nope", transactionId="{bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb}")
    with caplog.at_level("WARNING", logger="housing_star.data.land_registry"):
        out = normalise_transactions(pl.concat([good, bad]))
    assert out["transactionId"].to_list() == ["AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"]
    assert "Dropped 1 of 2" in caplog.text and "recordStatus" in caplog.text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pricePaid", "not a number"),
        ("pricePaid", "0"),
        ("transactionDate", "17/05/2020"),
        ("propertyAddress_postcode", "GL5"),
        ("propertyAddress_postcode", None),
        ("propertyType", "castle"),
        ("transactionId", "not-a-uuid"),
    ],
)
def test_one_malformed_sale_is_dropped_and_logged(field, value, caplog):
    other = _raw(transactionId="{bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb}")
    bad = _raw(**{field: value})
    with caplog.at_level("WARNING", logger="housing_star.data.land_registry"):
        out = normalise_transactions(pl.concat([other, bad], how="vertical_relaxed"))
    assert out["transactionId"].to_list() == ["BBBBBBBB-BBBB-BBBB-BBBB-BBBBBBBBBBBB"]
    assert f"'{field}': 1" in caplog.text


def test_a_frame_of_only_malformed_sales_normalises_to_empty(caplog):
    with caplog.at_level("WARNING", logger="housing_star.data.land_registry"):
        out = normalise_transactions(_raw(pricePaid="-5"))
    assert out.is_empty() and dict(out.schema) == TRANSACTION_SCHEMA
    assert "Dropped 1 of 1" in caplog.text


def test_valid_sales_log_nothing(caplog):
    with caplog.at_level("WARNING", logger="housing_star.data.land_registry"):
        normalise_transactions(_raw())
    assert caplog.text == ""


def test_normalise_transactions_of_empty_frame_has_canonical_schema():
    assert dict(normalise_transactions(pl.DataFrame()).schema) == TRANSACTION_SCHEMA


def test_normalise_transactions_requires_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        normalise_transactions(_raw().drop("pricePaid"))


def _payload(n: int, first: int = 0) -> dict:
    """``n`` distinct valid sales, as the endpoint would return them."""
    names = list(TRANSACTION_SCHEMA)
    bindings = []
    for k in range(first, first + n):
        raw = _raw(transactionId=f"AAAAAAAA-AAAA-AAAA-AAAA-{k:012d}").drop("extra")
        bindings.append({key: {"value": value} for key, value in raw.row(0, named=True).items()})
    return {"head": {"vars": names}, "results": {"bindings": bindings}}


def test_fetch_transactions_makes_one_request_when_the_answer_fits(monkeypatch):
    calls: list[str] = []

    def fake_run(query: str, **_: Any) -> dict:
        calls.append(query)
        return _payload(2)

    monkeypatch.setattr(lr, "run_sparql", fake_run)
    out = fetch_transactions(PPDQuerySpec(("GL5", "GL6"), "2020-01-01", "2021-01-01", limit=2))
    assert len(calls) == 1 and out.height == 2


def test_a_full_answer_is_split_by_date(monkeypatch):
    windows: list[tuple[str, str]] = []

    def fake_run(query: str, **_: Any) -> dict:
        start, end = re.findall(r'"(\d{4}-\d{2}-\d{2})"\^\^xsd:date', query)
        windows.append((start, end))
        # The whole year has more than the limit; each half fits.
        return _payload(
            3 if (start, end) == ("2020-01-01", "2021-01-01") else 2, first=len(windows) * 10
        )

    monkeypatch.setattr(lr, "run_sparql", fake_run)
    out = fetch_transactions(PPDQuerySpec("GL5", "2020-01-01", "2021-01-01", limit=2))
    assert windows == [
        ("2020-01-01", "2021-01-01"),
        ("2020-01-01", "2020-07-02"),
        ("2020-07-02", "2021-01-01"),
    ]
    assert out.height == 4  # the two halves, not the discarded full answer


def test_a_single_day_over_the_limit_cannot_be_split(monkeypatch):
    monkeypatch.setattr(lr, "run_sparql", lambda query, **_: _payload(3))
    with pytest.raises(RuntimeError, match="raise PPDQuerySpec.limit"):
        fetch_transactions(PPDQuerySpec("GL5", "2020-01-01", "2020-01-02", limit=2))


def test_resolve_transactions_rejects_an_unknown_record_status():
    frame = pl.DataFrame({"transactionId": ["A"], "recordStatus": ["MAYBE"]})
    with pytest.raises(ValueError, match="recordStatus"):
        lr.resolve_transactions(frame)
