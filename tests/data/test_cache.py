from datetime import date
from pathlib import Path

import polars as pl
import pytest

from housing_star.data.cache import (
    CACHE_ENV_VAR,
    DownloadStats,
    TransactionCache,
    ancestors,
    cache_filename,
    default_cache_dir,
    merge_intervals,
    missing_intervals,
    normalise_prefix,
    plan_requests,
    prefix_dir,
    round_window,
    year_pieces,
)
from housing_star.data.land_registry import PPDQuerySpec
from housing_star.data.schema import TRANSACTION_SCHEMA
from tests.conftest import fake_transaction_id, synthetic_transactions


def d(year: int, month: int = 1) -> date:
    return date(year, month, 1)


def test_default_cache_dir_prefers_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
    assert default_cache_dir() == tmp_path


def test_default_cache_dir_falls_back_to_user_cache(monkeypatch):
    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    assert default_cache_dir().name == "housing_star"


def test_normalise_prefix():
    assert normalise_prefix(" gl5   2b ") == "GL5 2B"


@pytest.mark.parametrize(
    ("prefix", "parts"),
    [("GL5", ["GL5"]), ("gl5 2", ["GL5", "2"]), ("GL5 2B", ["GL5", "2", "B"])],
)
def test_prefix_dir_nests_characters(tmp_path, prefix, parts):
    assert prefix_dir(tmp_path, prefix) == tmp_path.joinpath(*parts)


def test_prefix_dir_rejects_empty_prefix(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        prefix_dir(tmp_path, "  ")


def test_merge_intervals_joins_overlapping_and_adjacent():
    merged = merge_intervals([(d(2003), d(2005)), (d(2000), d(2002)), (d(2002), d(2003))])
    assert merged == [(d(2000), d(2005))]


@pytest.mark.parametrize(
    ("request_", "covered", "expected"),
    [
        ((d(2000), d(2005)), [], [(d(2000), d(2005))]),
        ((d(2000), d(2005)), [(d(1990), d(2010))], []),
        ((d(2000), d(2005)), [(d(2001), d(2002))], [(d(2000), d(2001)), (d(2002), d(2005))]),
        ((d(2000), d(2005)), [(d(1990), d(1995)), (d(2007), d(2009))], [(d(2000), d(2005))]),
        ((d(2000), d(2005)), [(d(2000), d(2003))], [(d(2003), d(2005))]),
    ],
)
def test_missing_intervals(request_, covered, expected):
    assert missing_intervals(request_, covered) == expected


class CountingFetcher:
    def __init__(self):
        self.specs = []

    def __call__(self, spec):
        self.specs.append((", ".join(spec.prefixes), spec.start_date, spec.end_date))
        return synthetic_transactions(spec)


@pytest.fixture
def fetcher():
    return CountingFetcher()


@pytest.fixture
def cache(tmp_path, fetcher):
    return TransactionCache(root=tmp_path, fetcher=fetcher)


def test_first_call_fetches_the_whole_window(cache, fetcher):
    out = cache.transactions("GL5", "2000-01-01", "2003-01-01")
    assert fetcher.specs == [("GL5", "2000-01-01", "2003-01-01")]
    assert out["transactionDate"].min() >= d(2000) and out["transactionDate"].max() < d(2003)
    assert dict(out.schema) == TRANSACTION_SCHEMA
    assert list((cache.directory / "GL5").iterdir()) == [
        cache.directory / "GL5" / "transactions_20000101_20030101.parquet"
    ]


def test_second_call_is_served_from_disk(cache, fetcher):
    first = cache.transactions("GL5", "2000-01-01", "2003-01-01")
    second = cache.transactions("gl5", d(2000), d(2003))
    assert len(fetcher.specs) == 1
    assert first.equals(second)


def test_extending_the_window_fetches_only_the_gap(cache, fetcher):
    cache.transactions("GL5", "2000-01-01", "2002-01-01")
    cache.transactions("GL5", "2001-01-01", "2004-01-01")
    assert fetcher.specs[1:] == [("GL5", "2002-01-01", "2004-01-01")]


def test_empty_responses_still_mark_the_interval_as_cached(tmp_path):
    calls = []
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: calls.append(spec) or pl.DataFrame()
    )
    assert cache.transactions("GL5", "2000-01-01", "2001-01-01").is_empty()
    cache.transactions("GL5", "2000-01-01", "2001-01-01")
    assert len(calls) == 1


def test_window_must_be_non_empty(cache):
    with pytest.raises(ValueError, match="must be after start"):
        cache.transactions("GL5", "2001-01-01", "2001-01-01")


def test_cached_ranges_ignores_unrelated_files(cache, tmp_path):
    cache.transactions("GL5", "2000-01-01", "2001-01-01")
    (tmp_path / "GL5" / "notes.txt").write_text("x")
    (tmp_path / "GL5" / "transactions_oops.parquet").write_bytes(b"")
    assert [(s, e) for s, e, _ in cache.cached_ranges("GL5")] == [(d(2000), d(2001))]


def test_files_written_by_older_versions_with_extra_columns_are_readable(tmp_path):
    folder = tmp_path / "GL5"
    folder.mkdir()
    legacy = synthetic_transactions(PPDQuerySpec("GL5", "2000-01-01", "2001-01-01")).with_columns(
        uniqueId=pl.lit(1, dtype=pl.UInt64)
    )
    legacy.write_parquet(folder / cache_filename(d(2000), d(2001)))
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    assert cache.transactions("GL5", "2000-01-01", "2001-01-01").height == legacy.height


def test_extracts_saved_without_a_transaction_id_are_refetched(tmp_path):
    folder = tmp_path / "GL5"
    folder.mkdir()
    legacy = folder / "transactions_20000101_20010101.parquet"
    pl.DataFrame({"pricePaid": [1]}).write_parquet(legacy)
    calls = []
    cache = TransactionCache(
        root=tmp_path,
        fetcher=lambda spec: calls.append(spec) or synthetic_transactions(spec),
        verbose=False,
    )
    assert cache.missing("GL5", "2000-01-01", "2001-01-01") == [(d(2000), d(2001))]
    out = cache.transactions("GL5", "2000-01-01", "2001-01-01")
    assert len(calls) == 1 and out.height > 0
    assert "transactionId" in pl.read_parquet_schema(legacy)


def test_versioned_filenames_are_readable_and_prefer_the_current_name(cache, fetcher):
    folder = cache.directory / "GL5"
    folder.mkdir()
    start, end = d(2000), d(2001)
    versioned = folder / "transactions_v2_20000101_20010101.parquet"
    pl.DataFrame(schema=TRANSACTION_SCHEMA).write_parquet(versioned)
    assert cache.cached_ranges("GL5") == [(start, end, versioned)]
    assert cache.transactions("GL5", start, end).is_empty()
    assert fetcher.specs == []

    current = folder / cache_filename(start, end)
    pl.DataFrame(schema=TRANSACTION_SCHEMA).write_parquet(current)
    assert cache.cached_ranges("GL5") == [(start, end, current)]


def _cached_sale(transaction_id: str, price: int, status: str = "ADD") -> dict:
    return {
        "pricePaid": price,
        "transactionDate": d(2020, 3),
        "propertyAddress_postcode": "GL5 2AD",
        "propertyType": "FLAT-MAISONETTE",
        "propertyAddress_paon": "12",
        "propertyAddress_street": "HIGH ST",
        "transactionId": transaction_id,
        "recordStatus": status,
    }


def test_overlapping_files_keep_each_flat_once_and_prefer_a_change(tmp_path):
    folder = tmp_path / "GL5"
    folder.mkdir()
    flat_1 = fake_transaction_id("flat", 1)
    flat_2 = fake_transaction_id("flat", 2)
    # The corrected price is in the earlier file. Status, not file order, decides.
    pl.DataFrame(
        [_cached_sale(flat_1, 180_000, "CHANGE"), _cached_sale(flat_2, 200_000)]
    ).write_parquet(folder / cache_filename(d(2020), d(2020, 7)))
    pl.DataFrame(
        [_cached_sale(flat_1, 100_000, "ADD"), _cached_sale(flat_2, 200_000)]
    ).write_parquet(folder / cache_filename(d(2020, 6), d(2021)))
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    out = cache.transactions("GL5", "2020-01-01", "2021-01-01").sort("transactionId")
    assert out.height == 2
    corrected = out.filter(pl.col("transactionId") == flat_1)
    assert corrected["pricePaid"].to_list() == [180_000]
    assert corrected["recordStatus"].to_list() == ["CHANGE"]


def test_a_delete_in_one_overlapping_file_removes_the_sale(tmp_path):
    folder = tmp_path / "GL5"
    folder.mkdir()
    ident = fake_transaction_id("gone")
    pl.DataFrame([_cached_sale(ident, 100_000, "ADD")]).write_parquet(
        folder / cache_filename(d(2020), d(2020, 7))
    )
    pl.DataFrame([_cached_sale(ident, 100_000, "DELETE")]).write_parquet(
        folder / cache_filename(d(2020, 6), d(2021))
    )
    cache = TransactionCache(
        root=tmp_path, fetcher=lambda spec: pytest.fail(str(spec)), verbose=False
    )
    assert cache.transactions("GL5", "2020-01-01", "2021-01-01").is_empty()


def test_root_defaults_to_the_user_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))
    assert TransactionCache().directory == Path(tmp_path)


def test_downloads_are_announced_but_cache_hits_are_silent(cache, capsys):
    cache.transactions("GL5", "2000-01-01", "2002-01-01")
    first = capsys.readouterr().err
    assert "downloading 1 postcode area(s)" in first and "slow the first time" in first
    assert "1/1: GL5 2000-01-01 to 2002-01-01, 16 sales" in first  # completion line
    cache.transactions("GL5", "2000-01-01", "2002-01-01")
    assert capsys.readouterr().err == ""


def test_verbose_can_be_switched_off(tmp_path, fetcher, capsys):
    quiet = TransactionCache(root=tmp_path, fetcher=fetcher, verbose=False)
    quiet.transactions("GL5", "2000-01-01", "2001-01-01")
    assert capsys.readouterr().err == ""


def test_missing_reports_what_would_be_downloaded(cache):
    assert cache.missing("GL5", "2000-01-01", "2003-01-01") == [(d(2000), d(2003))]
    cache.transactions("GL5", "2000-01-01", "2002-01-01")
    assert cache.missing("GL5", "2000-01-01", "2003-01-01") == [(d(2002), d(2003))]
    assert cache.missing("GL5", "2000-01-01", "2002-01-01") == []


def test_the_root_may_be_given_as_a_string(tmp_path, fetcher):
    text_root = TransactionCache(root=str(tmp_path), fetcher=fetcher, verbose=False)
    assert text_root.transactions("GL5", "2000-01-01", "2001-01-01").height > 0


def test_ancestors_and_year_pieces():
    assert ancestors("GL6 8A") == ["GL6 8", "GL6"]
    assert ancestors("GL6") == []
    assert year_pieces(date(2000, 6, 1), date(2002, 3, 1)) == [
        (date(2000, 6, 1), d(2001)),
        (d(2001), d(2002)),
        (d(2002), date(2002, 3, 1)),
    ]


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        ((date(2001, 3, 1), date(2005, 6, 1)), (d(2001), d(2006))),
        ((d(2001), d(2005)), (d(2001), d(2005))),
        # Never past the start of the current year: recent months stay uncached until asked for.
        ((d(2020), date(2026, 6, 1)), (d(2020), date(2026, 6, 1))),
        ((d(2020), date(2025, 6, 1)), (d(2020), d(2026))),
    ],
)
def test_round_window(window, expected):
    assert round_window(window, today=date(2026, 11, 19)) == expected


def test_plan_requests_batches_prefixes_and_cuts_long_gaps():
    gap = [(d(2000), d(2010))]
    # Ten districts planned at 500 sales a year: 5,000 a year, so one year per request at 5,000.
    jobs = plan_requests({f"GL{k}": gap for k in range(1, 11)}, rows_per_request=5_000)
    assert len(jobs) == 10 and all(len(job.prefixes) == 10 for job in jobs)
    assert [(job.start, job.end) for job in jobs][:2] == [(d(2000), d(2001)), (d(2001), d(2002))]
    # One district: ten years fit in one request.
    assert [(j.prefixes, j.start, j.end) for j in plan_requests({"GL5": gap}, 5_000)] == [
        (("GL5",), d(2000), d(2010))
    ]
    # Sectors are planned at a lower rate, so more of them fit.
    sectors = plan_requests({f"GL5 {k}": gap for k in range(10)}, rows_per_request=15_000)
    assert len(sectors) == 1 and len(sectors[0].prefixes) == 10


def test_plan_requests_splits_batches_by_size_and_count():
    gap = [(d(2000), d(2001))]
    jobs = plan_requests({f"GL{k}": gap for k in range(1, 5)}, rows_per_request=1_000)
    assert [len(job.prefixes) for job in jobs] == [2, 2]
    many = plan_requests({f"{a}{k}": gap for a in ("AB", "CD") for k in range(1, 51)}, 10**9)
    assert [len(job.prefixes) for job in many] == [40, 40, 20]


def test_plan_requests_keeps_different_gaps_apart():
    jobs = plan_requests({"GL5": [(d(2000), d(2002))], "GL6": [(d(2001), d(2002))]}, 10_000)
    assert {(job.prefixes, job.start) for job in jobs} == {(("GL5",), d(2000)), (("GL6",), d(2001))}


def test_prefetch_fetches_a_neighbourhood_in_one_request(cache, fetcher):
    cache.prefetch(["GL5", "GL6", "GL10"], "2000-01-01", "2002-01-01")
    assert fetcher.specs == [("GL10, GL5, GL6", "2000-01-01", "2002-01-01")]
    for prefix in ("GL5", "GL6", "GL10"):
        out = cache.transactions(prefix, "2000-01-01", "2002-01-01")
        assert (
            out.height == 16 and out["propertyAddress_postcode"].str.starts_with(f"{prefix} ").all()
        )
    assert len(fetcher.specs) == 1
    assert cache.stats.requests == 1 and cache.stats.sales == 48


def test_prefetch_runs_requests_in_parallel(tmp_path, fetcher):
    cache = TransactionCache(
        root=tmp_path, fetcher=fetcher, verbose=False, max_workers=3, rows_per_request=500
    )
    cache.prefetch(["GL5", "GL6"], "2000-01-01", "2004-01-01")
    assert len(fetcher.specs) == 8  # one district-year per request at this size
    assert cache.stats.requests == 8 and cache.stats.seconds > 0
    assert cache.transactions("GL6", "2000-01-01", "2004-01-01").height == 32
    assert len(fetcher.specs) == 8


def test_prefetch_with_nothing_missing_sends_nothing(cache, fetcher):
    cache.prefetch(["GL5"], "2000-01-01", "2001-01-01")
    cache.prefetch(["GL5"], "2000-01-01", "2001-01-01")
    assert len(fetcher.specs) == 1


def test_prefetch_rounds_a_past_window_out_to_whole_years(cache, fetcher):
    cache.transactions("GL5", "2000-03-01", "2001-06-01")
    assert fetcher.specs == [("GL5", "2000-01-01", "2002-01-01")]
    cache.transactions("GL5", "2001-07-01", "2001-12-01")
    assert len(fetcher.specs) == 1


def test_a_sector_is_read_from_its_district(cache, fetcher):
    cache.transactions("GL5", "2000-01-01", "2002-01-01")
    sector = cache.transactions("GL5 1", "2000-01-01", "2002-01-01")
    assert len(fetcher.specs) == 1  # no request for the sector
    assert sector.height == 16 and sector["propertyAddress_postcode"].str.starts_with("GL5 1").all()
    assert cache.transactions("GL5 2", "2000-01-01", "2002-01-01").is_empty()
    assert cache.missing("GL5 1", "2000-01-01", "2003-01-01") == [(d(2002), d(2003))]


def test_a_sector_fetches_only_what_its_district_does_not_cover(cache, fetcher):
    cache.transactions("GL5", "2000-01-01", "2001-01-01")
    cache.transactions("GL5 1", "2000-01-01", "2002-01-01")
    assert fetcher.specs == [
        ("GL5", "2000-01-01", "2001-01-01"),
        ("GL5 1", "2001-01-01", "2002-01-01"),
    ]


def test_a_sector_is_skipped_when_its_district_comes_in_the_same_prefetch(cache, fetcher):
    cache.prefetch(["GL5 1", "GL5"], "2000-01-01", "2001-01-01")
    assert fetcher.specs == [("GL5", "2000-01-01", "2001-01-01")]


def test_prefetch_checks_its_arguments(cache):
    with pytest.raises(ValueError, match="must be after start"):
        cache.prefetch(["GL5"], "2001-01-01", "2000-01-01")
    with pytest.raises(ValueError, match="Unsupported postcode prefix"):
        cache.prefetch(["not a postcode"], "2000-01-01", "2001-01-01")


def test_cache_settings_are_checked(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        TransactionCache(root=tmp_path, max_workers=0)
    with pytest.raises(ValueError, match="at least 1"):
        TransactionCache(root=tmp_path, rows_per_request=0)


def test_requests_carry_twice_the_planned_size_as_their_limit(tmp_path):
    specs: list[PPDQuerySpec] = []
    cache = TransactionCache(
        root=tmp_path,
        fetcher=lambda spec: specs.append(spec) or synthetic_transactions(spec),
        verbose=False,
        rows_per_request=5_000,
    )
    cache.transactions("GL5", "2000-01-01", "2001-01-01")
    assert specs[0].limit == 10_000


def test_download_stats_start_at_zero(tmp_path):
    assert TransactionCache(root=tmp_path).stats == DownloadStats()


def test_a_history_downloaded_in_pieces_is_merged_into_one_file(tmp_path, fetcher):
    cache = TransactionCache(root=tmp_path, fetcher=fetcher, verbose=False, rows_per_request=500)
    whole = cache.transactions("GL5", "2000-01-01", "2004-01-01")
    assert len(fetcher.specs) == 4  # one year per request at this size
    assert [p.name for p in (tmp_path / "GL5").iterdir()] == [
        "transactions_20000101_20040101.parquet"
    ]
    assert cache.transactions("GL5", "2000-01-01", "2004-01-01").equals(whole)


def test_merging_keeps_separate_runs_apart_and_reads_the_same(tmp_path, fetcher):
    cache = TransactionCache(root=tmp_path, fetcher=fetcher, verbose=False)
    cache.transactions("GL5", "2000-01-01", "2001-01-01")
    cache.transactions("GL5", "2003-01-01", "2004-01-01")
    assert sorted(p.name for p in (tmp_path / "GL5").iterdir()) == [
        "transactions_20000101_20010101.parquet",
        "transactions_20030101_20040101.parquet",
    ]
    before = cache.transactions("GL5", "2000-01-01", "2001-01-01")
    cache.transactions("GL5", "2001-01-01", "2003-01-01")  # fills the gap: one run now
    assert [p.name for p in (tmp_path / "GL5").iterdir()] == [
        "transactions_20000101_20040101.parquet"
    ]
    assert cache.transactions("GL5", "2000-01-01", "2001-01-01").equals(before)


def test_older_files_are_merged_with_new_ones(tmp_path, fetcher):
    folder = tmp_path / "GL5"
    folder.mkdir()
    legacy = synthetic_transactions(PPDQuerySpec("GL5", "2000-01-01", "2001-01-01")).with_columns(
        uniqueId=pl.lit(1, dtype=pl.UInt64)
    )
    legacy.write_parquet(folder / "transactions_v2_20000101_20010101.parquet")
    cache = TransactionCache(root=tmp_path, fetcher=fetcher, verbose=False)
    out = cache.transactions("GL5", "2000-01-01", "2002-01-01")
    assert fetcher.specs == [("GL5", "2001-01-01", "2002-01-01")]
    assert [p.name for p in folder.iterdir()] == ["transactions_20000101_20020101.parquet"]
    assert out.height == 16


def test_a_file_inside_another_files_interval_is_folded_into_it(tmp_path, fetcher):
    cache = TransactionCache(root=tmp_path, fetcher=fetcher, verbose=False)
    whole = cache.transactions("GL5", "2000-01-01", "2004-01-01")
    inner = synthetic_transactions(PPDQuerySpec("GL5", "2001-01-01", "2002-01-01"))
    inner.write_parquet(tmp_path / "GL5" / cache_filename(d(2001), d(2002)))
    again = cache.transactions("GL5", "2000-01-01", "2004-01-01")
    assert again.sort("transactionId").equals(whole.sort("transactionId"))  # row order may differ
    assert [p.name for p in (tmp_path / "GL5").iterdir()] == [
        "transactions_20000101_20040101.parquet"
    ]
    assert len(fetcher.specs) == 1
