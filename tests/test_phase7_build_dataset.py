from phase7.build_dataset import (
    _adjacent_comparable_pairs,
    _dominant_period_length_facts,
    _same_period_length,
    _stable_bucket,
)
from tools.schema import FinancialFact

TICKER, CIK = "ACME", "0000000001"


def _fact(period_start, period_end, value=100.0, accn="a", filed="2023-01-01"):
    return FinancialFact(
        ticker=TICKER, cik=CIK, concept="Revenues", label="Revenues", value=value, unit="USD",
        period_start=period_start, period_end=period_end, fiscal_year=2023, fiscal_period="Q1",
        form="10-Q", filed=filed, accession_number=accn,
    )


def test_same_period_length_true_for_two_quarters():
    q1 = _fact("2023-01-01", "2023-03-31")
    q2 = _fact("2023-04-01", "2023-06-30")
    assert _same_period_length(q1, q2)


def test_same_period_length_false_for_quarter_vs_nine_month_cumulative():
    # The actual bug this guards: a quarter and a year-to-date cumulative figure
    # can share the *same* period_end while covering very different spans.
    quarter = _fact("2016-08-01", "2016-10-30")  # ~90 days
    nine_month_ytd = _fact("2016-02-01", "2016-10-30")  # ~270 days, same end date
    assert not _same_period_length(quarter, nine_month_ytd)


def test_adjacent_comparable_pairs_excludes_same_period_end():
    quarter = _fact("2016-08-01", "2016-10-30", accn="q")
    ytd = _fact("2016-02-01", "2016-10-30", accn="ytd")
    pairs = _adjacent_comparable_pairs(sorted([quarter, ytd], key=lambda f: f.period_end))
    assert pairs == []


def test_adjacent_comparable_pairs_finds_the_nearest_comparable_prior_period():
    q1 = _fact("2023-01-01", "2023-03-31", accn="q1")
    q2 = _fact("2023-04-01", "2023-06-30", accn="q2")
    q3 = _fact("2023-07-01", "2023-09-30", accn="q3")
    pairs = _adjacent_comparable_pairs([q1, q2, q3])
    assert (q2, q3) in pairs
    assert (q1, q2) in pairs


def test_dominant_period_length_facts_picks_the_majority_bucket():
    quarters = [_fact(f"2023-0{i}-01", f"2023-0{i+2}-28", accn=f"q{i}") for i in range(1, 7, 2)]
    annual = [_fact("2022-01-01", "2022-12-31", accn="fy")]
    facts = quarters + annual
    result = _dominant_period_length_facts(facts)
    assert set(result) == set(quarters)
    assert annual[0] not in result


def test_stable_bucket_is_deterministic_across_calls():
    # Regression: the original implementation used Python's builtin hash(),
    # which is randomized per-process (PYTHONHASHSEED) for strings — meaning
    # re-running the dataset builder would silently produce a different
    # train/test split every time, not a stable one.
    assert _stable_bucket("abs-123") == _stable_bucket("abs-123")


def test_stable_bucket_matches_a_known_value():
    # Pins the actual hash function (not just "it's consistent with itself"),
    # so a future refactor that swaps the hash algorithm gets caught here.
    assert _stable_bucket("abs-123") == int(__import__("hashlib").md5(b"abs-123").hexdigest(), 16) % 5


def test_stable_bucket_spreads_across_all_buckets():
    buckets = {_stable_bucket(f"id-{i}") for i in range(200)}
    assert buckets == set(range(5))
