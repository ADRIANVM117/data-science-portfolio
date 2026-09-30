from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pit_cash_realization import OCF_TAG, cash_realization_as_of, ttm_operating_cash_flow_as_of  # noqa: E402


DECISION = "2025-01-03T14:30:00Z"


def fact(tag: str, start: str, end: str, value: float, form: str, accn: str, accepted: str = "2025-01-02T12:00:00Z") -> dict[str, object]:
    return {
        "issuer": "TEST", "cik": "0000000001", "tag": tag, "unit": "USD", "start": start, "end": end,
        "value": value, "form": form, "accn": accn, "acceptanceDateTime": accepted,
    }


def annual_facts(tag: str, q1: float, q2: float, q3: float, q4: float, accepted: str = "2025-01-02T12:00:00Z") -> pd.DataFrame:
    return pd.DataFrame([
        fact(tag, "2024-01-01", "2024-03-31", q1, "10-Q", "q1", accepted),
        fact(tag, "2024-01-01", "2024-06-30", q1 + q2, "10-Q", "h1", accepted),
        fact(tag, "2024-04-01", "2024-06-30", q2, "10-Q", "q2", accepted),
        fact(tag, "2024-01-01", "2024-09-30", q1 + q2 + q3, "10-Q", "y9", accepted),
        fact(tag, "2024-07-01", "2024-09-30", q3, "10-Q", "q3", accepted),
        fact(tag, "2024-01-01", "2024-12-31", q1 + q2 + q3 + q4, "10-K", "fy", accepted),
    ])


def test_ocf_ttm_reconstructs_quarters_and_retains_ocf_provenance() -> None:
    result = ttm_operating_cash_flow_as_of(annual_facts(OCF_TAG, 10, 20, 30, 40), "TEST", DECISION)
    assert result.status == "VALID"
    assert result.value == 100
    assert set(result.quarters["method"]) == {"direct", "reconstructed"}
    assert set(result.quarters["accounting_tag"]) == {OCF_TAG}
    assert result.quarters.iloc[-1]["source_accessions"] == ("fy", "y9")


def test_ocf_future_fact_and_ambiguous_vintage_return_na() -> None:
    future = annual_facts(OCF_TAG, 10, 20, 30, 40, "2025-01-03T15:00:00Z")
    assert ttm_operating_cash_flow_as_of(future, "TEST", DECISION).status == "NA_FEWER_THAN_FOUR_VALID_QUARTERS"
    ambiguous = annual_facts(OCF_TAG, 10, 20, 30, 40)
    ambiguous = pd.concat([ambiguous, pd.DataFrame([fact(OCF_TAG, "2024-01-01", "2024-03-31", 11, "10-Q", "conflict")])])
    assert ttm_operating_cash_flow_as_of(ambiguous, "TEST", DECISION).status != "VALID"


def test_ocf_amendment_applies_prospectively() -> None:
    original = annual_facts(OCF_TAG, 10, 20, 30, 40)
    amended = annual_facts(OCF_TAG, 12, 21, 31, 40, "2025-02-03T15:00:00Z")
    amended["form"] = amended["form"] + "/A"
    amended["accn"] = "amended-" + amended["accn"]
    facts = pd.concat([original, amended])
    before = ttm_operating_cash_flow_as_of(facts, "TEST", "2025-02-03T14:30:00Z")
    after = ttm_operating_cash_flow_as_of(facts, "TEST", "2025-02-04T14:30:00Z")
    assert before.value == 100
    assert after.value == 104


def test_ratio_below_one_is_not_changed() -> None:
    result = cash_realization_as_of(
        annual_facts("NetIncomeLoss", 25, 25, 25, 25), annual_facts(OCF_TAG, 20, 25, 25, 29), "TEST", DECISION
    )
    assert result.status == "VALID"
    assert result.raw_ratio == result.cash_realization == 0.99


def test_upper_cap_applies_only_above_one() -> None:
    result = cash_realization_as_of(
        annual_facts("NetIncomeLoss", 25, 25, 25, 25), annual_facts(OCF_TAG, 30, 30, 30, 30), "TEST", DECISION
    )
    assert result.raw_ratio == 1.2
    assert result.cash_realization == 1.0


def test_negative_ocf_is_not_floored() -> None:
    result = cash_realization_as_of(
        annual_facts("NetIncomeLoss", 25, 25, 25, 25), annual_facts(OCF_TAG, -10, -20, -30, -40), "TEST", DECISION
    )
    assert result.status == "VALID"
    assert result.raw_ratio == result.cash_realization == -1.0


def test_nonpositive_ni_and_invalid_ocf_return_explained_na() -> None:
    nonpositive = cash_realization_as_of(
        annual_facts("NetIncomeLoss", -10, 5, 5, 0), annual_facts(OCF_TAG, 10, 10, 10, 10), "TEST", DECISION
    )
    invalid_ocf = cash_realization_as_of(
        annual_facts("NetIncomeLoss", 25, 25, 25, 25), annual_facts(OCF_TAG, 10, 10, 10, 10).iloc[:-1], "TEST", DECISION
    )
    assert nonpositive.status == "NONPOSITIVE_NET_INCOME" and nonpositive.cash_realization is None
    assert invalid_ocf.status == "INVALID_TTM_OCF" and invalid_ocf.cash_realization is None


def test_invalid_pit_net_income_returns_explained_na() -> None:
    result = cash_realization_as_of(
        annual_facts("NetIncomeLoss", 25, 25, 25, 25).iloc[:-1],
        annual_facts(OCF_TAG, 25, 25, 25, 25), "TEST", DECISION,
    )
    assert result.status == "INVALID_TTM_NET_INCOME"
    assert result.cash_realization is None
