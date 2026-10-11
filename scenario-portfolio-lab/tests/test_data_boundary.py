from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import polars as pl

from scenario_portfolio_lab.catalog import CatalogError, read_catalog, report_catalog
from scenario_portfolio_lab.eligibility import eligible
from scenario_portfolio_lab.ingest import ProviderResponseError, parse_daily_adjusted, validate_cutoff
from scenario_portfolio_lab.matrices import coverage, simple_returns
from scenario_portfolio_lab.pipeline import build_from_raw
from scripts.verify_provider_symbols import append_event, successful_ids


def payload(symbol: str, *, zero_volume: bool = False) -> dict:
    return {"Meta Data":{"2. Symbol":symbol,"3. Last Refreshed":"2020-01-03","5. Time Zone":"US/Eastern"},"Time Series (Daily)":{"2020-01-02":{"1. open":"10","2. high":"11","3. low":"9","4. close":"10","5. adjusted close":"10","6. volume":"0" if zero_volume else "100","7. dividend amount":"0","8. split coefficient":"1"},"2020-01-03":{"1. open":"11","2. high":"12","3. low":"10","4. close":"11","5. adjusted close":"11","6. volume":"100","7. dividend amount":"0","8. split coefficient":"1"}}}


class DataBoundaryTests(unittest.TestCase):
    def test_provider_identity_and_nonfinite_are_rejected(self):
        with self.assertRaises(ProviderResponseError): parse_daily_adjusted(payload("AAA"), "BBB")
        broken=payload("AAA"); broken["Time Series (Daily)"]["2020-01-02"]["1. open"]="nan"
        with self.assertRaises(ProviderResponseError): parse_daily_adjusted(broken, "AAA")

    def test_period_is_enforced_and_missing_expected_session_has_no_daily_return(self):
        old=payload("AAA"); old["Time Series (Daily)"]={"2017-12-29":old["Time Series (Daily)"]["2020-01-02"]}
        with self.assertRaises(ProviderResponseError): parse_daily_adjusted(old, "AAA")
        prices=pl.DataFrame({"session_date":[date(2020,1,2),date(2020,1,6)],"AAA":[10.0,11.0]})
        returned=simple_returns(prices,{"AAA":{date(2020,1,2),date(2020,1,3),date(2020,1,6)}})
        self.assertIsNone(returned["AAA"][1])
        with self.assertRaises(ValueError): validate_cutoff("2026-10-10")

    def test_leading_expected_session_gaps_are_insufficient_history(self):
        prices=pl.DataFrame({"session_date":[date(2020,1,2),date(2020,1,3),date(2020,1,6)],"AAA":[None,10.0,11.0]})
        result=coverage(prices,"2020-01-06",2,{"AAA":{date(2020,1,2),date(2020,1,3),date(2020,1,6)}})[0]
        self.assertEqual((result.leading_nulls,result.interior_nulls,result.reason),(1,0,"INSUFFICIENT_HISTORY"))

    def test_catalog_duplicates_are_reported_and_ids_must_be_unique(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"u.csv"; p.write_text("instrument_id,reference_symbol,asset_class,provider_symbol_candidate,mapping_type,provider_mapping_state,listing_market,listing_currency,identity_verification_state,etf_structure_state,direction,leverage_multiple,public_evidence_status,review_reason\nU1,A,equity,AAA,x,PENDING,,,,NOT_APPLICABLE,,,PENDING,x\nU2,B,equity,AAA,x,PENDING,,,,NOT_APPLICABLE,,,PENDING,x\n",encoding="utf-8")
            r=report_catalog(read_catalog(p)); self.assertEqual(r.duplicate_provider_candidates,("AAA",))
            p.write_text(p.read_text(encoding="utf-8").replace("U2,B","U1,B"),encoding="utf-8")
            with self.assertRaises(CatalogError): read_catalog(p)

    def test_etf_restrictions_fail_closed(self):
        base={"asset_class":"etf","identity_verification_state":"VERIFIED","listing_currency":"USD","listing_market":"NYSE","etf_structure_state":"CONFIRMED_DIRECT_UNLEVERAGED","direction":"BULL","leverage_multiple":"1.0"}
        self.assertEqual(eligible(base,True),(True,"ELIGIBLE"))
        base["direction"]="INVERSE"; base["leverage_multiple"]="-1.0"; self.assertEqual(eligible(base,True),(False,"ETF_NOT_DIRECT_UNLEVERAGED"))
        base["etf_structure_state"]="PENDING"; self.assertEqual(eligible(base,True),(False,"ETF_STRUCTURE_REVIEW"))

    def test_full_offline_flow_keeps_zero_volume_and_writes_manifest(self):
        record={"asset_class":"equity","identity_verification_state":"VERIFIED","listing_currency":"USD","listing_market":"NASDAQ"}
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"cut_2020-01-03"; result=build_from_raw(raw_payloads={"AAA":json.dumps(payload("AAA",zero_volume=True)).encode()},records={"AAA":record},cutoff="2020-01-03",output=output,expected_sessions={"AAA":{date(2020,1,2),date(2020,1,3)}},calendar_metadata={"AAA":{"source":"synthetic","version":"test"}})
            self.assertEqual(result["status"],"COMPLETED"); self.assertIn("VALID_ZERO_VOLUME",result["quality"]["AAA"]["warnings"])
            self.assertTrue((output/"gold"/"simple_total_return_matrix.parquet").is_file()); self.assertTrue((output/"run_manifest.json").is_file())
            self.assertTrue((output/"reports"/"eligible_universe.csv").is_file())
            eligible_matrix=pl.read_parquet(output/"gold"/"eligible_simple_total_return_matrix.parquet")
            self.assertEqual(eligible_matrix.columns,["session_date","AAA"])

    def test_identity_review_cannot_enter_eligible_matrix(self):
        record={"asset_class":"equity","identity_verification_state":"PENDING","listing_currency":"USD","listing_market":"NASDAQ"}
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"cut_2020-01-03"
            build_from_raw(raw_payloads={"AAA":json.dumps(payload("AAA")).encode()},records={"AAA":record},cutoff="2020-01-03",output=output,expected_sessions={"AAA":{date(2020,1,2),date(2020,1,3)}},calendar_metadata={"AAA":{"source":"synthetic","version":"test"}})
            matrix=pl.read_parquet(output/"gold"/"eligible_simple_total_return_matrix.parquet")
            self.assertEqual(matrix.columns,["session_date"])

    @patch("scenario_portfolio_lab.capture.requests.get")
    def test_capture_error_does_not_echo_api_key_or_url(self, mocked_get):
        from scenario_portfolio_lab.capture import fetch_daily_adjusted
        import os
        import requests
        os.environ["ALPHAVANTAGE_API_KEY"]="test-secret"
        mocked_get.side_effect=requests.RequestException("upstream request failed")
        with self.assertRaises(ProviderResponseError) as raised: fetch_daily_adjusted("AAA")
        self.assertNotIn("test-secret",str(raised.exception)); self.assertNotIn("https://",str(raised.exception))
        del os.environ["ALPHAVANTAGE_API_KEY"]

    @patch("scenario_portfolio_lab.capture.requests.get")
    def test_capture_preserves_raw_provider_bytes(self, mocked_get):
        from scenario_portfolio_lab.capture import fetch_daily_adjusted
        import os
        response=mocked_get.return_value; response.status_code=200
        raw=json.dumps(payload("AAA"), separators=(",", ":")).encode("utf-8")
        response.content=raw
        os.environ["ALPHAVANTAGE_API_KEY"]="test-secret"
        self.assertEqual(fetch_daily_adjusted("AAA"),raw)
        del os.environ["ALPHAVANTAGE_API_KEY"]

    @patch("scenario_portfolio_lab.capture.requests.get")
    def test_connect_timeout_is_bounded_and_sanitized(self, mocked_get):
        from scenario_portfolio_lab.capture import ProviderRequestError, _provider_response
        import os
        import requests
        os.environ["ALPHAVANTAGE_API_KEY"]="test-secret"; mocked_get.side_effect=requests.ConnectTimeout("secret")
        with self.assertRaises(ProviderRequestError) as raised:
            _provider_response(params={"function":"SYMBOL_SEARCH","keywords":"AAA"},attempts=1,connect_timeout=1,read_timeout=2,max_operation_seconds=3)
        self.assertEqual(raised.exception.telemetry.classification,"CONNECT_TIMEOUT")
        self.assertEqual(mocked_get.call_args.kwargs["timeout"],(1,2))
        self.assertNotIn("secret",str(raised.exception)); del os.environ["ALPHAVANTAGE_API_KEY"]

    def test_verification_resume_skips_only_successful_instruments(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence=Path(temp)/"events.jsonl"
            append_event(evidence,{"instrument_id":"U1","provider_status":"MATCHED"})
            append_event(evidence,{"instrument_id":"U2","provider_status":"ERROR"})
            self.assertEqual(successful_ids(evidence),{"U1"})


if __name__ == "__main__": unittest.main()
