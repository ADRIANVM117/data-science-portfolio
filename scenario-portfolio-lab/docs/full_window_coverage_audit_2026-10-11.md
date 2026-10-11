# Full-window coverage audit: 2026-10-11

The approved common window is 2018-01-02 through 2026-10-09, inclusive: 2,205 US-equity sessions. Eligibility now requires every expected session in that window. A leading gap is `INSUFFICIENT_HISTORY`; it is not automatically described as predating the instrument's listing.

## Reprocessed result

| Stage | Count |
|---|---:|
| Entry universe | 138 |
| Silver / Gold | 138 / 138 |
| Complete-window eligible | 125 |
| Insufficient history | 12 |
| Insufficient history and interior null | 1 (FUBO) |

The 12 `INSUFFICIENT_HISTORY` instruments are ABNB, AFRM, LCID, MRNA, NU, PINS, PLTR, RIVN, SOFI, UBER, UPST, and ZM. Their leading gaps are retained in the diagnostic matrices. The run does not assert that they are pre-listing dates because no dated listing evidence was added in this phase.

FUBO has five leading missing expected sessions, two interior missing expected sessions, no trailing gap, and is blocked as `INSUFFICIENT_HISTORY_AND_INTERIOR_NULL`.

## Gold versus eligibility

Gold means that raw parsed into the required schema, had unique dates and finite/economically valid OHLC and adjusted prices, retained reported corporate-action fields, and had a cutoff price. It does not certify common-window coverage. Eligibility additionally requires verified catalog identity/market/currency, permitted ETF structure, and full coverage. Thus FUBO can correctly be Gold-approved but excluded from the eligible matrix.

## Matrix nulls

The final eligible return matrix has 2,205 rows and 126 columns (date plus 125 instruments). Every return column has exactly one null: the first session's return, for which a prior approved-session price is intentionally unavailable. There are no other return nulls or non-finite values.

## Calendar

With exchange_calendars 4.11.3, XNYS, XNAS and ARCX each contain the same 2,205 sessions for this period. The 31 ETF records that only retain `United States` as their market use this documented common US-equity calendar; this establishes equivalent sessions for the period but does not replace a future exact-exchange metadata cleanup.

The definitive ignored output is `data/published/cut_2026-10-09_run_2026-10-11_full_window_final/`. Its `reports/coverage_report.csv` and `reports/eligible_universe.csv` are the final per-instrument reports for this run.
