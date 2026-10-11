# Public data contract

Period: 2018-01-02 through 2026-10-09. The parser enforces this inclusive range. Frequency: daily. Quotation and analysis currency: USD. No FX conversion is performed.

Raw provider JSON is retained unchanged and hashed. Silver retains reported OHLC separately from `adjusted_close`, volume, dividends, and split coefficient. Gold admits only schema-valid series containing the requested cut-off price.

Simple daily returns are `adjusted_close[t] / adjusted_close[t-1] - 1` only when the two observations are adjacent expected sessions in that instrument's verified market calendar. A missing expected session produces a null return, never a multi-session return labelled daily. No price or return is imputed. Zero volume with valid prices is retained as `VALID_ZERO_VOLUME`; null volume is invalid. Duplicate dates, non-finite/non-positive prices, interior nulls, trailing nulls at the cut, and insufficient common history block eligibility.

The calendar is the verified primary listing market's daily session calendar, recorded with its source/version in each real capture manifest. A date before listing is coverage information, not a data-quality failure.

`simple_total_return_matrix.parquet` is the quality-approved Gold matrix, which may retain non-eligible columns for auditability. `eligible_simple_total_return_matrix.parquet` is its explicit eligibility-filtered derivative; only `ELIGIBLE` instruments may appear in it. Every capture manifest records the approved period, per-instrument calendar metadata, raw hashes, quality status, and eligibility reason. Provider transport errors are normalized and raw/manifests never store request URLs, API keys, or secrets.
