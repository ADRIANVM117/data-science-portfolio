# Historical capture status

The sealed capture at 2026-10-09 contains 138 raw provider responses: 100 equities and 38 ETFs. Raw files are intentionally excluded from Git; [the public consolidated manifest](capture_manifest_consolidated_2026-10-09.json) records their hashes, source batches, and cutoff without secrets or parameterized URLs.

The raw responses were materialized offline. All 138 passed Silver and Gold controls; the full-window eligible data universe contains 125 instruments. See the [coverage audit](full_window_coverage_audit_2026-10-11.md) for the 13 coverage blocks.
