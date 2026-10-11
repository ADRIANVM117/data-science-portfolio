# Offline data-boundary reproduction

The definitive full-window run used no API request or recapture. It verified the 138 raw SHA-256 values referenced by the sealed entry manifest and materialized a new ignored output directory from those immutable files.

## Results

| Stage | Count |
|---|---:|
| Approved entry universe | 138 |
| Silver parsed | 138 |
| Gold approved | 138 |
| Eligible data universe | 125 |
| Blocked from eligible matrix | 13 (12 insufficient histories; FUBO has leading and interior gaps) |

The diagnostic adjusted-close and return matrices have 2,205 rows, 139 columns (date plus 138 instruments), and range from 2018-01-02 through 2026-10-09. The final eligible return matrix has 2,205 rows and 126 columns (date plus 125 instruments). It contains no non-finite values; each return column has only the first-session null.

The source catalog is contemporary. It must not be represented as a historical point-in-time universe.

## Calendar and environment

The project-local `.venv` uses Python 3.13.2, Polars 1.44.2, Requests 2.32.5, python-dotenv 1.2.3, NumPy 2.5.4, pandas 3.0.6, and exchange_calendars 4.11.3. Expected sessions are from the versioned XNYS session index; this is independent of raw dates. `XNAS` resolves to the same session schedule in this dependency version. Some ETF catalog records retain `United States` rather than a named exchange, so the run records the common US-equity calendar and that listing-field limitation per instrument.

## Reproduce

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts\seal_and_build_offline.py --snapshot docs\universe_snapshots\entry_2026-10-11_cut_2026-10-09 --output data\published\cut_2026-10-09_run_2026-10-11
```

The output directory must be new. The second command will reject changed or missing raw hashes.
