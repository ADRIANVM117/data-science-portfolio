# Scenario Portfolio Lab

An educational portfolio-construction project built around scenario-based thinking, with a progressive implementation inspired by Anton Vorobets.

**Current status:** the data boundary is implemented and validated with real Alpha Vantage historical data: 138 approved entry instruments and 125 data-eligible instruments for the 2018-01-02 to 2026-10-09 common window. Scenario construction, entropy pooling, and optimization are deliberately still pending.

| Stage | Function | Result |
|---|---|---:|
| Raw | Original responses and provenance | 138 instruments |
| Silver | Parsing and a uniform schema | 138 series |
| Gold | Series-quality controls | 138 series |
| Matrices | Aligned prices and daily returns | Common window evaluated |
| Data-eligible universe | Full coverage and instrument requirements | 125 instruments |

This is not a portfolio selection: no economic ranking, security selection, weights, or allocation exists yet.

## Install and validate

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python -m unittest discover -s tests -v
```

The tests use shareable synthetic fixtures and make no API request. See the [local-raw offline reproduction](docs/offline_run_2026-10-11.md) for the sealed historical run; the raw files are intentionally not published. A new Alpha Vantage acquisition is a separate operation and may differ because the provider can revise historical data.

## Documentation

- [Data contract](docs/data-contract.md)
- [Full-window coverage audit](docs/full_window_coverage_audit_2026-10-11.md)
- [Methodology status](docs/methodology-status.md)

## Essential limitations

The source catalog is contemporary, not a historical point-in-time universe. Downloaded data, generated matrices, credentials, local working instructions, and the source PDF are excluded from the repository. Gold quality approval and data eligibility are separate controls; the latter requires complete common-window coverage.

## Roadmap

1. Scenario specification and views.
2. Entropy pooling and posterior distributions.
3. Robust portfolio construction and diagnostics.
4. Educational notebooks, one stage at a time.



