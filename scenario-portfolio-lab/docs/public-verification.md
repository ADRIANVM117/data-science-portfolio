# Public verification status

The project retains the 202 original catalog identities: 146 equities and 56 ETFs. This version seals 138 verified and captured USD instruments as its approved entry universe: 100 equities and 38 direct, unleveraged ETFs.

Fourteen ETFs are structural exclusions (inverse or leveraged). Fifty records remain outside this version's scope; they are retained as `REVIEW` rather than represented as proven exclusions. The three local/ADS mappings remain undecided and are documented in [ADR decisions](adr_substitution_decisions.md).

Provider-symbol evidence, official-source links, catalog states, and the sealed snapshot are retained in `docs/verification_runs/`, `docs/etf_verification_2026-10-11.csv`, and `docs/universe_snapshots/`. A provider match is only one part of verification; it is combined with identity, currency, and ETF-structure evidence where applicable.
