# From Market Listings to a Validated Return Matrix: Building the Data Foundation for Portfolio Research

**Format:** article outline, not a draft  
**Target length:** 1,000–1,400 words  
**Audience:** practitioners and students who want to understand what must be true of historical data before portfolio research starts.

## Opening — 202 → 138 → 125 is a research boundary, not a stock-picking result

- **Question:** What changed between the original catalog and the matrix used for research?
- **Main claim:** The project began with 202 catalog records, closed a verified input universe of 138 instruments, and retained 125 instruments with complete common-window coverage. The first reduction combines identity/provider verification, ETF-structure constraints, structural exclusions, and a defined scope closure; it must not be portrayed as a count of defective market-data records.
- **Evidence:** `docs/universe_snapshots/entry_2026-10-11_cut_2026-10-09_full_window_final/entry_universe_manifest.json` records 138 included, 14 structural exclusions, and 50 out-of-scope pending records. `docs/full_window_coverage_audit_2026-10-11.md` records 125 final data-eligible instruments.
- **Proposed figure/table/example:** A three-step funnel: 202 catalog records → 138 approved input instruments → 125 complete-window instruments, with a caption defining each transition.
- **Required limitation:** The catalog is contemporary, not a historical point-in-time universe; none of these counts is a portfolio recommendation or an economic ranking.

## 1. Define the contract before downloading or aligning anything

- **Question:** Which choices make a historical series comparable to another series?
- **Main claim:** A date window, frequency, quotation currency, adjustment convention, and missing-data policy are part of the experiment definition—not cleanup details added afterward.
- **Evidence:** `docs/data-contract.md` defines 2018-01-02 to 2026-10-09, daily USD analysis, reported OHLC retained separately from `adjusted_close`, and no price or return imputation.
- **Proposed figure/table/example:** A compact “data contract” box with the five decisions and their operational consequences.
- **Required limitation:** USD quotation does not remove the economic FX exposure of issuers or fund holdings; no FX conversion is implemented.

## 2. A ticker is not yet an instrument identity

- **Question:** Why can a familiar ticker still be an unsafe input to a provider request or a portfolio matrix?
- **Main claim:** A ticker coincidence is only a provider-symbol lead. Provider fields can contribute evidence such as an indicated USD currency, but they do not replace separate verification of instrument identity, listing, asset class, or permitted ETF structure.
- **Evidence:** `config/universe.csv`, `docs/public-verification.md`, `docs/etf_verification_2026-10-11.csv`, and `docs/adr_substitution_decisions.md` preserve confirmed records, reviews, exclusions, and undecided ADR/class substitutions.
- **Proposed figure/table/example:** A schematic mapping: catalog reference → verified instrument identity → provider symbol → raw response; add one deliberately ambiguous local-share/ADR example without resolving it.
- **Required limitation:** Provider search success does not itself prove USD quotation, the intended instrument identity, or permitted ETF structure.

## 3. Preserve raw evidence, then make a reproducible Silver representation

- **Question:** What is gained by retaining original provider responses instead of only publishing a final price table?
- **Main claim:** Immutable raw bytes, hashes, and manifests make parsing and later inspection reproducible, while Silver makes the provider schema explicit without erasing reported fields.
- **Evidence:** `docs/capture_manifest_consolidated_2026-10-09.json`, the sealed entry manifest, and the public parser in `src/scenario_portfolio_lab/ingest.py`.
- **Proposed figure/table/example:** A small synthetic Alpha Vantage-like response transformed into two Silver rows, showing reported close beside `adjusted_close`, dividend amount, and split coefficient.
- **Required limitation:** The raw files and downloaded series are intentionally excluded from the public repository; public manifests demonstrate provenance but do not let a reader independently re-run the exact historical cut.

## 4. Gold quality approval is necessary, but not the final admission rule

- **Question:** Why can a Gold-approved series still be absent from the eligible return matrix?
- **Main claim:** Gold checks parseability, economically valid values, unique dates, and a price at the cut; eligibility also requires verified instrument properties and complete coverage of the approved common window.
- **Evidence:** `docs/data-contract.md`, `docs/eligibility.md`, and `docs/full_window_coverage_audit_2026-10-11.md`. The audit reports Gold approval for all 138 captured series but only 125 complete-window eligible series.
- **Proposed figure/table/example:** A two-column comparison of a full-coverage instrument, ABNB (insufficient history), and FUBO (leading plus interior gaps).
- **Required limitation:** A leading gap is coverage evidence, not automatic proof that the instrument had not yet listed; this project did not add dated listing evidence for every short history.

## 5. The 137 → 125 correction: coverage logic changes the experiment

- **Question:** How did the project discover that an apparently valid matrix still violated its stated common-window rule?
- **Main claim:** The earlier rule failed to make leading expected-session gaps disqualifying. Once leading gaps were counted as insufficient history, 12 short-history series and FUBO no longer met full-window eligibility, changing the result from 137 to 125.
- **Evidence:** `docs/full_window_coverage_audit_2026-10-11.md` documents the corrected rule, the 12 `INSUFFICIENT_HISTORY` instruments, and FUBO’s five leading plus two interior missing expected sessions.
- **Proposed figure/table/example:** A before/after rule card: “interior/trailing only” → “leading + interior + trailing”; then show the count change 137 → 125.
- **Required limitation:** This is a correction to the data-admission experiment, not evidence of a better investment result, alpha, or forecast.

## 6. A daily return is defined by consecutive expected sessions

- **Question:** When should two observed prices *not* be converted into a daily return?
- **Main claim:** Simple returns use `adjusted_close[t] / adjusted_close[t-1] - 1` only across adjacent expected sessions. If an expected session is absent, the interval is not labelled a daily return and remains null.
- **Evidence:** `docs/data-contract.md`; public implementation in `src/scenario_portfolio_lab/matrices.py`; final audit reports 2,205 price dates and 2,204 usable returns per eligible instrument.
- **Proposed figure/table/example:** A three-session synthetic timeline where the middle expected session is missing; the last observed price does not receive a two-session “daily” return.
- **Required limitation:** Adjusted prices reflect the provider convention; retaining reported OHLC and corporate-action fields does not independently validate the provider’s adjustment methodology.

## Close — A complete matrix is a starting point, not an investment conclusion

- **Question:** What has—and has not—been established by reaching a 125-instrument aligned return matrix?
- **Main claim:** The data boundary establishes a transparent input for later research: 2,205 price dates, 2,204 usable daily returns per eligible instrument, and an explicitly filtered 125-instrument matrix.
- **Evidence:** `docs/offline_run_2026-10-11.md` and `docs/full_window_coverage_audit_2026-10-11.md`.
- **Proposed figure/table/example:** A final checklist separating “validated data admission” from “still unknown”: stationarity, predictability, alpha, scenarios, probabilities, and portfolio construction.
- **Required limitation:** A complete aligned matrix does not demonstrate stationarity, predictive validity, alpha, or a valid portfolio allocation. The next methodological stage will study scenarios and probabilities; it is not implemented here.
