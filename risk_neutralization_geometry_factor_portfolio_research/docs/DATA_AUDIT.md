# Stage 0 — Alpha Vantage data-feasibility audit plan

**Status:** Stage 0 raw acquisition and structural inspection completed on
2026-09-23 UTC. No characteristics, feature matrix, risk matrix, portfolio
weights, projections, or backtests were constructed.

**Scope boundary:** This document specifies only the smallest audit required to
decide whether Alpha Vantage can support a point-in-time raw-data basis for a
future characteristic matrix. It does not define characteristics, derived
features, portfolio weights, risk exposures, or a portfolio rule.

## 1. Current evidence and status

- **CONFIRMED:** Alpha Vantage is the planned primary provider.
- **CONFIRMED:** the methodology requires daily adjusted prices, company
  overview/classification, income statement, balance sheet, cash flow,
  earnings, and shares outstanding.
- **OBSERVED (local environment):** `ALPHAVANTAGE_API_KEY` was loaded from the
  git-ignored project-root `.env` and checked only for non-empty presence. Its
  value was neither displayed nor recorded.
- **OBSERVED:** `scripts/acquire_stage_0_alphavantage.py` defines exactly the
  five frozen symbols and seven approved endpoints. It reads only
  `ALPHAVANTAGE_API_KEY` by default, writes each response byte-for-byte using
  exclusive file creation, and records credential-free request metadata and a
  SHA-256 content hash in `data/raw/stage_0/manifest.json`.
- **OBSERVED:** the manifest contains 70 request records: 35 restricted-network
  attempts with `NETWORK_ERROR` and no payload, followed by 35 successful
  `HTTP_200` requests. The latter are the complete audit sample: one response
  for each of the 5 symbols x 7 approved endpoints. All 35 preserved payloads
  have a non-null path and SHA-256 checksum; no raw payload was overwritten.
- **PENDING:** endpoint access under the project's Premium plan, response
  schemas, historical depth, missingness, and timestamp semantics for every
  sample issuer.
- **PENDING:** whether Alpha Vantage alone provides an availability timestamp
  for each statement line item sufficient for point-in-time use.

The Alpha Vantage documentation states that `OVERVIEW`, the three financial
statements, and `SHARES_OUTSTANDING` are generally refreshed on the day a
company reports its latest earnings/financials. That is a provider refresh
description, not evidence of an observation-level publication timestamp. It
must not be substituted for one. [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)

## 2. Minimal heterogeneous sample

The audit sample is deliberately not a proposed investment universe.

| Symbol | Issuer | Intended heterogeneity to test | Status |
|---|---|---|---|
| AAPL | Apple Inc. | large consumer technology/hardware; non-calendar fiscal year | PENDING symbol and coverage validation |
| JPM | JPMorgan Chase & Co. | bank; financial-statement presentation unlike an industrial issuer | PENDING symbol and coverage validation |
| XOM | Exxon Mobil Corp. | energy issuer with commodity-linked business | PENDING symbol and coverage validation |
| UNH | UnitedHealth Group Inc. | managed-care/healthcare issuer | PENDING symbol and coverage validation |
| WMT | Walmart Inc. | retail issuer with a non-calendar fiscal year | PENDING symbol and coverage validation |

Before any substantive interpretation, validate each requested symbol, US
listing/exchange, currency, and issuer identity from the raw provider response.

## 3. Endpoints, questions, and raw fields to inspect

For every endpoint, save the unmodified response, capture the request function
and non-secret parameters, acquisition time in UTC, HTTP status, content type,
and a SHA-256 checksum. Never record the API key.

| Endpoint | Why inspect it | Questions that the audit must answer | Raw fields/schema to record (not characteristic definitions) | Point-in-time test |
|---|---|---|---|---|
| `TIME_SERIES_DAILY_ADJUSTED` with `outputsize=full` | Daily market history and corporate-action context. | Does the response cover enough trading history for all five issuers? Are dates unique, ordered, and consistently populated? Are adjusted close, dividends, and split coefficients present and parseable? Is the returned date a trading/session date only? | Metadata keys plus every time-series column, including raw OHLCV, adjusted close, dividend amount, and split coefficient. | Distinguish the daily bar date from provider retrieval/refresh time. **PENDING** whether the documented endpoint supplies an intraday availability timestamp for historical bars. |
| `OVERVIEW` | Identity, current classification, and current summary metrics. | Are identity/classification fields populated and internally consistent with the selected issuer? Which fields are absent or non-numeric? Does the response expose dated historical snapshots or only a current snapshot? | Entire top-level key list; explicitly inventory identity/classification, `FiscalYearEnd`, `LatestQuarter`, `SharesOutstanding`, and all ratio/metric fields actually returned. | Treat as current/as-of retrieval snapshot unless the raw response proves otherwise. It cannot populate historical observations without dated snapshots. **PENDING.** |
| `INCOME_STATEMENT` | Annual and quarterly standardized statement history. | Are annual and quarterly arrays present? What is the earliest/latest `fiscalDateEnding`? Which reported currency and line-item fields are populated, omitted, or encoded as strings? Does each observation include a publication/filing/availability timestamp? | Array names, key set per array, `fiscalDateEnding`, `reportedCurrency`, and all raw line-item names actually returned. | Test separately for accounting-period date and any availability-related date. If no observation-level availability date exists, record a point-in-time blocker; do not infer it from refresh behavior. |
| `BALANCE_SHEET` | Annual and quarterly standardized balance-sheet history. | Same completeness, depth, schema stability, and timestamp questions as the income statement; do banking and non-banking schemas materially differ? | Array names, key set per array, `fiscalDateEnding`, `reportedCurrency`, and all raw line-item names actually returned. | Same test. A period-end balance is not evidence that it was knowable on that date. |
| `CASH_FLOW` | Annual and quarterly standardized cash-flow history. | Same completeness, depth, schema stability, and timestamp questions as the other statements; are non-cash/revision-related fields consistently available? | Array names, key set per array, `fiscalDateEnding`, `reportedCurrency`, and all raw line-item names actually returned. | Same test; no availability date may be manufactured. |
| `EARNINGS` | Earnings history and the only candidate endpoint in the required set expected to expose quarterly report dates. | Are annual and quarterly arrays present? Is `reportedDate` supplied, non-null, parsable, and plausible relative to `fiscalDateEnding` for every quarterly observation? Are EPS/estimate/surprise fields consistently typed? | Array names and all keys actually returned; explicitly record `fiscalDateEnding`, `reportedDate`, `reportedEPS`, `estimatedEPS`, `surprise`, and `surprisePercentage` where present. | Establish only what `reportedDate` demonstrably means for this endpoint. It must not be assumed to date the income, balance-sheet, cash-flow, or share observations. |
| `SHARES_OUTSTANDING` | Historical quarterly basic and diluted share-count coverage. | Does the endpoint return both stated share-count variants for every issuer? What are its exact field names, frequency, date coverage, and missing values? Does it include a report/availability date? | Entire schema and all date/share-count field names exactly as returned; do not pre-label undocumented keys. | Test whether the response distinguishes fiscal period, report, and availability dates. If it does not, record the limitation. |

`TIME_SERIES_DAILY_ADJUSTED` is documented as offering 25+ years of raw daily
OHLCV, adjusted close, and split/dividend history; `outputsize=full` is needed
to test that claim rather than merely seeing the latest 100 observations. The
documentation describes `SHARES_OUTSTANDING` as quarterly basic and diluted
share values, while `EARNINGS` provides annual and quarterly EPS history.
[Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)

## 4. Minimal audit procedure

1. Configure a local API key only through an environment variable. **PENDING:**
   exact variable name and local secret-handling setup. Do not place a key in a
   repository file, notebook, payload filename, or log.
2. Request the seven endpoints above for each of the five symbols (35 raw
   requests). Use `outputsize=full` only for daily adjusted prices. Record the
   actual entitlement/error/rate-limit response rather than presuming access.
3. Preserve responses byte-for-byte and create an inventory of endpoint keys,
   array cardinalities, dates, data types, explicit provider null tokens, and
   missingness. No imputation, forward filling, dropping, ranking, or
   transformation is permitted.
4. For each fundamental observation, place `fiscalDateEnding`, any
   `reportedDate`, any other supplied timestamp, and the local UTC acquisition
   time in separate audit columns. Do not synthesize an availability date.
5. Cross-check only structural relationships: report dates must not precede
   their stated fiscal period without an explicit provider explanation; price
   dates must be trading dates; and classification/identity must match the
   requested issuer. These are data-quality checks, not feature construction.
6. Conclude with a per-endpoint and per-field status: `CONFIRMED`, `OBSERVED`,
   `ASSUMED`, or `PENDING`. A field with a fiscal period but no defensible
   availability timestamp remains `PENDING` for point-in-time use.

## 5. Minimal repository artifacts

No implementation code is required for this plan. When retrieval is authorized,
the minimum auditable artifact set is:

| Path | Purpose | Commit status |
|---|---|---|
| `docs/DATA_AUDIT.md` | This plan, then the completed evidence-based audit and decision. | Commit |
| `data/raw/stage_0/<SYMBOL>/<ENDPOINT>.<json-or-csv>` | Immutable, unmodified response for each of 35 requests. | Keep locally; do not commit without explicit approval. |
| `data/raw/stage_0/manifest.json` | Response inventory: symbol, endpoint, non-secret parameters, UTC acquisition time, HTTP status, content type, byte size, checksum, and filename. | Commit only if it contains no credentials and raw-data policy permits. |
| `.gitignore` | Exclude `.env` and, unless approval is granted, raw audit responses. | **PENDING:** create when retrieval setup begins. |
| `.env.example` | Placeholder-only documentation of the chosen environment-variable name. | **PENDING:** needed only if local retrieval utilities are introduced. |

No `data/interim/` or `data/processed/` artifact is required in Stage 0.

## 6. Evidence required to complete this document

The final `DATA_AUDIT.md` must include, for every endpoint and symbol:

- manifest references and checksums for the preserved raw response;
- exact response schema/field dictionary, raw type or provider null token, and
  missingness count/rate—without replacing missing values;
- annual and quarterly observation counts plus earliest/latest dates where
  applicable;
- separate evidence for period dates, report dates, provider refresh metadata,
  and local retrieval time, with their meanings labeled as observed or pending;
- endpoint-access evidence, including any entitlement, rate-limit, or error
  response;
- an identity/classification verification for all five symbols;
- cross-issuer schema differences, especially JPM versus non-financial issuers;
- a field-level mapping only to *potential* economic families (Value, Quality,
  Growth, Momentum) and no formulas, transformations, scores, or final
  definitions;
- a field-level point-in-time verdict: `CONFIRMED`, `LIMITED`, or `PENDING`,
  with a reason; and
- an overall Go / Conditional Go / No-Go decision for constructing a defensible
  point-in-time raw-data basis, including every unresolved dependency.

## 7. Observed Stage 0 evidence

### 7.1 Evidence inventory and integrity

The 35 successful records are in
`data/raw/stage_0/manifest.json`; each gives the symbol, endpoint, non-secret
parameters, UTC retrieval time, HTTP status, content type, byte size, local
raw path, and SHA-256 checksum. The raw paths use the immutable form
`data/raw/stage_0/<SYMBOL>/<ENDPOINT>__20260923T053258894161Z.json`.
The raw directory remains git-ignored.

All successful responses were `HTTP_200`. HTTP success is only transport
evidence; payload semantics were inspected separately below. No credential
appears in the manifest, path names, source code, or this document.

### 7.2 Identity and current overview snapshots

`OVERVIEW` returned 58 top-level fields for every issuer. Raw identity and
classification fields match the frozen sample: AAPL/NASDAQ/USD/Technology,
JPM/NYSE/USD/Financial Services, XOM/NYSE/USD/Energy,
UNH/NYSE/USD/Healthcare, and WMT/NASDAQ/USD/Consumer Defensive. The observed
fiscal-year-end values are September (AAPL), December (JPM/XOM/UNH), and
January (WMT). The only observed provider null token among these snapshots was
`EBITDA = "None"` for JPM.

The payload has no dated historical snapshots. `LatestQuarter` and
`FiscalYearEnd` are fields in a current retrieval snapshot, not evidence of
historical availability. Therefore `OVERVIEW` is **NOT SUPPORTED** for
historical point-in-time use from this endpoint alone.

### 7.3 Daily adjusted prices

For every issuer, `TIME_SERIES_DAILY_ADJUSTED` returned 6,763 dated rows from
1999-11-01 through 2026-09-22. Each row exposes the same eight raw columns:
open, high, low, close, adjusted close, volume, dividend amount, and split
coefficient. Metadata contains Information, Symbol, Last Refreshed, Output
Size, and Time Zone.

The date keys are trading/session dates. `Last Refreshed` is endpoint metadata
at retrieval, not an observation-level historical availability timestamp. The
endpoint is **PENDING** for precise intraday point-in-time availability; the
raw response supports dated daily bars but does not itself establish when a
given historical bar was available on that date.

### 7.4 Financial statements and earnings

For each issuer, `INCOME_STATEMENT`, `BALANCE_SHEET`, and `CASH_FLOW` expose
`symbol`, `annualReports`, and `quarterlyReports`. Each has 20 annual and 81
quarterly records. The annual histories begin in 2006 or 2007 and end in 2025
or 2026; quarterly histories run from 2006 through 2026, with issuer-specific
fiscal calendars. The schemas are identical across these five issuers,
including JPM; this does not establish economic comparability of the raw
line-items.

Each statement observation includes `fiscalDateEnding` and `reportedCurrency`.
None includes `reportedDate`, filing date, publication date, or another
observation-level availability timestamp. Raw provider null tokens are
materially present: 693--781 values in income statements, 848--988 in balance
sheets, and 1,565--1,674 in cash-flow statements across the 101 records for
each issuer. They remain unmodified.

`EARNINGS` exposes `annualEarnings` and `quarterlyEarnings`, with 31 annual
and 122 quarterly records per issuer. Its quarterly schema includes
`fiscalDateEnding`, `reportedDate`, `reportedEPS`, `estimatedEPS`, `surprise`,
`surprisePercentage`, and `reportTime`. This is evidence of a separate dated
earnings endpoint only. It does **not** date the statement or share-count
observations.

Point-in-time verdicts based on the raw evidence are:

| Raw endpoint | Verdict | Reason |
| --- | --- | --- |
| `INCOME_STATEMENT` | **NOT SUPPORTED** | `fiscalDateEnding` is present, but no observation-level availability date is present. |
| `BALANCE_SHEET` | **NOT SUPPORTED** | Same limitation; period end is not availability. |
| `CASH_FLOW` | **NOT SUPPORTED** | Same limitation; period end is not availability. |
| `EARNINGS` | **PENDING** | `reportedDate` is supplied for quarterly earnings, but its exact provider semantics and its relationship to other fundamental datasets were not established by these payloads. |

### 7.5 Shares outstanding

`SHARES_OUTSTANDING` returned `date`, `shares_outstanding_basic`, and
`shares_outstanding_diluted` for AAPL (74 rows, 2008-06-28 to 2026-06-30), JPM
(67, 2008-06-30 to 2025-03-31), UNH (73, 2008-06-30 to 2026-06-30), and WMT
(67, 2008-07-31 to 2025-04-30), with no observed null tokens. XOM returned
`status = "invalid request"` and an empty `data` array, despite HTTP 200.

The endpoint supplies only `date`; it does not distinguish fiscal period,
report/publication, and availability dates. It is therefore **NOT SUPPORTED**
for point-in-time share observations from the available evidence. XOM share
coverage is additionally unavailable in this sample.

### 7.6 Stage 0 decision and unresolved limitations

**Decision: No-Go for point-in-time fundamental characteristics using Alpha
Vantage alone.** Daily dated price history is observed and adequate for a
separate future decision, but the statement and share endpoints lack evidenced
observation-level availability timestamps. `fiscalDateEnding` has not been,
and must never be, used as an availability date.

Before any later stage, research approval would be required for a separately
audited source of filing/publication timestamps or for an explicit decision to
exclude fundamentals that lack a defensible availability rule. The XOM
`SHARES_OUTSTANDING` invalid response also requires provider investigation.
No such extension is implemented here.

## 8. Proposed Stage 0B — Fundamental Point-in-Time Feasibility Audit (SEC EDGAR)

**Status: completed. SEC raw evidence was acquired and structurally inspected.
No characteristic, formula, market-data join, portfolio object, or Stage 1
activity was performed.**

### 8.1 Objective and frozen sample

Stage 0B tests a narrowly defined proposition:

> Can official SEC EDGAR filings provide auditable evidence for the fiscal
> period, filing identity, and a conservative research-availability convention
> required before later research could define historically defensible Value,
> Quality, or Growth characteristics?

The frozen feasibility sample remains AAPL, JPM, XOM, UNH, and WMT, subject to
SEC issuer-identity verification. It is not a universe definition. No Value,
Quality, or Growth formula is proposed here; the tags below are raw candidate
disclosures only.

### 8.2 Minimum official SEC sources

| Source | Minimum use in Stage 0B | Why it is required |
| --- | --- | --- |
| `https://www.sec.gov/files/company_tickers_exchange.json` | Initial ticker → CIK candidate and exchange/name cross-check. | Tickers are not the primary historical identity; SEC cautions that these association files are periodically updated and not guaranteed for accuracy or scope. |
| `https://data.sec.gov/submissions/CIK##########.json` and every historical file named in `filings.files` | Inventory 10-K, 10-Q, 10-K/A, and 10-Q/A filings, their accession number, `filingDate`, `reportDate`, `acceptanceDateTime`, and primary document. | The current submissions object covers at least one year or 1,000 recent filings; historical files are necessary for complete history. |
| `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | Inspect raw US-GAAP/DEI fact arrays, including fact values, units, periods, forms, filing dates, accession numbers, fiscal-year/period labels, and frames where supplied. | It aggregates company XBRL facts across submissions; its filing identifiers must be reconciled to the submissions history. |
| `https://www.sec.gov/Archives/edgar/data/{CIK-without-leading-zeroes}/{accession-without-dashes}/{accession}.hdr.sgml` and filing index | Verify the source filing header and filing documents for a selected audit subset, especially amendments and duplicate-period cases. | The filing header distinguishes `CONFORMED PERIOD OF REPORT`, `FILED AS OF DATE`, and acceptance timestamp evidence. |

`companyconcept` is not required initially because `companyfacts` contains the
company's standard-taxonomy concepts in one response. It may be used only as a
read-only cross-check if a selected concept needs schema clarification. SEC
describes the submissions and XBRL APIs, their scope, and their near-real-time
update behavior in its [EDGAR API documentation](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
SEC's [EDGAR data-access documentation](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)
is authoritative for ticker/CIK associations, filing indexes, accession paths,
and post-acceptance corrections.

### 8.3 Issuer identity protocol

1. Start with the frozen ticker in `company_tickers_exchange.json`; preserve
   its returned CIK, conformed name, ticker, and exchange as raw evidence.
2. Left-pad the accepted CIK to ten digits for `data.sec.gov` endpoints and
   retain the numeric CIK separately for archive paths.
3. Verify the candidate against the issuer name, ticker/exchange associations,
   and CIK returned by that issuer's `submissions` response.
4. Freeze a ticker–CIK mapping only when those records agree. A renamed ticker,
   multiple CIK association, or identity conflict is a **PENDING** audit result,
   not a mapping to repair heuristically.

### 8.4 Raw filing and XBRL fields to audit

The audit will inventory raw fields rather than define characteristics.

| Record | Required raw fields to inspect |
| --- | --- |
| Submission / filing header | `accessionNumber`, `form`, `filingDate`, `reportDate`, `acceptanceDateTime`, `primaryDocument`, `items`, and header equivalents of accepted date-time, filed-as-of date, and conformed period of report. |
| XBRL fact provenance | taxonomy, tag, unit, `val`, `start`, `end`, `fy`, `fp`, `form`, `filed`, `accn`, and `frame` when present. |
| Candidate standard raw disclosures | `us-gaap:Assets`, `Liabilities`, `StockholdersEquity` or `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`, `RevenueFromContractWithCustomerExcludingAssessedTax` or `SalesRevenueNet`, `NetIncomeLoss`, `OperatingIncomeLoss`, `GrossProfit`, `NetCashProvidedByUsedInOperatingActivities`, `PaymentsToAcquirePropertyPlantAndEquipment`, and DEI shares-outstanding disclosures where present. |

Tag availability, units, and issuer-specific extensions are audit observations.
No fallback hierarchy, ratio, growth calculation, normalization, imputation, or
cross-sectional scoring is authorized.

### 8.5 Temporal semantics and point-in-time discipline

Every candidate fact must retain distinct fields:

| Field | Meaning in the audit | Prohibited inference |
| --- | --- | --- |
| `t_period` | Accounting period represented by XBRL `start`/`end`, `reportDate`, or the header's conformed period of report. | It is not when the market or researcher knew the fact. |
| `t_filed` | Official EDGAR `filingDate` / header `FILED AS OF DATE`. | It is not an intraday public-availability timestamp. |
| `t_accepted` | EDGAR `acceptanceDateTime` / header acceptance evidence, where available. | It is not automatically the first moment content was accessible on SEC.gov. |
| `t_available_research` | A separately governed date-time or conservative effective-session convention used by research. | It may not be filled from `t_period`, inferred from a provider refresh, or silently equated to `t_filed`/`t_accepted`. |

SEC states that its data APIs update as filings are disseminated, but its
developer FAQ also states that there is no timestamp indicating when filing
content first became available on SEC.gov. Therefore, Stage 0B begins with
`t_available_research = PENDING`. A later research convention could, for
example, restrict use to a trading session strictly after a verified filing
date; that would be a conservative research rule, not an observed SEC
availability timestamp, and requires explicit approval before use.

### 8.6 Amendments, duplicates, and revisions

- Include `10-K`, `10-Q`, `10-K/A`, and `10-Q/A` in the filing inventory.
  Classify amendments from the filing form, retain their own accession number,
  and never overwrite the original filing or its facts.
- Detect candidate duplicate facts with the provenance key:
  `CIK + taxonomy + tag + unit + start + end + fy + fp + form + accn`.
  For same-period facts across different accessions, record whether values are
  identical, differ, or have incompatible context/units.
- Detect duplicate or competing filings using at least:
  `CIK + form family + reportDate/conformed period + accessionNumber`, then
  compare `filingDate`, `acceptanceDateTime`, and amendment status.
- A later fact or amendment must not retrospectively replace the fact that was
  available to a historical decision. Any eventual as-of selection rule must
  choose only filings permitted by the then-approved `t_available_research`
  convention and must preserve the rejected/competing provenance.

SEC notes that post-acceptance corrections and deletions can alter later
indexes, so an audit must preserve the retrieved raw filing inventory and its
retrieval timestamp rather than rely on an undated current index alone.

### 8.7 Evidence required for a point-in-time usable fact

A fundamental observation can be labeled **CONFIRMED** for later point-in-time
research only if all of the following are evidenced in preserved SEC records:

1. issuer identity is confirmed by ticker/name/exchange/CIK checks;
2. the fact is linked through `accn` to a preserved 10-K or 10-Q family filing;
3. taxonomy, tag, unit, value, and accounting period are explicit and
   internally coherent;
4. filing-date and, where available, acceptance-date-time evidence is retained
   separately from the fiscal period;
5. competing facts, duplicate periods, and amendments are inventoried rather
   than silently collapsed; and
6. an explicitly approved, non-look-ahead `t_available_research` convention is
   applicable to that filing.

Absent any item, classify the fact **PENDING**. If the SEC record lacks the
necessary underlying field or provenance link, classify it **NOT SUPPORTED**.
Raw facts remain missing where the filing reports no usable value.

### 8.8 Future alignment boundary (not implemented)

If Stage 0B evidence later satisfies the criteria above, a separate approved
stage could align a fact to Alpha Vantage daily data only through explicit
keys: CIK/ticker identity, accession, `t_period`, `t_filed`, `t_accepted`,
`t_available_research`, and an Alpha Vantage trading-session date. The join
must select only sessions on or after the approved research-availability rule;
it must never backdate a fact to its fiscal period end. No such join, market
data retrieval, or integration is part of Stage 0B.

### 8.9 Stage 0B stop condition

Stage 0B ends with a documented feasibility decision only:

- **Go:** official SEC evidence and an approved conservative availability rule
  support a bounded set of raw facts;
- **Conditional Go:** some facts are usable and all exclusions are explicit;
- **No-Go:** filing/provenance/availability evidence remains insufficient.

Any outcome stops before Stage 1. Characteristic definitions, `X_char`,
portfolio methodology, risk exposures, and portfolio construction remain out
of scope.

### 8.10 Observed SEC acquisition and structural evidence

The acquisition utility used a local, non-empty `SEC_EDGAR_USER_AGENT` and a
0.25-second inter-request pause. The User-Agent string is not stored in raw
paths, source output, or the manifest. Two overlapping acquisition attempts
completed while the execution environment was resolving a long-running command.
The immutable manifest preserves 215 successful `HTTP_200` requests plus one
explicit identity-resolution record; their content hashes reduce to 110 unique
payloads used for the findings below. The complete successful unique evidence
set consists of:

| Endpoint | Unique preserved payloads |
| --- | ---: |
| `company_tickers_exchange` | 1 |
| `submissions` | 6 |
| `submissions_history` | 77 |
| `companyfacts` | 6 |
| selected `filing_header` | 20 |

Every manifest-backed raw path and SHA-256 checksum was preserved. The audit
uses one payload per unique checksum only; duplicates are retained as
acquisition evidence and are not silently deleted.

#### Identity

SEC ticker-source and submissions evidence agree for AAPL/CIK 0000320193
(Apple Inc.), JPM/CIK 0000019617 (JPMORGAN CHASE & CO),
UNH/CIK 0000731766 (UNITEDHEALTH GROUP INC), and WMT/CIK 0000104169
(Walmart Inc.).

The current SEC ticker association for `XOM` returns CIK 0002115436,
`ExxonMobil Holdings Corp`, a distinct entity whose submissions response has
only 30 recent filings, no historical submissions files, and one in-scope
10-K/10-Q-family filing. SEC [filing detail](https://www.sec.gov/Archives/edgar/data/34088/000119312526226496/0001193125-26-226496-index.htm)
and the registrant's [10-K](https://www.sec.gov/Archives/edgar/data/34088/000003408826000045/xom-20251231.htm)
identify the intended frozen issuer, Exxon Mobil Corporation / EXXON MOBIL
CORP, as CIK 0000034088. Its historical filings also show Common Stock ticker
`XOM` on NYSE. This is an unambiguous identity resolution, not a silent substitution:
the manifest records both the current ticker-source CIK and an explicit
`XOM=0000034088` override used only for this additional audit acquisition.

The evidence supports the limited conclusion that the current SEC ticker file
is a current association, not a historical ticker-to-issuer map. It did not
explain the corporate event behind the reuse/association, and this audit makes
no inference about it.

#### Filing time evidence

For the four identity-confirmed issuers, the complete submission inventory
contains the following 10-K/10-Q-family records:

| Issuer | In-scope filings | Amendments | Same-period competing filing groups | Missing `acceptanceDateTime` |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 132 | 4 | 3 | 0 |
| JPM | 137 | 12 | 7 | 0 |
| UNH | 132 | 8 | 8 | 0 |
| WMT | 132 | 5 | 5 | 0 |
| XOM (resolved CIK 0000034088) | 132 | 3 | 3 | 0 |

The selected 17 raw filing headers all contain `<ACCEPTANCE-DATETIME>`,
`<FILING-DATE>`, and `<PERIOD>`. These are observed as distinct header fields.
They correspond respectively to EDGAR acceptance time, filing date, and the
reporting period; none is treated as the other.

#### Company-facts provenance and duplicates

Across the raw candidate standard tags, the audit observed the following
in-scope fact counts and distinct accession provenance groups:

| Issuer | Candidate tags present | 10-K/10-Q-family facts | Distinct `(accn, filed, form)` groups | Duplicate period keys | Facts without submissions accession link |
| --- | ---: | ---: | ---: | ---: | ---: |
| AAPL | 10 | 1,939 | 70 | 23 | 0 |
| JPM | 6 | 970 | 70 | 13 | 0 |
| UNH | 8 | 1,408 | 69 | 0 | 0 |
| WMT | 9 | 1,855 | 70 | 25 | 0 |
| XOM (resolved CIK 0000034088) | 8 | 1,275 | 69 | 19 | 0 |

The duplicate-period count uses the raw key
`tag + unit + start + end + fy + fp`; it is a flag for inspection, not a
deduplication rule. Every audited candidate fact has an `accn` that links to a
preserved submissions record. `companyfacts` supplies fact-level `filed` and
accession provenance, while the submissions/header evidence supplies the
separate acceptance timestamp. Amendments and competing same-period filings
remain distinct raw records.

Candidate standard-tag coverage differs materially by issuer. For example,
JPM exposes six of the inspected standard candidates. This is an availability
observation, not a formula or a reason to fill missing tags.

#### Availability limitation and daily-frequency candidates

The SEC documentation remains decisive: there is no timestamp identifying the
exact time filing content first became available on SEC.gov. Consequently,
neither `filingDate` nor `acceptanceDateTime` is labeled dissemination/public
availability, and `t_available_research` remains **PENDING**.

Nevertheless, the observed provenance can support review of conservative
daily-frequency conventions that avoid backdating facts to the reporting
period. The following are candidates only; none is adopted or implemented:

| Candidate research convention | Evidence used | Assumption and limitation |
| --- | --- | --- |
| First eligible trading session strictly after `filingDate` | Official filing date, accession, and report period. | Conservative at daily resolution, but assumes public dissemination by the subsequent eligible session. It does not identify the true publication time. |
| First eligible trading session after `acceptanceDateTime` | Exact EDGAR acceptance datetime linked through accession. | More temporally granular, but acceptance is not an observed SEC.gov content-availability timestamp; an after-hours/session-calendar rule remains necessary. |

Both candidates would require a separately approved market-session calendar,
explicit treatment of weekends/holidays, an amendment selection policy, and a
frozen as-of rule before any factual use or market-data alignment. None
authorizes a join to Alpha Vantage data now.

### 8.11 Acceptance-time daily-frequency convention evaluation

The SEC `submissions` evidence has two raw timestamp encodings: current records
use ISO-8601 UTC timestamps (for example, with a `Z` suffix), while historical
submission records use EDGAR's compact acceptance-time form. For audit-only
clock classification, the former was converted to US Eastern clock time and
the latter retained as the header-consistent Eastern clock representation. The
raw values were not replaced.

| Issuer | Pre-open acceptances | Regular-hours acceptances | After-close acceptances | Weekend acceptances | Fixed-date holiday acceptances* |
| --- | ---: | ---: | ---: | ---: | ---: |
| AAPL | 51 | 0 | 81 | 0 | 0 |
| JPM | 36 | 19 | 82 | 0 | 0 |
| UNH | 37 | 6 | 89 | 0 | 0 |
| WMT | 33 | 15 | 84 | 0 | 0 |
| XOM (resolved) | 33 | 78 | 21 | 0 | 0 |

\*The audit checked only New Year's Day, Independence Day, and Christmas by
calendar date. It did not construct or join an exchange calendar; movable and
market-specific holidays remain an implementation prerequisite.

The tested candidate is stated precisely as a **daily research convention**:

> `t_available_research` is the opening of the first *full eligible regular
> trading session whose opening occurs strictly after t_accepted`.

This means a pre-open acceptance may be eligible at that same day's regular
open; an acceptance during regular hours or after close waits until the next
eligible regular session. A weekend or market-holiday acceptance waits until
the next eligible session. This convention does not use a fact during the
daily session already in progress, avoiding use of that day's complete bar.

For amendments and multiple filings for the same period, each filing retains
its own `accn` and `t_accepted`. A later amendment becomes eligible only under
its own later cutoff; it never rewrites the earlier as-of information set.
Where multiple filings have the same fiscal period, the convention orders them
by acceptance datetime, while period dates alone cannot do so.

Compared with `first eligible session after filingDate`, acceptance-based
timing gains intraday ordering: it distinguishes pre-open, in-session, and
after-close filings on the same filing date, and orders amendments or multiple
same-day accessions. A filing-date-only rule loses that ordering and must treat
all same-date filings conservatively as one batch. Under a deliberately
next-session-only daily policy, the resulting effective dates can often agree,
but the acceptance-based provenance remains more informative and auditable.

The key limitation remains explicit: `t_accepted` is not an observed
dissemination/public-availability time. SEC states that it does not provide a
timestamp for when filing content first becomes available on SEC.gov in its
[developer FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions).
The convention assumes that a filing accepted by EDGAR is publicly usable no
later than its calculated next full eligible session. It is a conservative
research timing policy, not a claim about the exact SEC.gov publication instant.

### 8.12 Revised Stage 0B verdict

**Conditional Go — all five frozen issuers now have an auditable SEC identity
and filing/fact provenance chain, subject to the limitations below.** The
recommended convention to freeze for Experiment 03 is the acceptance-based
daily convention in Section 8.11, with its exact wording and assumptions
preserved. It is preferable to filing-date-only timing because it preserves
auditable intraday ordering while remaining conservative for daily research.

Freezing this convention does **not** assert that exact public dissemination
time is known. It requires a future implementation to use a reproducible
regular-session calendar, to retain raw timestamp encodings and provenance,
and to apply a deterministic amendment/competing-filing selection rule. Those
are governance requirements, not authorization to define a characteristic or
join data.

No formula, `X_char`, Alpha Vantage join, portfolio, or Stage 1 work follows
from this verdict.

## 9. Stage 0A decision rule and resolved status

**Conditional Go** is possible only if daily prices have adequate history and
the raw fundamental fields needed for a later research decision are present,
typed, and traceable to an observed availability timestamp (or are explicitly
excluded from point-in-time use). **No-Go for point-in-time fundamentals** is
required if availability timestamps for statement and share observations cannot
be evidenced from Alpha Vantage responses or provider documentation. In that
case, document the gap and propose, but do not implement, a separately approved
source for filing/publication timestamps.

**Resolved Stage 0A decision: No-Go for point-in-time fundamentals using Alpha
Vantage alone.** API evidence was collected and is summarized in Section 7.
Stage 0B is proposed solely to audit SEC EDGAR as a complementary source; it
does not authorize Stage 1.

## 10. Stage 1A dependency evidence (no characteristic construction)

**Status:** prerequisite documentation only. This section does not change the
Stage 0B verdict, select a characteristic, or join any SEC fact to market data.

- **RESOLVED:** the preserved SEC `companyfacts` responses contain
  `dei:EntityCommonStockSharesOutstanding` facts for AAPL (70), JPM (73), XOM
  (69), UNH (69), and WMT (70). Every inspected fact has `end`, `val`, `accn`,
  and `filed`; `accn` links it to the preserved filing inventory. These are
  discrete, filing-linked shares observations, not daily series and not an
  observed dissemination time.
- **NOT SUPPORTED:** Alpha Vantage `OVERVIEW.SharesOutstanding`, Alpha Vantage
  shares records with only `date`, and weighted-average EPS shares cannot be
  substituted for historical point-in-time shares outstanding.
- **DECISION REQUIRED:** an eventual market-equity candidate must state whether
  it uses the latest eligible reported SEC shares observation between filings,
  how it limits or measures staleness, and how amendments/competing filings are
  selected without replacing earlier as-of information. This audit does not
  make that decision.
- **DECISION REQUIRED:** the already recommended acceptance-based convention
  needs a versioned regular-session calendar and an explicit early-close rule;
  the official NYSE calendar is the authoritative schedule reference. No
  calendar or market-data implementation was added in Stage 1A.

## 11. LISTING_STATUS security-eligibility feasibility supplement

**Status:** raw-only audit completed 2026-09-27. This supplement evaluates only whether Alpha Vantage `LISTING_STATUS` can support a future Security Eligibility layer. It does not define a universe, a reconstitution rule, thresholds, characteristics, weights, or any Stage 2 object.

### 11.1 Preserved requests and observed schema

The immutable manifest at `data/raw/listing_status_audit/manifest.json` records eight successful `HTTP_200` requests, plus eight earlier `NETWORK_ERROR` attempts with no payload. Each successful raw response has a local path, retrieval timestamp, request parameters, byte size, and SHA-256 hash; no credential is recorded. The four requested dates were `2010-01-04`, `2014-07-10`, `2020-08-31`, and `2024-01-02`, each queried with `state=active` and `state=delisted`.

Every successful payload is a CSV-formatted response (provider content type: `application/x-download`) with exactly these observed columns:

```text
symbol, name, exchange, assetType, ipoDate, delistingDate, status
```

| Date | State | Rows | Unique `symbol` | `assetType` counts | Observed exchanges |
| --- | --- | ---: | ---: | --- | --- |
| 2010-01-04 | active | 5,329 | 5,302 | Stock 4,091; ETF 1,238 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |
| 2010-01-04 | delisted | 47 | 47 | Stock 47 | NYSE, NASDAQ |
| 2014-07-10 | active | 7,098 | 7,057 | Stock 5,017; ETF 2,081 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |
| 2014-07-10 | delisted | 429 | 429 | Stock 423; ETF 6 | NYSE, NASDAQ |
| 2020-08-31 | active | 8,788 | 8,715 | Stock 5,955; ETF 2,833 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |
| 2020-08-31 | delisted | 4,236 | 4,234 | Stock 3,292; ETF 944 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |
| 2024-01-02 | active | 10,586 | 10,528 | Stock 6,777; ETF 3,809 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |
| 2024-01-02 | delisted | 8,091 | 7,943 | Stock 6,534; ETF 1,557 | NYSE, NASDAQ, NYSE MKT, NYSE ARCA, BATS |

The official provider documentation states that the endpoint returns active or delisted US stocks and ETFs for the latest trading day or a specified historical date, and that specified dates after 2010-01-01 are supported. It does not, in the documentation inspected for this audit, define a permanent security identifier or a common-stock-only classification. [Alpha Vantage API documentation](https://www.alphavantage.co/documentation/)

### 11.2 What the evidence supports

- **OBSERVED:** `state=active` responses contain only `status=Active`; their `delistingDate` field is the literal provider token `null`. `state=delisted` responses contain only `status=Delisted` and populated date strings in the sampled payloads.
- **OBSERVED:** response sizes and membership differ across requested dates, consistent with the documented historical-date parameter. This is evidence of provider response behavior, not independent proof that the response is the complete set known or tradable on that date.
- **OBSERVED:** `ipoDate` and `delistingDate` are available as provider-labelled date fields. Their exact economic semantics (for example, original listing, exchange listing, or another provider convention) are **UNVERIFIED** by the raw payload alone.
- **OBSERVED sanity check:** the familiar common-equity/ETF contrast is represented as expected in all four active snapshots: `AAPL` is `Stock` and `SPY` is `ETF`. This demonstrates only the coarse Stock-versus-ETF distinction, not common-stock purity within `Stock`.
- **OBSERVED:** active and delisted responses may share a ticker on the same requested date. There were 2, 111, and 381 overlapping ticker strings on 2014-07-10, 2020-08-31, and 2024-01-02 respectively. Examples include `ACB` (Aurora Cannabis Inc active versus ACap Energy Ltd delisted) and `ACI` (Albertsons Companies Inc - Class A active versus Arch Coal Inc delisted) in the 2024-01-02 responses.
- **OBSERVED:** 346 `(date, state, symbol)` groups contained more than one row. For example, the 2010-01-04 active response has two `ARIS` rows with different names/exchanges, and two `BNY` rows with different `assetType` values.

### 11.3 What the evidence does not support

`assetType` has only the observed values `Stock` and `ETF`. It does separate many ETFs from `Stock`, but it is not granular enough for the intended common-equity eligibility rule. The `Stock` label is attached in the raw responses to examples named as units and warrants, including `ACACU` (units) and `ACHR-WS` (warrants), and also appears for foreign-labelled securities. Therefore the endpoint cannot, on observed evidence, distinguish US common stock from all preferred securities, warrants, units, ADRs, foreign listings, or other non-common instruments.

No observed field is a permanent issuer or security identifier: the schema has no CIK, ISIN, FIGI, or provider security ID. `symbol` cannot be used as such an identifier because it is duplicated and reused. Consequently, the endpoint cannot by itself establish ticker-change continuity, distinguish a reused ticker from its predecessor, or safely combine active and delisted records into one deduplicated historical security population.

The endpoint documents historical queries only after 2010-01-01. Neither the documentation nor these responses establish historical adjustment vintages, retrieval-time knowledge, exchange-level tradability during an entire session, halt status, or completeness beyond the provider's stated US stocks/ETFs scope. Those claims remain **UNVERIFIED**.

### 11.4 Security-eligibility verdict

`LISTING_STATUS` is useful raw evidence for a later audit of provider-labelled active/delisted stock-or-ETF listings from 2010 onward. It can contribute candidate ticker/date records and provider-labelled lifecycle dates.

It cannot resolve the required conjunction

\[
\text{historical existence}
\;+\;
\text{common-security type}
\;+\;
\text{listing/delisting continuity}
\;+\;
\text{persistent identity}.
\]

**Verdict: `LISTING_STATUS — NO-GO` for Security Eligibility as the sole source.** A further source would need a historical, persistent security-level identifier and time-versioned security-type and listing-status coverage. This verdict does not select that source or begin Stage 2.

## 12. OpenFIGI complement audit for Security Eligibility

**Status:** raw-only complement audit completed 2026-09-27. This section tests whether free OpenFIGI mapping can complement preserved historical Alpha Vantage `LISTING_STATUS` rows. It does not replace Alpha Vantage, define a universe, select thresholds, or begin Stage 2.

### 12.1 Free access, raw evidence, and observed response contract

OpenFIGI documentation describes its mapping API as free and open to the public, with a lower unauthenticated rate limit. The documented unauthenticated Mapping API allowance is 25 requests per minute; the documented unauthenticated mapping-request limit is five jobs. [OpenFIGI API documentation](https://www.openfigi.com/api/documentation)

Two unauthenticated, credential-free `POST /v3/mapping` requests of five jobs each returned `HTTP_200`. The immutable manifest at `data/raw/openfigi_complement_audit/manifest.json` retains both successful response paths, jobs, retrieval timestamps, SHA-256 hashes, and byte sizes, plus two earlier network-error attempts without payloads. No OpenFIGI API key was sent, stored, or required for these responses.

The observed success-record fields are `figi`, `name`, `ticker`, `exchCode`, `compositeFIGI`, `securityType`, `marketSector`, `shareClassFIGI`, `securityType2`, and `securityDescription`. A no-match job returns `warning` instead of `data`. The provider documentation describes `FIGI` as an individual-instrument identifier that does not change once issued; `compositeFIGI` and `shareClassFIGI` express different aggregation levels. [OpenFIGI API documentation](https://www.openfigi.com/api/documentation/)

### 12.2 AV sample mappings and classification evidence

The requests used only ticker values preserved in the Alpha Vantage audit and OpenFIGI's provider exchange code `US`. The AV exchange labels NYSE/NASDAQ were not silently converted into a more granular OpenFIGI venue mapping.

| AV audit case | OpenFIGI request/result | Classification evidence | Audit implication |
| --- | --- | --- | --- |
| AAPL, active common-stock control | One result: `BBG000B9XRY4` | `securityType=Common Stock`, `securityType2=Common Stock`, `marketSector=Equity` | Common-stock classification is demonstrated for this match. |
| SPY, active ETF control | One result: `BBG000BDTBL9` | `securityType=ETP`, `securityType2=Mutual Fund`, `marketSector=Equity` | ETF-like vehicle is distinct from AAPL's Common Stock result. |
| ACACU, AV `Stock` unit | No result in ordinary mapping; one result with `includeUnlistedEquities=true`: `BBG0160DYNQ0` | `securityType=Unit`, `securityType2=Unit` | OpenFIGI can classify this matched unit; its availability depends on query options. |
| ACHR-WS, AV `Stock` warrant | `warning: No identifier found` in both tests | No OpenFIGI classification returned | A free ticker mapping cannot prove the instrument type or eligibility for this case. |
| CNH, active and delisted ticker collision | Ordinary mapping returns only CNH Industrial `BBG0059JSF49`; include-unlisted returns that FIGI plus CNH Global `BBG000BBG2C7` | Both are Common Stock; response has no date interval | Multiple FIGIs are returned but cannot be assigned to the AV row's historical date without an external/as-of rule. |
| ACB / ACI reused tickers | Include-unlisted returns Aurora Cannabis / Albertsons respectively | Both current/common results; no ACap Energy or Arch Coal result | Historical predecessor mapping remains unresolved. |

### 12.3 Historical mapping result

OpenFIGI's documented Mapping Job has identifiers, exchange, currency, market-sector, security-type, and unlisted-equity filters, but no historical date or as-of parameter. The observed response fields also contain no listing date, delisting date, effective-from date, effective-to date, or mapping-vintage timestamp.

This is decisive for the critical mapping input

\[
(\text{symbol},\ \text{exchange},\ \text{historical date}).
\]

`symbol + exchange` can return a current match, no match, or multiple FIGIs. The CNH result demonstrates multiplicity; ACB and ACI demonstrate that a returned current FIGI does not identify the delisted predecessor represented by an AV row. `includeUnlistedEquities=true` can expose some additional FIGIs, but it does not supply a date rule for choosing among them. Thus a successful current mapping is not evidence that the returned FIGI was the security existing on the AV historical date.

### 12.4 Combined feasibility

| Requirement | AV contribution | OpenFIGI contribution | Combined result | Evidence | Limitation |
| --- | --- | --- | --- | --- | --- |
| Historical existence | Provider-labelled active/delisted rows from 2010 onward | No as-of mapping date | **UNVERIFIED** | AV date responses differ; OpenFIGI lacks date fields | Neither source proves a complete historical set known/tradable at the date. |
| Common-security classification | Coarse `Stock` / `ETF` only | Demonstrated `Common Stock`, `ETP`, and `Unit` where a FIGI is found | **CONDITIONAL / incomplete** | AAPL, SPY, ACACU matches | ACHR-WS has no match; no rule covers all AV `Stock` rows or all excluded classes. |
| Listing/delisting | Provider-labelled `ipoDate`, `delistingDate`, `status` | No observed lifecycle dates | **CONDITIONAL / provider-labelled only** | AV raw CSV; OpenFIGI mapping output | Date semantics and effective tradability remain unverified; OpenFIGI adds no lifecycle chronology. |
| Identity continuity | No persistent ID; ticker duplication/reuse observed | Persistent FIGI only after a correct instrument is selected | **NOT SUPPORTED** | CNH multiple FIGIs; ACB/ACI predecessor failures | No historical/as-of bridge selects the correct FIGI across ticker reuse, changes, or delistings. |

OpenFIGI materially improves classification for records it maps, and its FIGI hierarchy is useful after identity is established. It does not establish that identity for the historical AV record. Therefore the free-source conjunction does not resolve historical existence plus common-security type plus listing/delisting plus identity continuity. No third source was researched or selected.

**Verdict: `AV LISTING_STATUS + OpenFIGI — NO-GO`.**

## 13. Alpha Vantage provider-defined stock candidate-set contamination audit

**Status:** read-only contamination audit completed 2026-09-27. The candidate object assessed here is `Candidate_t = LISTING_STATUS(date=t, state=active)` restricted to `assetType=Stock`. It is deliberately not called a historical US common-stock universe. This section does not construct `U_t`, choose size or liquidity thresholds, or begin Stage 2.

### 13.1 Time sample and conservative classification protocol

The four already-preserved active snapshots are sufficient for this limited audit: `2010-01-04` is near the provider's documented historical lower bound; `2014-07-10` and `2020-08-31` provide intermediate observations; and `2024-01-02` is the latest preserved reconstitution-like date. No daily snapshots and no new provider calls were required.

The audit uses only the observed Alpha Vantage fields. A row is **confirmed non-common** only when its provider `name` contains an explicit instrument descriptor under a narrow deterministic rule: `Warrant`, `Units` (plural) or `Unit` followed by a quantity/parenthesis, `Contingent Value Right` or `Tradeable Right`, `Preferred`, `Depositary`, `ADR`/`ADS`/`American Depositary`, `ETF`, `ETN`, or `Closed-End Fund`. The rule is intentionally a lower bound. It does not treat a ticker suffix as conclusive, and it does not classify generic words such as `Trust`, `Fund`, `Foreign`, or `Acquisition Corp` as non-common. Those remain **UNKNOWN**.

`apparently_eligible` below means only that a Stock-labelled row survives those observed lower-bound exclusions. Because no positive common-stock field exists, every apparently eligible row is also `UNKNOWN`; it is not a confirmed common operating equity. The contamination lower bound is `confirmed_non_common / stock_labelled`; the unknown rate is `unknown / stock_labelled`.

### 13.2 Temporal contamination table

| Date | active_total | stock_labelled | ETF_labelled | symbol_duplicate_groups | exact_duplicate_extra_rows | confirmed_non_common | unknown | apparently_eligible | contamination_rate_lower_bound | unknown_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010-01-04 | 5,329 | 4,091 | 1,238 | 27 | 1 | 87 | 4,004 | 4,004 | 2.13% | 97.87% |
| 2014-07-10 | 7,098 | 5,017 | 2,081 | 41 | 1 | 147 | 4,870 | 4,870 | 2.93% | 97.07% |
| 2020-08-31 | 8,788 | 5,955 | 2,833 | 72 | 2 | 377 | 5,578 | 5,578 | 6.33% | 93.67% |
| 2024-01-02 | 10,586 | 6,777 | 3,809 | 57 | 1 | 691 | 6,086 | 6,086 | 10.20% | 89.80% |

All four snapshots expose the same exchanges: BATS, NASDAQ, NYSE, NYSE ARCA, and NYSE MKT. `symbol_duplicate_groups` counts repeated ticker strings within a single active response; `exact_duplicate_extra_rows` is narrower and counts byte-equivalent field tuples beyond their first occurrence. Neither count is resolved into a deduplicated security population.

### 13.3 Observed contaminant taxonomy

The following counts are non-exclusive because one provider name can contain more than one explicit descriptor. The row-level `confirmed_non_common` total in Section 13.2 deduplicates those overlaps.

| Explicit provider-name descriptor within `assetType=Stock` | 2010 | 2014 | 2020 | 2024 | Example observed in raw evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Warrant | 6 | 20 | 187 | 452 | `AIG-WS` — “... Warrants ...” |
| Unit (narrow rule) | 0 | 6 | 163 | 195 | `PACQU` — “... Units (1 Cls A Ord & 1/2 War)” |
| Contingent/tradeable right | 0 | 2 | 2 | 2 | `GCVRZ` — “Contingent Value Rights” |
| Preferred | 30 | 48 | 19 | 24 | `AES-P-C` — “Class C Preferred Stock” |
| Depositary | 11 | 20 | 1 | 9 | `AF-P-C` — “Depositary Shs ...” |
| ADR / ADS | 29 | 38 | 3 | 6 | `ASMI` — “(ADR)” |
| ETF / ETN named as Stock | 13 | 18 | 14 | 15 | `ARIA` — “... ETF” |
| Closed-End Fund exact descriptor | 0 | 0 | 0 | 0 | No exact phrase observed |

The increase in the lower bound is material: direct, provider-name-labelled non-common rows grow from 87 to 691, and the lower-bound rate rises from 2.13% to 10.20%. The endpoint's `assetType=Stock` label is therefore empirically heterogeneous, not merely contaminated by an isolated edge case.

### 13.4 Unknown classes and supported exclusions

The following observable patterns were measured but are not exclusions: SPAC-like names (`Acquisition Corp`, `Acquisition Company`, or `Blank Check`) occur 4, 18, 74, and 317 times across the four dates; fund-like names occur 1, 1, 31, and 62 times; trust-like names occur 92, 121, 167, and 167 times; foreign-labelled names occur 7, 3, 0, and 0 times. They remain `UNKNOWN`. For example, `Trust` can describe a REIT common equity as well as a non-operating vehicle, and an Acquisition Corp label alone does not determine the security class.

Supported lower-bound exclusions are limited to the explicit provider-name descriptors in Section 13.3. Their false-negative rate is unknown; their output must not be described as confirmed common stock. A naive singular `Unit` pattern was rejected during the audit because it would falsely classify the operating company name `Unit Corp` as a unit. No ticker-only exclusion rule is supported by this evidence.

### 13.5 Provider-defined interpretation and verdict

The candidate can be described only as an **Alpha Vantage provider-defined active Stock-labelled candidate set**, after explicit lower-bound exclusions. It cannot be described as all historical investable US common equities, a common-stock universe, or a deduplicated security population.

Size and liquidity gates might later remove some problem instruments, but this audit neither selects those gates nor treats them as a resolution of classification contamination. The remaining accepted limitation would be that all apparent survivors are unconfirmed security types and that the reported non-common rate is only a lower bound.

## 14. Size and Liquidity gate feasibility audit — provider-defined candidate set

**Status:** read/research-only audit completed 2026-09-27. This audit evaluates the provider-defined candidate list only. It does not construct `U_t`, select a threshold/window, or begin Stage 2.

### 14.1 Pre-outcome pilot and candidate population

The candidate remains `LISTING_STATUS(date=t, state=active, assetType=Stock)` after only the direct provider-name exclusions of Section 13: not a historical US common-stock universe. Its observed survivor counts are 4,004, 4,870, 5,578, and 6,086 on 2010-01-04, 2014-07-10, 2020-08-31, and 2024-01-02.

Before any OHLCV outcome was inspected, a deterministic pilot selected one surviving row from each observed exchange stratum (BATS, NASDAQ, NYSE, NYSE ARCA, NYSE MKT): five rows per date and 20 total. Within every date/exchange stratum it selects the lexicographically smallest SHA-256 digest of the seven raw LISTING_STATUS fields. This is reproducible and is not a population coverage estimate. `data/raw/size_liquidity_pilot/manifest.json` preserves 20 successful raw `TIME_SERIES_DAILY_ADJUSTED(outputsize=full)` responses, hashes, local paths, request status, and retrieval time, plus 20 earlier network errors with no payload. No credential is recorded.

This source-linkage test selected, among others, BATS `ZBZX` ("Bats Listed Test"), NYSE `PRE-P-G` ("Partnerre Ltd"), and NYSE ARCA `ESUS` ("UBS AG London Branch"). Their survival under intentionally narrow direct-name exclusions documents provider-defined scope, not a new classification.

### 14.2 SIZE

The only permitted definition remains frozen:

\[
\mathrm{MarketCap}^{PIT}_{i,t}=\mathrm{raw\_close}_{i,t-1}\times\mathrm{Shares}^{PIT}_{i,t}.
\]

It uses SEC `dei:EntityCommonStockSharesOutstanding` under the frozen acceptance-to-first-eligible-XNYS-open rule and the existing corporate-action/split-basis gate. No adjusted close, current shares, synthetic shares, or substitute share source was used.

| Metric definition | PIT feasibility | Coverage | `UNKNOWN` causes | Structural limitations | Verdict |
| --- | --- | --- | --- | --- | --- |
| Frozen `raw_close(t-1) * Shares_PIT(t)` and existing split-basis gate | **NOT DEMONSTRATED** for a historical candidate row | Five manually identity-resolved trace issuers were audited earlier; candidate-wide coverage is **0 / 20 pilot rows valid**, and cannot be estimated for the 4,004–6,086 row populations | No defensible historical AV-row-to-CIK bridge. Therefore shares, accession provenance, split compatibility, and price/shares compatibility cannot be tested. Conditional later causes remain missing/ambiguous shares, missing price, and split mismatch. | `LISTING_STATUS` has no persistent identifier and showed ticker duplication/reuse; OpenFIGI supplied no historical/as-of bridge. SEC facts identify issuers, not a selected historical AV row. Scaling would also require issuer mapping plus companyfacts/submissions and filing provenance work. | **NO-GO** |

Zero valid does not claim that every issuer has no shares. It means the pilot has no demonstrated historical identity evidence sufficient to call Market Cap `VALID`. Current ticker-to-CIK mapping would be an invalid repair of this known look-ahead/ticker-reuse defect.

**SIZE MEASUREMENT — NO-GO**

### 14.3 LIQUIDITY

Only the still-unfrozen input was inspected:

\[
\mathrm{DollarVolume}_{i,s}=\mathrm{RawClose}_{i,s}\times\mathrm{Volume}_{i,s}.
\]

All sessions must precede a future reconstitution date. No `L`, aggregation `h`, zero-volume treatment, or threshold was selected.

| Candidate representation | PIT feasibility | Coverage | `UNKNOWN` causes | Structural limitations | Verdict |
| --- | --- | --- | --- | --- | --- |
| Trailing `raw_close * volume` from Alpha Vantage daily responses | Fields **OBSERVED**; candidate-specific PIT linkage **NOT DEMONSTRATED** | All 20 payloads have the expected daily schema and at least one prior row with positive raw close/non-negative volume. Field presence is not candidate measurement: **0 / 20** are sufficiently identified. | No proven linkage from `(historical symbol, exchange, date)` to ticker-only OHLCV; no observed bar-availability timestamp/vintage; no defined history requirement; zeros need an explicit future treatment. | OHLCV returns a ticker series, not a persistent security ID/exchange-effective interval. It cannot prove continuity through reuse/change/exchange move. Present history extending beyond the historical date does not establish the data or adjustment vintage known then. | **NO-GO** |

The raw responses contain `open`, `high`, `low`, `close`, `adjusted close`, `volume`, dividend, and split fields. Pre-date histories range from 360 to 6,080 rows; none of the inspected rows has a non-positive close or negative volume. Yet 16 of 20 selections have zero-volume rows, ranging from 1 to 1,211 (including 910 of 922 pre-date rows for selected BATS `CETH`). A zero is data, not a parsing failure; its economic interpretation is **UNVERIFIED**. Field availability does not repair identity.

**LIQUIDITY MEASUREMENT — NO-GO**

### 14.4 UNKNOWN ANALYSIS

This is pilot-only. `candidate_count` is five deterministic rows; population is shown in parentheses. `liquidity_sufficient` means a PIT-safe candidate-level dollar-volume history, not mere field presence. `UNKNOWN != FAIL`.

| date | candidate_count | size_valid | size_unknown | liquidity_sufficient | liquidity_unknown | both_measurable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2010-01-04 | 5 (population 4,004) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) |
| 2014-07-10 | 5 (population 4,870) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) |
| 2020-08-31 | 5 (population 5,578) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) |
| 2024-01-02 | 5 (population 6,086) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) |

The hypothesis that excluding `UNKNOWN` is merely conservative without materially redefining the cross-section is **falsified for the current sources**. `UNKNOWN` is universal across all dates and exchanges in this pilot because a common identity bridge is absent, not because a small set happens to lack a field. No causal claim about true size, liquidity, or investment quality follows, and no candidate is called a `FAIL` merely for being `UNKNOWN`.

**UNKNOWN EXCLUSION RULE — NOT SUPPORTABLE**

This audit authorizes neither a size threshold nor an ADV threshold/window, and does not construct a universe, `X_char`, weights, risk exposures, portfolios, or backtests.

## 15. Contemporary Universe Feasibility Audit — Portfolio Geometry #03B

**Status:** audit/research-only, completed 2026-09-27. This section supersedes neither the historical no-go findings nor their scope. It evaluates one current cross-section only; it does not reconstruct historical identity, construct `U_{t_0}`, select thresholds/windows, construct `X_char`, weights, `F_risk`, a projection, or a backtest.

### 15.1 Frozen contemporary observation point

The audit fixes `t_0` as the XNYS regular-session opening on **2026-09-25 13:30:00 UTC** (09:30 America/New_York). The versioned repository XNYS calendar confirms that 2026-09-25 is a session, its prior session is 2026-09-24, and its regular close is 20:00 UTC. Thus market inputs are restricted to completed sessions through 2026-09-24. SEC facts are accepted only where their `acceptanceDateTime` maps through the frozen first-XNYS-open-strictly-after-acceptance rule to no later than this decision time.

This is operationally reproducible because the date, decision timestamp, calendar, raw source manifests, and the exclusion/mapping protocol below are explicit. It does **not** claim an intraday availability timestamp for Alpha Vantage daily bars, nor does it make historical-vintage claims about a currently retrieved response.

### 15.2 Preserved current sources and candidate set

`data/raw/contemporary_universe_audit/manifest.json` preserves a real Alpha Vantage `LISTING_STATUS(date=2026-09-25, state=active)` response and the official SEC current `company_tickers_exchange.json` response. The manifest has two successful raw payloads with SHA-256 hashes and two earlier payload-free network errors. The Alpha Vantage key and SEC User-Agent are neither printed nor stored in the manifest.

Only `assetType=Stock` is considered, followed by the existing direct provider-name exclusions from Section 13. No new ticker/name rule was introduced. The object is therefore still called **Alpha Vantage provider-defined active Stock-labelled candidate set**, not all US common stocks or the complete US equity market.

| raw_active_count | stock_labelled_count | explicitly_excluded | remaining_candidates |
| ---: | ---: | ---: | ---: |
| 14,483 | 8,608 | 1,227 | 7,381 |

### 15.3 IDENTITY

For this current-only audit, a candidate is `IDENTITY_RESOLVED` only if its surviving AV ticker occurs once in the current AV candidate response and exactly once in the current SEC ticker/CIK response. The resulting current CIK and both source exchange labels are retained as evidence. This is an explicit current mapping rule, not an inference of historical continuity. The raw exchange labels are not silently normalized: observed pairs include AV NASDAQ / SEC Nasdaq (3,668), AV NYSE / SEC NYSE (2,113), AV AMEX / SEC NYSE (216), AV BATS / SEC CBOE (19), and smaller differing pairs. These differences remain provenance, not proof of a historical venue identity.

| status | count | percentage of 7,381 | main causes |
| --- | ---: | ---: | --- |
| `IDENTITY_RESOLVED` | 6,057 | 82.06% | One surviving AV row and one current SEC ticker/CIK record. |
| `NO_SEC_MAPPING` | 1,324 | 17.94% | No current SEC ticker record for the AV ticker. |
| `IDENTITY_AMBIGUOUS` | 0 | 0.00% | No duplicate surviving AV ticker or duplicate SEC ticker in this snapshot. |
| `OTHER_UNRESOLVED` | 0 | 0.00% | None observed under the stated current-only protocol. |

**CURRENT IDENTITY — CONDITIONAL GO**

It is conditional because the evidence is only a current ticker/CIK relationship and provider-defined candidate membership. It cannot be described as a common-stock classification, historical identity bridge, or complete-market security master.

### 15.4 SIZE

For a pre-outcome, reproducible measurement pilot, the three lowest canonical-row SHA-256 digests in each AV exchange stratum were selected among current `IDENTITY_RESOLVED` candidates: 18 names across BATS, NASDAQ, NYSE, AMEX, NYSE MKT, and NYSE ARCA. `data/raw/contemporary_measurement_pilot/manifest.json` preserves submissions, companyfacts, and full daily adjusted-price responses for each selected candidate: 108 successful hashed raw responses and 54 earlier payload-free network errors.

The audit applied the existing implementation without modification:

\[
\mathrm{MarketCap}^{PIT}_{i,t_0}=\mathrm{raw\_close}_{i,t_0-1}\times\mathrm{Shares}^{PIT}_{i,t_0},
\]

with SEC `dei:EntityCommonStockSharesOutstanding`, acceptance-based eligibility, raw (not adjusted) close, and the unchanged split-basis gate.

| status | count | percentage of 18-pilot | main causes |
| --- | ---: | ---: | --- |
| `VALID` | 13 | 72.22% | Latest PIT-eligible shares and raw close have compatible observed split basis. |
| `UNKNOWN` | 5 | 27.78% | Three `SPLIT_BASIS_MISMATCH`; two `MISSING_SHARES`. |
| `MISSING_PRICE` | 0 | 0.00% | No prior-session raw-close failure observed in this pilot. |
| `AMBIGUITY` | 0 | 0.00% | No deterministic selection ambiguity observed in this pilot. |

The 13/18 rate is a stratified pilot result, not a claim of 72.22% coverage among all 6,057 resolved current candidates. It does demonstrate that the frozen construction can produce `VALID` contemporary observations without substituting current shares or adjusted prices; it also demonstrates that the split/missing-share statuses remain material and must remain `UNKNOWN`, not silently repaired.

**CURRENT SIZE MEASUREMENT — CONDITIONAL GO**

### 15.5 LIQUIDITY

The audit inspected only raw field availability for the unfrozen representation

\[
\mathrm{DollarVolume}_{i,s}=\mathrm{RawClose}_{i,s}\times\mathrm{Volume}_{i,s},\qquad s<t_0.
\]

No trailing window, aggregation function, threshold, or zero-volume treatment was chosen. Every one of the 18 pilot responses has at least 25 valid pre-`t_0` rows with positive raw close and non-negative volume (range: 25 to 6,765 rows). Eight pilot series contain one or more zero-volume rows. Zeros are preserved as observed data; their economic treatment remains a later decision.

| status | count | percentage of 18-pilot | main causes |
| --- | ---: | ---: | --- |
| `RAW_HISTORY_AVAILABLE` | 18 | 100.00% | Raw close and volume exist before `t_0` for all selected current identities. |
| `LIQUIDITY_UNKNOWN` | 0 | 0.00% | No missing/non-positive close or missing/negative-volume prerequisite observed. |
| `PENDING_REPRESENTATION` | 18 | 100.00% | `L`, aggregation, zero-volume treatment, and threshold remain deliberately unfrozen. |

**CURRENT LIQUIDITY MEASUREMENT — CONDITIONAL GO**

The condition is narrow: data prerequisites exist in the pilot, but the Alpha Vantage daily response does not evidence a bar-publication timestamp/vintage. This current cross-sectional audit therefore does not claim historical point-in-time bar provenance.

### 15.6 JOINT FEASIBILITY

| candidate_count | identity_resolved | size_valid | liquidity_sufficient | all_three_measurable |
| ---: | ---: | ---: | ---: | ---: |
| 7,381 | 6,057 (82.06%) | 13 / 18 pilot (72.22%) | 18 / 18 pilot (100.00%) | 13 / 18 pilot (72.22%) |

`size_valid`, `liquidity_sufficient`, and `all_three_measurable` are explicitly pilot denominators, not extrapolated population counts. The loss that is already measured is 1,324 candidates without a current SEC mapping, plus five of the 18 tested resolved candidates without a valid frozen Market Cap. The remaining loss cannot be quantified without a separately authorized scale-up; it must not be assumed random, small, or economically neutral.

The future interpretation is limited to **a contemporary cross-section drawn from an Alpha Vantage provider-defined Stock-labelled candidate set with positive evidence of current identity and investibility measurement**. It does not generalize to the complete US equity market, a survivorship-free historical universe, historical portfolio performance, persistent alpha, or historical implementability.

## 16. Liquidity Representation Audit — Portfolio Geometry #03B

**Status:** candidate-representation audit only, completed 2026-09-27. The object tested is recent typical dollar trading activity before the frozen opening decision `t_0 = 2026-09-25 13:30 UTC`, as a coarse investibility characteristic. It is not a depth, spread, impact, execution-cost, intraday-liquidity, or abnormal-event-volume model. No threshold, final universe, `X_char`, weights, risk matrix, projection, or backtest is constructed.

### 16.1 Candidate, information boundary, and sample

For `s < t_0`, the candidate raw observation is

\[
DV_{i,s}=\mathrm{RawClose}_{i,s}\times\mathrm{Volume}_{i,s},
\]

and its proposed summary is

\[
\mathrm{MedianDollarVolume}_{60\,\mathrm{Sessions},i,t_0}
=\operatorname{median}\{DV_{i,s}:s\text{ is one of the prior 60 XNYS sessions}\}.
\]

The exact session window is 2026-07-01 through 2026-09-24. No same-day or future row is used. Alpha Vantage documents its daily series as raw/as-traded OHLCV, while its daily-adjusted endpoint separately exposes adjusted close, split, and dividend information. [Alpha Vantage documentation](https://www.alphavantage.co/documentation/)

The base pilot is the pre-outcome 18-name sample in Section 15.4. A stress extension adds every current uniquely-SEC-mapped candidate in the complete BATS (19) and NYSE ARCA (7) strata, selected before OHLCV outcomes; six overlap the base set, yielding 38 distinct securities. `data/raw/contemporary_liquidity_stress/manifest.json` preserves two successful retrieval passes (52 successful raw responses/hashes for the same 26 selected records) and 26 earlier payload-free network errors. This is a stress sample, not a population-prevalence estimate.

### 16.2 LIQUIDITY REPRESENTATION PILOT

The audit verifies unique provider-date keys, chronological reindexing to the expected XNYS sessions, `s < t_0`, `RawClose > 0`, and `Volume >= 0`, without gap filling. A no-row expected session is `MISSING_OBSERVATION`; a present row failing either numerical predicate is `INVALID_OBSERVATION`; positive-close rows with positive or zero volume are `VALID_POSITIVE_VOLUME` and `VALID_ZERO_VOLUME`. No zero is converted to missing, nor vice versa.

| n_securities | full_history | partial_history | unknown | zero_volume_securities | zero_volume_observations | missing_observations | invalid_observations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 38 | 25 | 13 | 0 | 0 | 0 | 408 | 0 |

There are 1,872 `VALID_POSITIVE_VOLUME` observations. `n_valid` ranges 20–60 (median 60). `MedianDV_20` ranges from about $284 to $262.9m (median $74.0k); `MedianDV_60` from about $272 to $276.6m (median $47.8k). Median `MeanDV_20 / MedianDV_20` is 1.78 (maximum 116.73); median `MeanDV_60 / MedianDV_60` is 2.00 (maximum 89.90). These are diagnostics, not performance selection.

### 16.3 STRESS CASE A — isolated spike

`TOVX` is full-history and has exactly one 60-window session above five times its median: 2026-07-07 has $8.85m dollar volume, 81.1x `MedianDV_60` ($109.1k). Its next two largest observations are about $390.0k and $378.7k.

| TOVX diagnostic | value |
| --- | ---: |
| `MedianDV_20` / `MeanDV_20` | $103.8k / $120.5k |
| `MedianDV_60` / `MeanDV_60` | $109.1k / $275.1k |
| `max(DV_60) / MedianDV_60` | 81.1x |

The 60-session mean is 2.52x its median because of the isolated spike, while the 20/60 medians remain close. This supports median aggregation for *typical* activity in this real stress case; it does not label all high-volume events irrelevant.

### 16.4 STRESS CASE B — recent activity change

`ZCSH` is full-history and has `MedianDV_20` about $64.65m versus `MedianDV_60` about $2.61m (24.72x). `MeanDV_20` is $68.19m and `MeanDV_60` $24.84m; 22 of 60 sessions exceed five times its 60-session median. This is sustained recent activity, not a one-day spike.

The 60-session median remains dominated by the older low-activity observations until a majority of the window changes. Its lag is visible and is coherent with the stated coarse/structural object, but would not be suitable for current execution capacity. No later data or portfolio outcome is used.

### 16.5 STRESS CASE C — limited history

Thirteen securities are `PARTIAL_HISTORY`: six have 25 valid observations, four 32, two 37, and one (`MNGU`) exactly 20. All 408 missing expected sessions are leading-window gaps; every partial case has a complete trailing suffix through 2026-09-24, rather than intermittent gaps within the available suffix.

The partial median is arithmetically interpretable as the median of available valid observations in the maximum 60-session window, with `n_valid` retained. It has less evidential support than a full-60 value and must not silently be treated as equivalent. The data support retaining it as a labelled candidate state, not as an unqualified full-history result.

### 16.6 ZERO / MISSING OBSERVATION AUDIT

No `VALID_ZERO_VOLUME` case occurs: zero securities and zero observations in the 2,280 expected-session cells. The source schema can distinguish zero from missing, but this sample cannot establish whether a future zero should enter the median as real zero activity, receive a provider-quality status, or another treatment. `MISSING_OBSERVATION` remains missing with reason; it is never imputed as zero. `INVALID_OBSERVATION` is zero. Two full-history series (`IBO`, `OPTT`) have a split event inside the window; raw close times observed volume remains the audited raw input, but a separate historical volume-adjustment vintage is not evidenced.

**ZERO_VOLUME_POLICY — UNRESOLVED**

### 16.7 REPRESENTATION ASSESSMENT

- Median Dollar Volume is coherent with typical activity in the observed spike case.
- Sixty sessions are defensible for a coarse/structural object, with explicitly observed lag.
- Twenty observations are descriptively workable only as a labelled `PARTIAL_HISTORY` estimate with `n_valid`, not as equivalently supported history.
- Missing rows remain missing and determine history status; no imputation is permitted.
- The zero-volume policy cannot be frozen because no current-window zero was observed.

The economic object remains measurable, but the zero-observation/history-policy component lacks evidence required for a frozen representation. No liquidity threshold is selected.

### 16.8 Alpha Vantage zero-volume semantics micro-audit

**Scope:** this micro-audit addresses only the provider meaning of a present row where `RawClose > 0` and `Volume == 0`. It does not reopen the median, 60-session, 20-observation, partial-history, market-cap, identity, or universe decisions.

**Exact source.** The project uses `function=TIME_SERIES_DAILY_ADJUSTED`, `outputsize=full`, JSON responses. The inspected raw field is exactly `6. volume`; raw price is `4. close`. Alpha Vantage's official documentation says this endpoint supplies raw/as-traded daily open, high, low, close, and volume, together with adjusted close and split/dividend events. [Alpha Vantage documentation](https://www.alphavantage.co/documentation/)

| Claim | Evidence status |
| --- | --- |
| The endpoint contains a daily volume field and raw/as-traded OHLCV values. | **DOCUMENTED** |
| A present `6. volume = 0` literally means zero reported shares traded. | **NOT DOCUMENTED** |
| Zero is a provider missing-data marker, unavailable-data marker, placeholder, or non-trading observation. | **NOT DOCUMENTED** |
| A row absent for an expected session is distinct in the returned JSON structure from a present row carrying `"6. volume": "0"`. | **OBSERVED** in preserved raw payloads |

The documentation inspected contains no field definition or zero-value rule that resolves the latter two claims. The raw data therefore cannot elevate the economically plausible statement “zero volume may mean zero trading activity” into provider-documented semantics.

**Existing raw evidence (auxiliary only).** The read-only scan in `scripts/inspect_zero_volume_semantics.py` finds positive-close/zero-volume rows in preserved payloads. For example, `OXM` on 2000-10-11 has `open=high=low=close=17.44` and `volume=0`, between rows with positive volume; `QSOL` on 2025-12-23 has positive OHLC at 12.4687 and `volume=0`, between positive-volume dates. `CETH` has 994 such rows from 2016-02-11 through 2026-09-25, including consecutive rows with the same positive OHLC and zero volume. These are observed provider payload patterns, not evidence of their economic/provider semantics. In particular, a populated OHLC row with zero volume does not prove whether the provider is reporting zero trades, a carried quote, a thin/inactive instrument convention, or another process.

**Consequence.** `MISSING_OBSERVATION` remains a distinct, non-imputed state. No policy is frozen for a present zero-volume row: it must not yet be automatically counted as valid, mapped to `DollarVolume = 0`, included in the median, or counted toward `n_valid`.

### 16.9 Liquidity Representation Closure -- Zero-Volume Policy

**Status:** completed 2026-09-30. This section supersedes the unresolved
zero-volume-policy status in Sections 16.6--16.8 only. It does not alter the
Size result in Section 17, choose a Size or Liquidity threshold, or construct
`U_t0`, `X_char`, weights, `F_risk`, a projection, or a backtest.

**Provider clarification.** Alpha Vantage Support supplied the following
direct responses; their wording is preserved exactly:

| Question to support | Exact provider answer |
| --- | --- |
| "If TIME_SERIES_DAILY_ADJUSTED returns a daily row with valid OHLC prices and volume = 0, can I always interpret 0 as zero recorded trading volume, and never as missing/unavailable volume data?" | "Yes, you can always interpret 0 as zero traded trading volume" |
| "If volume data is missing/unavailable for a trading day, how does TIME_SERIES_DAILY_ADJUSTED represent it?" | "The volume field will be 'null'" |

The clarification resolves the semantic ambiguity from Section 16.8. Its
meaning is consistent with the prior raw observations of positive-price rows
with `volume = 0` (OXM, QSOL, and CETH), but those raw examples alone did not
establish that semantic. The provider statement defines its field behavior; it
does not endorse the project's aggregation or history methodology.

**Frozen zero-volume policy.** The following is the project's methodological
decision derived from the provider clarification and existing audit:

| Returned daily-row state | Classification | Dollar-volume treatment | `n_valid` |
| --- | --- | --- | --- |
| Valid raw price fields and `volume = 0` | `VALID_ZERO_VOLUME` | `RawClose * 0 = 0` | Counts as valid. |
| `volume = null` | `MISSING_VOLUME` | Missing; never converted to zero. | Does not count. |
| Expected XNYS session with no returned row | `MISSING_OBSERVATION` | Missing; never imputed. | Does not count. |
| Present row with another invalid required input | `INVALID_OBSERVATION` | Missing for this representation. | Does not count. |

Thus `volume = 0 != volume = null`: a valid observed zero enters the median
as zero dollar activity, whereas null and absent observations remain missing
and reduce `n_valid`.

**Frozen history policy.** For the 60 eligible XNYS sessions strictly before
`t0 = 2026-09-25 13:30 UTC`:

| Status | Exact rule | Representation |
| --- | --- | --- |
| `FULL_HISTORY` | `n_valid = 60` | Median of 60 valid observations. |
| `PARTIAL_HISTORY` | `20 <= n_valid < 60` | Median of valid observations, retained with `n_valid` and status. |
| `UNKNOWN` | `n_valid < 20` | No Liquidity representation. |

`PARTIAL_HISTORY` remains lower-evidence than `FULL_HISTORY`; the
20-observation minimum is unchanged. The 13 earlier partial cases had leading
history gaps and continuous recent suffixes, which remains the empirical basis
for retaining that labelled state.

**Frozen Liquidity representation.** The economic object is recent typical
dollar trading activity before the decision time for coarse investibility, not
depth, spread, impact, execution cost, or intraday liquidity. For each valid
observation in the 60-session window:

\[
\mathrm{DollarVolume}_{i,s}=\mathrm{RawClose}_{i,s}\times\mathrm{Volume}_{i,s}.
\]

For `FULL_HISTORY` or `PARTIAL_HISTORY`:

\[
\mathrm{MedianDollarVolume}_{60\,\mathrm{Sessions},i,t_0}
=\operatorname{median}\{\mathrm{DollarVolume}_{i,s}:s\in W_{60}(t_0),\
\;\text{observation valid}\}.
\]

The prior stress evidence still applies: median suppresses isolated spikes,
the 60-session measure intentionally reacts slowly to sustained changes, and
partial observations retain their explicit lower history depth.

**Remaining limitations.** This is coarse investibility liquidity, not an
execution model; the median intentionally suppresses isolated spikes; the
60-session median intentionally lags recent regime changes; and
`PARTIAL_HISTORY` is not equivalent to `FULL_HISTORY`. The clarification does
not prove security eligibility, tradability, or execution capacity. No
Liquidity threshold, percentile/rank cutoff, or universe size is selected.

**Closure assessment:** zero semantics are resolved; valid zeros can enter as
`DollarVolume = 0`; `null` remains missing; the 60-session median remains
coherent for this stated object; and no blocker remains for the representation
itself. Threshold selection, candidate eligibility, and Size coverage remain
separate Stage 2 issues.

**LIQUIDITY REPRESENTATION — FROZEN**

## 17. Contemporary Size Measurement Scale-Up and Cross-Section Audit

**Status:** completed 2026-09-30 for the frozen contemporary point only. This is a measurement-coverage audit, not a Size-gate decision. It does not construct \(U_{t_0}\) and does not modify the frozen definition:

\[
\mathrm{MarketCap}^{PIT}_{i,t_0}=\mathrm{RawClose}_{i,t_0-1}\times\mathrm{Shares}^{PIT}_{i,t_0}.
\]

The decision is the XNYS opening on 2026-09-25 at 13:30 UTC. Price is the completed 2026-09-24 raw close and SEC shares must be PIT eligible by the decision. Adjusted price, provider-current shares, weighted-average shares, synthetic shares, and a forward fill across a split remain forbidden.

### 17.1 Pre-outcome scale-up sample and evidence

The relevant current-only population is the 6,057 `IDENTITY_RESOLVED` members from Section 15. Acquiring three endpoints for all 6,057 names was not operationally proportionate. Before any Size result was inspected, `scripts/acquire_contemporary_size_scaleup.py` froze 171 selections in `data/raw/contemporary_size_scaleup/selection.json`:

- census of each exchange with at most 50 resolved candidates: BATS 19, NYSE ARCA 7, NYSE MKT 25;
- for AMEX, NASDAQ, and NYSE, ten lowest canonical-row SHA-256 digests in each nonempty reported-IPO cohort: 40 each.

The selected IPO cohorts contain 31 (through 2000), 33 (2001-2010), 34 (2011-2020), and 73 (2021+) candidates. This is a broad reproducible sample, but not a probability sample proportional to the 6,057 names. All coverage figures are sample estimates, not population claims.

Each selection has raw SEC `submissions`, SEC `companyfacts`, and AV `TIME_SERIES_DAILY_ADJUSTED(outputsize=full)` evidence. The manifest contains 513 request records: 511 `HTTP_200` payloads and two preserved `HTTP_ERROR_404` `companyfacts` responses (`SCZM`, `IPB`). It records selection rule, symbol/exchange/CIK, endpoint, retrieval time, local path, status, byte size, and SHA-256 without any credential. All 511 stored payload hashes revalidated. A few files from an interrupted pre-manifest attempt are excluded from the audit.

`scripts/inspect_contemporary_size_scaleup.py` applies the existing split gate. It does not define a new Market Cap formula.

### 17.2 SIZE SCALE-UP COVERAGE

| Field | Count | Percent of 171 | Interpretation |
| --- | ---: | ---: | --- |
| `population_identity_resolved` | 6,057 | - | Current-only candidate population; not fully acquired. |
| `n_candidates_audited` | 171 | 100.00% | Frozen stratified/census sample. |
| `SIZE_VALID` | 102 | 59.65% | Price, security-to-registrant shares linkage, PIT eligibility, and split basis pass. |
| `SIZE_UNKNOWN` total | 69 | 40.35% | Insufficient evidence under the frozen rule; never a small-size result. |

### 17.3 SIZE UNKNOWN BREAKDOWN

| Status | Count | Percent of 171 | Interpretation |
| --- | ---: | ---: | --- |
| `SIZE_UNKNOWN_MISSING_SHARES` | 17 | 9.94% | No deterministic PIT-eligible DEI shares observation. |
| `SIZE_UNKNOWN_SPLIT_BASIS` | 6 | 3.51% | An intervening split invalidates raw-close times stale reported shares. |
| `SIZE_UNKNOWN_MISSING_PRICE` | 0 | 0.00% | No sample case. |
| `SIZE_UNKNOWN_IDENTITY_OR_MAPPING` | 44 | 25.73% | Multiple SEC registrant tickers; no evidence links issuer-level DEI common shares to the selected security. |
| `SIZE_UNKNOWN_OTHER` | 2 | 1.17% | Preserved `companyfacts` 404 for `SCZM` and `IPB`. |

The identity/mapping check is input validation, not a new Size rule. DEI shares belong to the registrant. Where `submissions` lists more than one ticker, the sources do not provide a field assigning that share count to the selected ticker. For example, 17 selected BATS/NYSE ARCA instruments map to Bank of Montreal, whose SEC submission lists 45 tickers. A mechanical product would use issuer shares for the selected instruments, so it is `UNKNOWN`, not `VALID`.

### 17.4 UNKNOWN STRUCTURE

| AV exchange | Audited | Valid | Unknown | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| AMEX | 40 | 34 | 6 | 15.00% |
| BATS | 19 | 7 | 12 | 63.16% |
| NASDAQ | 40 | 28 | 12 | 30.00% |
| NYSE | 40 | 21 | 19 | 47.50% |
| NYSE ARCA | 7 | 1 | 6 | 85.71% |
| NYSE MKT | 25 | 11 | 14 | 56.00% |

| Reported IPO cohort | Audited | Valid | Unknown | Unknown rate |
| --- | ---: | ---: | ---: | ---: |
| Through 2000 | 31 | 24 | 7 | 22.58% |
| 2001-2010 | 33 | 27 | 6 | 18.18% |
| 2011-2020 | 34 | 23 | 11 | 32.35% |
| 2021+ | 73 | 28 | 45 | 61.64% |

These are descriptive associations, not causal claims. They show that excluding `SIZE_UNKNOWN` would materially reshape the observed sample by exchange and reported listing cohort. The largest cause is the documented security-to-registrant ambiguity; missing DEI shares and split incompatibility add independent constraints.

### 17.5 VALID MARKET-CAP DISTRIBUTION

Only the 102 `SIZE_VALID` records enter this un-winsorized, un-clipped US-dollar distribution.

| Statistic | MarketCap PIT |
| --- | ---: |
| Minimum | $2.27m |
| P01 | $2.62m |
| P05 | $5.16m |
| P10 | $9.18m |
| P25 | $32.99m |
| Median | $231.90m |
| P75 | $1.37bn |
| P90 | $7.70bn |
| P95 | $27.38bn |
| P99 | $109.58bn |
| Maximum | $673.08bn |

### 17.6 LOWER-TAIL AND EXTREME DIAGNOSTICS

The inspector predeclared $100m, $1bn, and $10bn solely as round diagnostic reference levels: 36/102 (35.29%), 73/102 (71.57%), and 94/102 (92.16%) fall below them. They are not thresholds or candidate thresholds.

The low `OGEN` extreme is $0.5030 raw close times 4,511,957 PIT shares (accession `0001493152-26-037749`, eligible 2026-08-14), yielding $2.27m with no split. The high `EC` extreme is $16.37 times 41,116,694,690 PIT shares (accession `0001104659-26-053172`, eligible 2026-05-01), yielding $673.08bn with no split. Both are `VALID_EXTREME`, retained without clipping. `MTNB` ($2.62m) and `MPC` ($109.79bn) independently pass the same component checks. In contrast, multi-ticker registrants are excluded before distribution calculation; `CANF`, `MI`, `IBO`, `OPTT`, `HLSQ`, and `MPU` remain `SIZE_UNKNOWN_SPLIT_BASIS` rather than being repaired.

### 17.7 METHODOLOGICAL ASSESSMENT

1. PIT Market Cap is demonstrably measurable for a strictly verified subset, with raw provenance and hash validation, but not at sufficiently broad security-specific coverage for the provider-defined cross-section.
2. `UNKNOWN` is 40.35% of the audit sample, not a population estimate.
3. The main cause is security-to-registrant shares ambiguity, followed by missing shares, split-basis incompatibility, and two unavailable SEC payloads.
4. `UNKNOWN` is neither small nor dispersed; excluding it risks defining a later candidate cross-section through source availability.
5. The valid subset describes a broad Size range, but is not sufficient to design a Size investibility gate for the full candidate cross-section without resolving security-specific shares/identity.

No Size threshold, gate family, liquidity rule, final universe, \(X_{\mathrm{char}}\), \(w_{\mathrm{raw}}\), \(F_{\mathrm{risk}}\), projection, or performance result is selected or constructed here.

**SIZE SCALE-UP — NO-GO**

**ZERO_VOLUME SEMANTICS — NOT DOCUMENTED**

**LIQUIDITY REPRESENTATION — REQUIRES_REVISION**

**CONTEMPORARY UNIVERSE FEASIBILITY — CONDITIONAL GO**

**Verdict: `Alpha Vantage provider-defined stock candidate set — CONDITIONAL GO`.** The condition is explicit acceptance of the provider-defined scope, application of only the audited direct-descriptor exclusions, retention of `UNKNOWN` status for all survivors, and prohibition on describing the resulting candidate set as a historical common-stock universe. This is a feasibility finding, not authorization to construct `U_t` or start Stage 2.
