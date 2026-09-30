# Stage 1 — Characteristic Research Proposal

**Status:** Stage 1C is **CLOSED**. The seven raw issuer-level components are
frozen: EarningsYield; Profitability; CashRealization; RevenueCAGR_3Y;
PositiveAnnualGrowthFraction_3Y; R_12M_AdjustedClose; and
R_1M_AdjustedClose. Historical audit sections below remain part of the record
and may retain their original status where explicitly superseded. No component
is ranked, normalized, winsorized, imputed, or placed in `X_char` here.

## Scope

This document preserves the required chain:

\[
\text{economic concept} \rightarrow \text{candidate measure} \rightarrow
\text{raw data and provenance} \rightarrow \text{frozen raw component}.
\]

The candidates are not risk exposures, do not establish alpha, and do not
define `g(X)`, `F_risk`, portfolio weights, or a projection.

## Data and temporal constraints

| Source | Evidence | Constraint |
| --- | --- | --- |
| SEC EDGAR | Audited facts link to CIK, accession, form, period, filing date, and acceptance timestamp. | Tag coverage, units, amendments, and duplicate facts vary. |
| SEC timing | Research candidate: first full eligible regular session strictly after acceptance. | It is a conservative convention, not observed public dissemination time. |
| Alpha Vantage daily | Adjusted close, dividends, splits, and OHLCV were observed. | Momentum uses adjusted close through the prior XNYS close; historical adjustment vintage and bar-publication time remain limitations. |
| Alpha Vantage overview | Current snapshot only. | Not usable for historical fundamentals, sector history, or PIT market cap. |

`Observed / conditional` below means the raw input exists but its coverage,
comparability, or timing is unresolved. `Pending` means the needed PIT use is
not yet established.

**Raw-field vocabulary, not formulas.** SEC candidates include
`us-gaap:Assets`, `Liabilities`, `StockholdersEquity` (or its attributable-
interest variant), `NetIncomeLoss`, `OperatingIncomeLoss`, `GrossProfit`,
`NetCashProvidedByUsedInOperatingActivities`,
`PaymentsToAcquirePropertyPlantAndEquipment`, and issuer-specific revenue tags.
Historical share count requires a separately verified DEI or filing fact;
Alpha Vantage `OVERVIEW.SharesOutstanding` is not point-in-time historical
evidence. Alpha Vantage daily price candidates use `5. adjusted close`,
`7. dividend amount`, and `8. split coefficient` from the audited raw series.

## Value

**Economic concept.** Value compares market price with an evidenced economic
resource, earnings stream, or cash-generation stream. It is not synonymous
with a convenient provider ratio. Book-to-market, earnings-to-price, and
cash-flow-to-price are distinct constructions in the Fama–French definitions.
[Kenneth French variable definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html)

| Candidate empirical measure | Raw inputs and support | High / low direction | Problems, sectors, timing, and overlap |
| --- | --- | --- | --- |
| Book equity to market equity (B/P) | SEC equity: observed / conditional; PIT shares and price: conditional. | High = more book equity per market price; low = higher price relative to book equity. | Negative equity, buybacks, goodwill, intangibles, and standards matter. Banks may be more book-oriented than asset-light firms. Shares a price denominator with E/P and CF/P. |
| Earnings to market equity (E/P) | `NetIncomeLoss`: observed / conditional; PIT shares and price: conditional. | High positive = more reported earnings per price; negative earnings are not automatically cheap. | Losses, tax, one-offs, cyclicality, and trailing-period policy matter. Overlaps with profitability Quality. |
| Operating cash flow to market equity (CF/P) | Operating cash flow: observed / conditional; shares and price: conditional. | High = more operating cash flow per price. | Working-capital swings, negative cash flow, and financial-sector comparison need policy. Overlaps with cash-conversion Quality and E/P. |
| Enterprise value to operating measure | Price, shares, debt, cash, and revenue or operating measure: conditional. | Lower multiple conventionally indicates cheaper enterprise valuation. | Debt, cash, leases, minority interest, and preferred equity definitions are unresolved; financial firms are often unsuitable. Adds capital-structure overlap. |

**Frozen Value decision:** PIT Earnings Yield is the selected raw Value
representation for this experiment. Book equity, cash-flow, enterprise-value,
loss-treatment, sector-comparability, and any future transformation policy
remain separate decisions.

### Stage 1C frozen policy — quarterly Net Income information set

**CONDITIONAL GO for `NI_TTM_PIT`.** Its empirical basis is the
executed AAPL/JPM/XOM/UNH/WMT audit of preserved `us-gaap:NetIncomeLoss` facts,
accession-linked filing provenance, direct-quarter facts, cumulative facts,
amendments, and comparative duplicates.

At portfolio decision time `t`, the usable SEC information set for issuer `i`
is:

```text
F_(i,t) = {SEC facts PIT-eligible for issuer i at t}
```

The following information-vintage and ambiguity policy is **FROZEN** for the
future quarterly Net Income reconstruction:

1. **No retrospective restatement.** A later filing must never rewrite an
   earlier information set. Historical features must not be constructed from
   today's latest-restated history.
2. **Amendments are prospective information events.** A `10-Q/A` or `10-K/A`
   becomes usable only from its own PIT eligibility date. Before then, the
   previously eligible filing vintage remains the available information.
3. **Context compatibility is required.** Arithmetic may use facts only when
   issuer/CIK, accounting concept, unit, fiscal-period structure, nested period
   boundaries required for subtraction, and information vintage/filing
   provenance are economically and accountingly compatible.
4. **Direct and reconstructed quarters are consistency checks.** The only
   candidate identities are `Q2 = H1_YTD - Q1`, `Q3 = 9M_YTD - H1_YTD`, and
   `Q4 = FY - 9M_YTD`; they do not authorize subtraction of arbitrary SEC rows.
   Each identity requires the compatibility checks above. Where an appropriate
   direct-quarter fact exists, it is compared with the reconstructed candidate.
5. **Unresolved inconsistency means missing.** If direct and reconstructed
   representations materially disagree and provenance/context cannot resolve
   the difference deterministically, `quarter = NA`; no representation is
   silently selected.
6. **Unresolved non-amendment duplicates mean missing.** If multiple
   PIT-eligible facts represent the required economic object and provenance or
   context gives no deterministic selection basis, `quarter = NA`. Never
   average, choose the desired value, select by downstream behavior, or silently
   use the latest database row.
7. **Missing or incompatible components remain missing.** If a required quarter
   lacks a compatible PIT-eligible direct or reconstructed path, `quarter = NA`.
   A future TTM requires four valid, economically distinct quarters; missing
   quarters are never imputed.
8. **The policy is deliberately conservative.** Its purpose is not to recover
   every issuer-quarter, but to avoid inventing fundamental information for the
   downstream geometric experiment.

Observed durations include quarter-duration, H1/YTD-duration, 9M/YTD-duration,
and annual-duration facts. Consequently, `fp` alone does not identify a
standalone quarter. This policy resolves the Net-Income-specific ambiguity and
missingness requirement; generic Stage 1A amendment alternatives below remain
historical discussion for other future accounting concepts.

`src/pit_net_income.py` now implements the policy as an auditable input object:
`build_quarters_as_of(facts, issuer, decision_time)` returns standalone-quarter
records with CIK, sources, acceptance timestamps, PIT eligibility, method, and
status; `ttm_net_income_as_of(...)` returns `NI_TTM_PIT` only when the four
latest valid, distinct, consecutive quarters are available. Original
non-amendment facts are selected by first eligible acceptance; an amendment
supersedes prospectively only under its own eligibility. Conflicting rows at
the selected event remain `NA`.

The `pit_net_income` implementation itself does not calculate Market Cap or
Earnings Yield; those composition rules are frozen separately below. It does
not construct a Value score or `X_char`.

### Stage 1C frozen implementation control — XNYS PIT eligibility

The Stage 1B availability convention is materialized by
`src/xnys_pit.py`, using `exchange-calendars==4.12` and its `XNYS` calendar.
`eligibility_from_acceptance(acceptance_datetime)` requires a timezone-aware
timestamp, converts it to UTC, and returns an `XNYSEligibility` record with
three distinct fields: `acceptance_datetime`, `eligibility_session`, and
`portfolio_decision_time` (the session opening in UTC).

The selected session is the first scheduled XNYS session opening strictly after
SEC acceptance. Thus pre-open acceptance can use that day's opening, while
in-session and post-close acceptance wait for the next session. Weekends and
scheduled holidays are skipped. Scheduled early-close sessions remain eligible;
they retain a regular session opening. This is an auditable implementation of a
research timing convention, not proof of the exact SEC.gov dissemination time.

### Stage 1C price/share-basis audit — conditional raw-close candidate

**FROZEN for PIT Market Cap.** The preserved Alpha Vantage AAPL history identifies a
4-for-1 split on 2020-08-31 (`split coefficient = 4.0`). Its raw close moves
from 499.23 on 2020-08-28 to 129.04 on the split date, while the earlier
adjusted close is already 120.97. The latest PIT-eligible SEC reported-share
state remains 4.276bn shares through that event; only the later 10-K reports
17.002bn post-split shares. Consequently, raw-close times the reported state is
economically coherent before the split and again after the post-split shares
observation becomes eligible, but not in the intervening stale-shares interval.
Adjusted close times those historical shares is incompatible before the split:
it retrospectively incorporates the later split and also contains dividend
adjustment. WMT's 3-for-1 split on 2024-02-26 shows the same pattern.

The frozen formula is:

```text
MarketCap_(i,t) = raw_close_(i,t-1) * Shares_PIT_(i,t)
```

At the XNYS session open on `t`, `raw_close_(i,t-1)` is the completed close of
the preceding eligible market session and `Shares_PIT_(i,t)` is the latest
PIT-eligible SEC `dei:EntityCommonStockSharesOutstanding` observation.
`adjusted_close` is forbidden for this historical reported-share Market Cap
object; it may answer a different, separately governed return-series question.

The corporate-action compatibility gate is also **FROZEN**: return
`SPLIT_BASIS_MISMATCH` and `MarketCap_PIT = NA` whenever a split/coefficient
event occurred after the selected shares reference date and on or before the
raw-price observation date. Do not forward-fill a pre-split reported share
state across that event and do not synthetically adjust SEC shares. Other
explicit outcomes include `VALID`, `MISSING_PRICE`, `MISSING_SHARES`, and
`AMBIGUOUS_SHARES`.

`src/pit_value.py` materializes this rule through
`market_cap_as_of(prices, shares, issuer, decision_time)`, retaining raw-price
date/value, SEC shares value/accession/reference date/PIT eligibility, relevant
split event, Market Cap, status, and reason.

### Stage 1C frozen raw Value characteristic — PIT Earnings Yield

The Value concept for this experiment is **current economic generation relative
to market valuation**. Its selected raw representation is:

```text
EarningsYield_(i,t) = NI_TTM_PIT_(i,t) / MarketCap_PIT_(i,t)
```

`NI_TTM_PIT` follows the separately frozen quarterly Net Income policy. If its
status or the Market Cap status is not `VALID`, `EarningsYield = NA` with the
component provenance/status retained; no value is imputed. The reusable
interface is `earnings_yield_as_of(facts, prices, shares, issuer,
decision_time)`. This freezes a raw issuer-level characteristic only. It does
not create ranks, normalization, scores, `X_char`, portfolio weights, or a
backtest.

## Quality

**Economic concept.** Quality concerns profitability, cash generation,
balance-sheet resilience, and earnings quality. It is not recent stock return
or accounting growth. Gross profitability is one candidate representation;
operating profitability and investment are distinct in Fama–French research.
[Novy-Marx](https://www.nber.org/papers/w15940), [Fama & French (2015)](https://doi.org/10.1016/j.jfineco.2014.10.010)

### Stage 1C prerequisite — PIT Assets for a future Profitability denominator

**Historical prerequisite record — superseded by the frozen Stage 1C decision
below.** `us-gaap:Assets` is an instant fact: its reference date is not a
Net-Income-like duration, filing vintage, or PIT availability date. The Quality
notebook links every inspected Assets fact to accession, form, acceptance time,
and the frozen XNYS eligibility convention.

For AAPL's valid 2025-11-03 TTM (economic coverage 2024-09-29 through
2025-09-27), the candidate beginning snapshot is 2024-09-28 Assets of
$364.980bn from 10-K `0000320193-24-000123`, PIT eligible on 2024-11-01; the
ending snapshot is 2025-09-27 Assets of $359.241bn from 10-K
`0000320193-25-000079`, PIT eligible on 2025-10-31. Thus both were known at
the selected decision. JPM, XOM, UNH, and WMT also supplied the natural
beginning/end endpoints for their selected recent TTM decisions, but all five
traces exposed repeated/comparative reference dates.

Candidate rule, still subject to the unresolved issues below: for valid
`NI_TTM_PIT(i,t)`, select compatible PIT Assets instants at the day before the
first TTM quarter and at the final TTM quarter-end; use their arithmetic mean
only if both are unambiguous, same issuer/CIK/unit, and independently
PIT-eligible. Otherwise return `NA`; no imputation or retrospective rewrite.
Amendments remain prospective.

`ROA_end` is simpler but combines a trailing flow with a terminal stock.
`ROA_average` is economically better aligned with resources employed over the
flow, but adds missingness, staleness, duplicate-vintage, fiscal-transition,
and acquisition sensitivity. Unresolved: generic duplicate selection,
materiality/staleness, financial-sector interpretation, and whether two points
adequately represent intra-year assets. The later Stage 1C decision freezes
only raw Profitability; no Quality composite or `X_char` is authorized here.

| Candidate empirical measure — not selected | Raw inputs and support | High / low direction | Problems, sectors, timing, and overlap |
| --- | --- | --- | --- |
| Gross profitability to assets | `GrossProfit` and `Assets`: conditional; gross profit is absent for some audited issuers. | High = greater gross profit per asset base. | Cost classification, acquisitions, and asset-light firms matter; banks/insurers do not map cleanly. Overlaps with margins and asset efficiency. |
| Operating profitability or margin | Revenue, cost, SG&A, interest, or `OperatingIncomeLoss`; assets/equity/revenue: conditional. | High = stronger operations under the eventual selected denominator. | The definition is a research choice, not a provider field. Sector-specific interest and SG&A matter. Overlaps with gross profit, ROA, E/P, and earnings Growth. |
| Return on assets or equity | Net income and assets/equity: observed / conditional. | High = more income per capital base. | Leverage changes ROE mechanically; negative denominators, buybacks, intangibles, and one-offs need policy. Average versus ending balances is unresolved. |
| Cash-generation quality | Operating cash flow, net income, assets/sales: observed / conditional. | High conversion may indicate stronger cash realization. | Working capital and cash-flow classification can be transitory. Overlaps with CF/P and profitability. |
| Balance-sheet conservatism | Assets, liabilities, debt, cash, equity: conditional. | Lower leverage / stronger liquidity conventionally indicates resilience. | Banks, insurers, utilities, leases, pensions, and off-balance-sheet obligations need special treatment. Intersects Value and asset growth. |

**Quality decisions pending:** whether Quality is one composite or several
dimensions; financial-sector treatment; balance-sheet/cash-flow definitions;
and whether margins, ROA, and cash conversion are nonredundant.

## Growth

**Economic concept.** Growth is expansion in an economically meaningful
operating or capital base over comparable periods. It is not a valuation
multiple. Acquisitions, inflation, base effects, and restatements must be
distinguished from sustainable expansion. Fama–French asset investment is a
candidate viewpoint, not a Growth definition here. [Kenneth French variable definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html)

| Candidate empirical measure — not selected | Raw inputs and support | High / low direction | Problems, sectors, timing, and overlap |
| --- | --- | --- | --- |
| Revenue growth | Two comparable revenue facts: conditional, with issuer-specific tag choice. | High = faster top-line expansion; low/negative = slower growth or contraction. | Organic/acquired growth, currency, fiscal calendars, base effects, and bank revenue semantics matter. Each report has its own as-of date. |
| Earnings or operating-income growth | Two net-income or operating-income facts: conditional. | High = faster reported earnings expansion. | Small/negative bases, tax, impairments, one-offs, and restatements can dominate. Overlaps with Quality and E/P. |
| Operating-cash-flow growth | Two operating-cash-flow facts: observed / conditional. | High = expansion in operating cash generation. | Working-capital changes and cyclical cash conversion may be transitory. Overlaps with Quality and CF/P. |
| Asset growth / investment intensity | Two `Assets` facts: observed / conditional. | High = rapidly expanding asset base. | Expansion may be productive, acquisitive, or inefficient; financial assets are not comparable with industrial assets. Intersects asset-efficiency Quality. |

**Growth decisions pending:** annual versus quarterly horizon; rate versus
acceleration; negative/small-base treatment; organic/acquisition treatment;
and sector-specific revenue semantics.

## Momentum

**Economic concept.** Momentum is persistence in a security's own past market
performance over a pre-specified horizon. It is market-data based, not
fundamental Growth. Jegadeesh and Titman document winner/loser continuation
over 3-to-12-month horizons; this motivates candidates but selects none.
[Jegadeesh & Titman (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x)

| Candidate empirical measure — not selected | Raw inputs and support | High / low direction | Problems, sectors, timing, and overlap |
| --- | --- | --- | --- |
| Intermediate-horizon adjusted-return momentum with skip interval | PIT adjusted daily closes; fixed formation date/window. Price fields observed; daily decision-time pending. | High = stronger own prior adjusted performance; low = weaker performance. | Lookback and skip cannot follow results. Delistings, halts, stale prices, adjustments, and close-time availability need policy. |
| Shorter-horizon continuation | Same price series, shorter fixed lookback: observed / conditional. | High = stronger recent performance. | More sensitive to reversal, microstructure, and turnover. Likely redundant without a distinct rationale. |
| Distance to trailing high / 52-week-high proximity | Adjusted price history and fixed high window: observed / conditional. | High = price near its own prior high. | Adjustment/window convention must be frozen; it differs from cumulative return after volatile paths. Substantially overlaps with momentum. |

**Momentum decisions pending:** formation frequency; horizon and skip;
total-return adjustment semantics; close-to-decision timing; treatment of
missing sessions, halts, and delistings; and whether one view is sufficient.

## Cross-family overlap and decisions before freezing

| Relationship | Why it requires a decision |
| --- | --- |
| Value ↔ Quality | E/P and CF/P use earnings/cash flow also used for profitability; combining them can double-count accounting sources. |
| Quality ↔ Growth | Profitability level and change are distinct but can be mechanically correlated. |
| Growth ↔ Value | High-growth firms often have lower B/P or E/P; this is an economic tension, not a sign error. |
| Momentum ↔ fundamentals | Momentum uses a price clock; the other families use filing clocks. Availability rules must remain separate. |
| Financials ↔ non-financials | Revenue, debt, cash flow, assets, and book equity differ economically; a sector policy is needed before any comparison. |

Before a final definition, research must decide: (1) a justified nonredundant
candidate set; (2) SEC amendment and competing-filing as-of selection; (3) an
eligible regular-session calendar; (4) daily price availability/decision time;
(5) historical shares/market-equity provenance; (6) sector treatment; and
(7) missing, negative, zero, and unusual-accounting policies. No transformation,
ranking, normalization, winsorization, or imputation is implied here.

## Stage 1A — Characteristic Prerequisite Resolution

**Status:** research prerequisites documented; no candidate measure is selected
and no characteristic is calculated. The statuses below have only three
meanings: **RESOLVED** = evidence supports a bounded input or rule;
**DECISION REQUIRED** = more than one defensible research policy remains; and
**NOT SUPPORTED** = the audited source cannot substantiate the proposed use.

### 1. Historical shares and market capitalization

The audited SEC `companyfacts` payloads contain the DEI fact
`dei:EntityCommonStockSharesOutstanding` for all five frozen issuers. It is a
distinct, instantaneous shares fact rather than an EPS denominator: the SEC
XBRL guide describes it as an instant-period common-stock-shares disclosure,
and the SEC separately warns not to confuse outstanding with issued shares.
[SEC XBRL Guide](https://www.sec.gov/files/edgar/filer-information/specifications/xbrl-guide-2024-07-08.pdf)
[SEC data-quality reminder](https://www.sec.gov/newsroom/whats-new/osd-announcement-2210-dqreminder-entitycommonstocksharesoutstanding)

| Frozen issuer | Observed DEI shares facts | Observed date range | Provenance fields on every inspected fact | Status |
| --- | ---: | --- | --- | --- |
| AAPL | 70 | 2009-06-27 to 2026-07-17 | `end`, `val`, `accn`, `filed` | **RESOLVED** for discrete filing observations |
| JPM | 73 | 2009-07-31 to 2026-06-30 | `end`, `val`, `accn`, `filed` | **RESOLVED** for discrete filing observations |
| XOM | 69 | 2009-06-30 to 2026-03-31 | `end`, `val`, `accn`, `filed` | **RESOLVED** for discrete filing observations |
| UNH | 69 | 2009-07-30 to 2026-07-31 | `end`, `val`, `accn`, `filed` | **RESOLVED** for discrete filing observations |
| WMT | 70 | 2009-09-04 to 2026-08-26 | `end`, `val`, `accn`, `filed` | **RESOLVED** for discrete filing observations |

The observed forms are predominantly 10-K/10-Q, with amendments in AAPL,
JPM, XOM, and WMT and three JPM 8-K observations. The count is evidence of
coverage, not evidence that a fact has been selected, deduplicated, or
converted into a daily series. Each fact must still be linked through its
accession to the separate accepted filing and governed by the Stage 0B timing
rule.

The following quantities are deliberately not interchangeable:

| Quantity | Meaning | Can it be substituted for PIT shares outstanding? | Stage 1A status |
| --- | --- | --- | --- |
| Shares outstanding | A stock count at the disclosed instant, represented here by the DEI fact and its `end`. | Only as that discrete disclosed observation, after accession/timing selection. | **RESOLVED** for the audited discrete SEC observations. |
| Weighted-average basic or diluted shares | A duration-weighted EPS denominator. Diluted shares can include assumed conversions; both can differ from an instant share count. | No. | **NOT SUPPORTED** as a substitute. |
| Split-adjusted share quantity | A quantity restated by a provider or issuer to maintain comparability after a split. Its adjustment convention must be known. | No; it may be useful for a separate per-share series only after its adjustment history is audited. | **NOT SUPPORTED** as a substitute. |
| Current shares | A retrieval-time snapshot, such as Alpha Vantage `OVERVIEW.SharesOutstanding`. | No; it has no historical as-of provenance. | **NOT SUPPORTED** for historical use. |
| Historical point-in-time shares | The latest *eligible*, accession-linked discrete shares observation under the frozen as-of policy. | This is the required object for historical market equity, explicitly with a discrete reported-state assumption between filings. | **RESOLVED** as the approved reported-shares-as-of convention. |

Thus, a historical market-equity candidate would be
`eligible historical shares outstanding × an eligible price`, not an EPS share
count or a present-day overview value. The SEC facts establish a dated,
filing-linked sequence, but not the exact number legally outstanding on every
intervening day: issuance, repurchases, conversions, or a new filing can occur
between observations. Stage 1B freezes the **reported-shares-as-of** convention:
use the latest PIT-eligible reported shares-outstanding observation, explicitly
as a discrete reported state rather than as a continuously observed true daily
share count. Its staleness remains a documented measurement limitation.

| Candidate measure already proposed | Historical market capitalization required? | Consequence today |
| --- | --- | --- |
| B/P, E/P, CF/P | Yes: each uses market equity as its price-side denominator. | Raw-close basis is conditionally supported only with the frozen split-transition `NA` guard; each candidate's accounting policy remains unresolved. |
| Enterprise-value multiples | Yes, plus debt/cash and a separate enterprise-value policy. | Shares convention is frozen; enterprise-value definitions remain unresolved. |
| Gross/operating profitability, margin, ROA/ROE, cash conversion, balance-sheet conservatism | No market capitalization is intrinsic to these fundamental-only candidates. | Still conditional on SEC tags, amendment policy, period comparability, and timing; not blocked by shares. |
| Revenue, earnings, cash-flow, or asset growth | No market capitalization is intrinsic. | Still conditional on two eligible comparable facts and amendment policy; not blocked by shares. |
| Price momentum / distance to high | No share quantity is intrinsic. | Depends instead on price availability and adjustment semantics. |

### 2. Amendment and competing-filing selection

**RESOLVED:** chronology is non-negotiable. A fact may enter an information set
only when its own filing becomes eligible under the acceptance-based session
rule. An original filing must remain historically visible; a later 10-K/A or
10-Q/A must never rewrite a pre-amendment information set. The SEC identifies
the acceptance date/time separately from the official filing date and states
that an accession is the unique identifier of an accepted submission.
[SEC EDGAR timestamp FAQ](https://www.sec.gov/about/webmaster-frequently-asked-questions)

Before a final definition, the following deterministic alternatives must be
chosen explicitly:

| Candidate policy | Deterministic as-of rule | Implication |
| --- | --- | --- |
| Filing-vintage supersession | For an exact economic fact key (issuer, taxonomy, tag, unit, context/period), retain every accession. At each as-of time, select the latest eligible accession in the appropriate 10-K/10-Q family; an amendment supersedes only from its own eligibility time onward. | Preserves what was knowable, incorporates a disclosed correction when known, and requires a precise fact/context equivalence key. |
| Amendment-only supersession | Keep the original unless a later eligible `10-K/A` or `10-Q/A` states the same fact key; then use the amendment from its own eligibility time. Non-amendment duplicate filings remain competing records. | More conservative about non-amendment re-filings, but can retain an obsolete original if a correction arrives through another eligible form. |
| Original-only | Use the first eligible original for its period permanently; retain amendments as evidence but never use them. | Strictly historical but knowingly ignores corrections after they became public. |
| Latest-restated dataset | Use the newest fact known today for every historical period. | **NOT SUPPORTED** for this experiment's point-in-time information set because it creates look-ahead before the later filing. |

**DECISION REQUIRED:** freeze one of the first three alternatives, define the
fact/context equivalence key (including dimensions and unit), and specify how
to handle a 8-K or non-amendment filing that presents the same period. Regardless
of policy, raw originals, amendments, accession numbers, acceptance datetimes,
and rejected competing facts remain retained as provenance.

### 3. Exchange calendar required by the SEC availability rule

**RESOLVED:** the relevant reference market for this frozen U.S. sample is the
U.S. regular equity session. NYSE publishes core regular hours as 09:30–16:00
Eastern Time and publishes holidays and early closes.
[NYSE holidays and trading hours](https://www.nyse.com/trade/hours-calendars)
The implementation must use a versioned schedule with session opens, full
closures, early closes, daylight-saving changes, and exceptional closures—not a
weekday rule. `exchange_calendars` with the `XNYS` calendar is an established,
reproducible implementation candidate; its release/version and the underlying
NYSE schedule used must be recorded and checked against the official exchange
calendar. [exchange_calendars documentation](https://pypi.org/project/exchange-calendars/)

**FROZEN CONVENTION:** scheduled XNYS early-close sessions are eligible
sessions. The future implementation must still freeze and record the calendar
version/snapshot and must handle extraordinary closures explicitly. No SEC/price
join is performed here.

### 4. Momentum decision time and legal daily prices

Momentum has a price clock, not a filing clock. The appropriate information
set depends on when the portfolio is formed; neither convention below selects a
formula, window, skip interval, or frequency.

| Formation convention | Information legally usable at decision time | Information not usable | Status |
| --- | --- | --- | --- |
| At session `t` open | Daily observations whose sessions ended no later than `t-1`; in particular, close/adjusted-close through `t-1`, subject to verified vendor availability by the open. | Any `t` OHLCV, close, adjusted close, dividend, split, or end-of-day field. | **RESOLVED** logically; provider availability of the previous bar remains conditional. |
| After the close of session `t` | Close/adjusted-close for `t` only after the official close *and* after the specific vendor has made a completed, correctly adjusted daily bar available; otherwise only through `t-1`. | A provisional/in-progress `t` bar; a `t` bar whose vendor publication time is not evidenced. | **NOT SUPPORTED** today for same-day Alpha Vantage use: Stage 0A observed no historical bar-availability timestamp. |

The audited Alpha Vantage response provides session dates, OHLCV, adjusted
close, dividends, and split coefficients, but `Last Refreshed` is endpoint
metadata rather than evidence that any historical daily bar was available by a
particular decision time. A current adjusted-price history may also embody a
later split/dividend adjustment. Therefore an eventual total-return momentum
series needs an explicit, point-in-time corporate-action adjustment policy;
it must not assume today's adjusted history was known unchanged in the past.
The portfolio formation convention is frozen below. Corporate-action adjustment
semantics, momentum horizon, skip interval, and frequency remain
**DECISION REQUIRED**.

### 5. Sector treatment is a representation decision, not a default

No sector neutralization or sector-relative normalization is imposed in Stage
1A. The following alternatives answer different questions:

| Approach | What it does | Consequence |
| --- | --- | --- |
| Raw cross-sectional comparison | Compares a candidate value across all issuers unchanged. | Retains absolute levels but can reflect sector composition and accounting-model differences; a high/low value may not have the same meaning for a bank and an industrial. |
| Within-sector characteristic comparison | Compares an issuer with assigned sector peers before later use. | Changes the characteristic representation itself: it would modify a future `X_char` and requires a point-in-time sector taxonomy, membership history, and minimum-peer policy. |
| Later sector adjustment at portfolio/risk stage | Leaves the characteristic representation unchanged and later treats sectors as candidate exposure directions or constraints. | Would act on the portfolio/risk object after a future `g(X)`, potentially through a future `F_risk`; it does not make the underlying characteristic sector-relative. |

**FROZEN CONVENTION:** no sector-relative scoring or sector neutralization is
applied inside a future `X_char` at this stage. A later geometric removal of a
sector direction is not equivalent to sector-neutralizing `X_char`: the former
would change a future portfolio vector; the latter would change the input
representation. No `F_risk`, normalization, or sector adjustment is defined
here.

### Stage 1A status ledger

| Prerequisite | Classification | Reason |
| --- | --- | --- |
| Discrete SEC shares-outstanding observations | **RESOLVED** | All five audited issuers have DEI facts with date, value, accession, and filed date. |
| Weighted-average/diluted EPS shares as replacement for shares outstanding | **NOT SUPPORTED** | They are different economic quantities. |
| Alpha Vantage current or undated shares as historical PIT shares | **NOT SUPPORTED** | The audited endpoint has no availability/provenance adequate for the use. |
| Daily reported-shares-as-of convention for market capitalization | **RESOLVED** | Use the latest PIT-eligible reported discrete shares observation; it is not a true continuously observed daily count. |
| Amendment/duplicate selection policy for quarterly Net Income | **RESOLVED** | Stage 1C freezes prospective amendments and `NA` for unresolved competing non-amendment facts; generic policies for other future accounting concepts remain separate. |
| Exchange-session source/mechanism | **RESOLVED** | Official NYSE schedule plus a versioned XNYS implementation candidate identifies the required mechanism. |
| Early-close inclusion | **RESOLVED** | Scheduled XNYS early-close sessions are eligible; calendar version remains an implementation control. |
| Momentum at session open | **RESOLVED** as an information-set boundary | Through-`t-1` completed daily observations only; vendor timeliness still needs evidence. |
| Same-session post-close Alpha Vantage price use | **NOT SUPPORTED** | No historical daily-bar availability timestamp was audited. |
| Momentum adjustment/formation policy | **DECISION REQUIRED** | Corporate-action vintages, decision time, horizon, and skip remain unfrozen. |
| Sector treatment inside a future `X_char` | **RESOLVED** | No sector-relative scoring or sector neutralization is applied at this stage. |
| Sector treatment at a later portfolio/risk stage | **DECISION REQUIRED** | It is a separate future decision and cannot redefine the characteristic representation. |

## Stage 1B — Frozen timing and representation conventions

These conventions govern feasibility and timing only. They do **not** select a
candidate, freeze a characteristic formula, define transformations, or construct
`X_char`.

| Research object | Frozen convention |
| --- | --- |
| Portfolio decision time | The regular-session open on session `t`. |
| Eligible market information | Completed market observations through session `t-1` only. No observation from session `t` may enter the open decision. |
| Eligible SEC fundamentals | The Stage 0B conservative convention: the opening of the first eligible XNYS regular trading session strictly after the filing's SEC `acceptanceDateTime`. This remains a research eligibility rule, not an observed SEC.gov dissemination timestamp. |
| Scheduled early closes | Scheduled XNYS early-close sessions are eligible sessions under the SEC eligibility convention. |
| Historical shares | The latest PIT-eligible `dei:EntityCommonStockSharesOutstanding` observation. It represents the latest reported discrete state, not a continuously observed true daily share count. |
| Amendments | An amendment enters prospectively from its own PIT eligibility date. It never rewrites an earlier historical information set. The broader duplicate/non-amendment selection policy remains separately documented as a decision requirement. |
| Sector treatment inside characteristics | No sector-relative scoring and no sector neutralization inside a future `X_char` at this stage. |

## Characteristic decision table

This is a decision aid, not a formula-selection table. “Conditional” PIT
feasibility means that the raw candidate input exists or is auditable, but final
use still depends on the frozen timing conventions, accession/context checks,
and any stated remaining policy. “High” describes the expected economic
direction, not a trading recommendation or a score.

| Family / candidate | Economic interpretation | Formula concept (not frozen) | Required inputs | PIT feasibility under frozen conventions | Major measurement weakness | Sector-comparability issue | Redundancy / overlap | Expected direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Value — book equity to market equity | Book resources relative to the market's valuation. | Eligible book equity divided by eligible market equity. | SEC equity; latest eligible reported shares; raw close through `t-1`. | Conditional: raw-close basis needs the split-transition `NA` guard; accounting definition remains open. | Negative equity, buybacks, goodwill, and intangibles. | Banks can be more naturally book-based than asset-light issuers. | Same market-equity denominator as E/P and CF/P. | Higher is conventionally more value-like. |
| Value — PIT Earnings Yield **(SELECTED)** | Current PIT economic generation relative to PIT market valuation. | `NI_TTM_PIT / (raw_close_(t-1) × Shares_PIT_(t))`. | Valid PIT Net Income TTM; latest eligible SEC reported shares; raw close through `t-1`; split gate. | Implemented conditionally: invalid numerator/denominator or a split-basis mismatch returns `NA`. | Losses, tax, one-offs, cyclicality, reported-share staleness, and split-transition missingness. | Earnings definitions and bank economics differ across sectors. | Uses earnings also appearing in profitability Quality and earnings Growth. | Higher positive yield is conventionally more value-like; negative earnings are not automatically cheap. |
| Value — operating cash flow to market equity | Operating cash generation relative to market valuation. | Eligible operating cash flow divided by eligible market equity. | SEC operating cash flow; latest eligible reported shares; raw close through `t-1`. | Conditional: trailing-period policy and split-transition `NA` guard remain required. | Working-capital swings, negative operating cash flow. | Cash-flow reporting and financial-sector interpretation differ. | Overlaps cash-generation Quality and the same denominator as B/P and E/P. | Higher is conventionally more value-like. |
| Value — enterprise value to operating measure | Value of the operating enterprise relative to its operating scale. | Eligible enterprise-value concept divided by revenue or operating measure. | Price, shares, debt, cash, and eligible revenue or operating measure. | Conditional with additional unresolved definitions for debt, cash, leases, and enterprise value. | Capital-structure and lease/minority-interest definitions. | Often not meaningful for financial firms. | Shares price-side information with other Value candidates; capital-structure overlap with leverage Quality. | Lower multiple is conventionally more value-like. |
| Quality — gross profitability to assets | Gross profit produced by the asset base. | Eligible gross profit divided by eligible assets. | SEC `GrossProfit`, `Assets`. | Conditional: tags are not uniformly present; both facts need eligible accession/context selection. | Cost classification, acquisitions, and asset-light models. | Banks and insurers do not map cleanly to gross profit. | Overlaps margins and asset-efficiency measures. | Higher indicates stronger gross profitability. |
| Quality — operating profitability or margin | Operating performance before the eventually selected denominator. | Eligible operating income relative to revenue, assets, or equity. | SEC operating income plus selected denominator. | Conditional: denominator and tag/context policy are not frozen. | Interest, SG&A, and definition choice can drive results. | Sector-specific operating models make a single denominator uneven. | Overlaps gross profitability, ROA/ROE, E/P, and earnings Growth. | Higher indicates stronger operating profitability. |
| Quality — return on assets or equity | Earnings generated per accounting capital base. | Eligible net income divided by eligible assets or equity. | SEC net income and assets or equity. | Conditional: facts have SEC provenance; average-versus-ending balance policy remains open. | Leverage mechanically affects ROE; one-offs and negative denominators. | Financial issuers have structurally different balance sheets. | Uses net income also in E/P and earnings Growth. | Higher generally indicates greater accounting profitability. |
| Quality — cash-generation quality | Degree to which accounting income or activity is realized as operating cash. | Eligible operating cash flow relative to net income, assets, or sales. | SEC operating cash flow, net income, assets or sales. | Conditional: inputs are auditable where tags exist; exact comparison concept remains open. | Working capital can dominate short-term cash conversion. | Cash-flow classification and banking economics differ. | Overlaps CF/P, profitability, and cash-flow Growth. | Higher conversion conventionally indicates stronger cash realization. |
| Quality — balance-sheet conservatism | Financial resilience through leverage or liquidity. | Eligible debt/liability/cash/equity relationship under a future definition. | SEC assets, liabilities, debt, cash, equity. | Conditional: tag coverage and debt/cash definition remain unresolved. | Leases, pensions, off-balance-sheet obligations. | Banks, insurers, and utilities require special interpretation. | Intersects enterprise-value inputs and asset Growth. | Lower leverage / stronger liquidity is conventionally higher quality. |
| Growth — revenue growth | Expansion in top-line operating activity. | Change between two eligible comparable revenue facts. | Two eligible SEC revenue facts. | Conditional: issuer-specific revenue-tag selection and same-horizon policy remain open. | Acquisitions, currency, fiscal calendars, and base effects. | Bank revenue has different economic semantics. | Related to operating profitability and valuation-growth tension. | Higher positive change indicates faster top-line growth. |
| Growth — earnings or operating-income growth | Expansion in reported profitability. | Change between two eligible net-income or operating-income facts. | Two eligible SEC income facts. | Conditional: comparable periods and treatment of restatements are required. | Small/negative bases, tax, impairments, and one-offs. | Profit measurement differs by sector. | Overlaps Quality profitability and E/P. | Higher positive change indicates faster earnings expansion. |
| Growth — operating-cash-flow growth | Expansion in operating cash generation. | Change between two eligible operating-cash-flow facts. | Two eligible SEC operating-cash-flow facts. | Conditional: eligible SEC facts exist where reported; horizon remains open. | Working-capital variation can masquerade as growth. | Financial-sector cash-flow meaning differs. | Overlaps cash-generation Quality and CF/P. | Higher positive change indicates faster cash-flow expansion. |
| Growth — asset growth / investment intensity | Expansion of the accounting asset base. | Change between two eligible assets facts. | Two eligible SEC `Assets` facts. | Conditional: eligible facts and horizon are available; no growth horizon is selected. | Acquisitions can dominate organic investment. | Financial assets are not comparable with industrial assets. | Intersects balance-sheet Quality and capital intensity. | Higher indicates faster asset expansion, not automatically better quality. |
| Momentum — intermediate-horizon adjusted-return with skip | Persistence of prior own-market performance while avoiding the most recent interval. | Return over a fixed past window excluding a fixed recent window. | Eligible daily price history through `t-1`; corporate-action treatment. | Conditional: open-time cutoff is frozen; adjustment vintage, horizon, and skip remain open. | Delistings, halts, stale prices, and retrospective adjustments. | Directly comparable as own-return history, though sector composition can still affect cross-sectional outcomes. | Substantially overlaps shorter-horizon continuation and 52-week-high proximity. | Higher prior return is conventionally stronger momentum. |
| Momentum — shorter-horizon continuation | Persistence over a more recent, shorter price window. | Return over a shorter fixed past window. | Eligible daily price history through `t-1`; corporate-action treatment. | Conditional: same timing and adjustment limitations as intermediate momentum. | Reversal, microstructure, and higher turnover sensitivity. | Same comparability caveat as other price momentum. | Likely redundant with intermediate momentum absent a distinct rationale. | Higher prior return is conventionally stronger continuation. |
| Momentum — distance to trailing high / 52-week-high proximity | Whether price is near its own trailing peak. | Latest eligible price relative to the maximum eligible price in a fixed trailing window. | Eligible daily price history through `t-1`; corporate-action treatment. | Conditional: timing cutoff is frozen; window and adjustment semantics remain open. | Sensitive to split/dividend adjustment and chosen window. | Uses own-price history, but sector momentum can influence cross-sectional patterns. | Overlaps cumulative-return momentum but differs after volatile paths. | Higher proximity to the trailing high is conventionally stronger momentum. |

## References and stop condition

- Fama, E. F. and K. R. French, [variable definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/variable_definitions.html).
- Fama, E. F. and K. R. French, [*A Five-Factor Asset Pricing Model*](https://doi.org/10.1016/j.jfineco.2014.10.010), 2015.
- Novy-Marx, R., [*The Other Side of Value*](https://www.nber.org/papers/w15940), 2013.
- Jegadeesh, N. and S. Titman, [*Returns to Buying Winners and Selling Losers*](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x), 1993.

Except for the separately frozen raw Value / PIT Earnings Yield and Quality /
PIT Profitability specifications, this proposal does not authorize final
definitions, calculations, `X_char`, `g(X)`, `F_risk`, universe construction,
portfolio construction, or geometric decomposition.
### Stage 1C decision — PIT Profitability (supersedes the prior prerequisite text)

**FROZEN — Quality / Profitability raw component**

\[
\mathrm{Profitability}_{i,t}
= \frac{NI\_TTM\_PIT_{i,t}}{\mathrm{AverageAssets\_PIT}_{i,t}},
\qquad
\mathrm{AverageAssets\_PIT}_{i,t}
= \frac{Assets_{begin,PIT}+Assets_{end,PIT}}{2}.
\]

Profitability means the ability to generate profit efficiently using the total
economic resources employed by the company. `AverageAssets_PIT` is a
two-snapshot approximation to resources employed during the same TTM interval,
not a continuously observed daily asset count.

**Frozen selection policy.** For a valid `NI_TTM_PIT(i,t)`, use only
`us-gaap:Assets` instants at the calendar day before the first TTM economic
quarter and at the final TTM economic quarter-end. Each selected fact must be
the same issuer/CIK, `Assets` concept, and compatible unit; it must be
PIT-eligible at decision time and retain its accession, form, acceptance, and
eligibility provenance. Do not substitute a nearby Assets date. If a required
boundary is absent or future-only, return `NA` with
`INCOMPATIBLE_ASSET_BOUNDARY`; if a selected vintage is ambiguous or contexts
differ, return `NA` with its explicit failure status. No imputation or
retrospective rewrite is permitted. Amendments apply only prospectively from
their own eligibility; a later non-amendment comparative fact cannot rewrite a
previous information set.

`ROA_end` combines a TTM flow with a terminal stock. The selected average is
better aligned with resources employed during the TTM, while retaining explicit
limitations: acquisitions, disposals, major consolidation-perimeter changes,
large intra-period asset changes, and the special economic interpretation of
financial-sector balance sheets. These limitations do not permit silent
adjustments.

Cash Realization, Balance-sheet Resilience, a Quality composite, ranks,
normalization, and `X_char` remain unimplemented.
### Stage 1C audit — Operating Cash Flow for a possible Cash Realization component

**Status: PENDING — not frozen or implemented.** The audited common SEC
concept for AAPL, JPM, XOM, UNH, and WMT is
`us-gaap:NetCashProvidedByUsedInOperatingActivities`. The preserved facts
carry issuer/CIK, unit, start/end, FY/FP, form, accession, acceptance time,
and derived XNYS PIT eligibility. Observed structures are Q1 (about 88–90
days), H1 YTD (180–181), 9M YTD (272–273), and FY (363–370); direct Q2/Q3
OCF must not be assumed.

**OCF TTM reconstruction verdict: CONDITIONAL GO.** The candidate identities
`Q2 = H1 - Q1`, `Q3 = 9M - H1`, and `Q4 = FY - 9M` are feasible only when
issuer/CIK, concept, unit, period structure, and PIT vintage are compatible.
Each operand must be independently eligible at the decision time. Amendments
remain prospective and unresolved duplicate contexts or unavailable operands
must return `NA`; later comparative filings cannot rewrite an earlier
information set.

At the AAPL 2025-11-03 decision, four eligible OCF quarters cover the same
2024-09-29 to 2025-09-27 interval as `NI_TTM_PIT`: Q1 is direct and Q2/Q3/Q4
are compatible cumulative reconstructions. The resulting OCF TTM is
$111.482bn against NI TTM $112.010bn, or a raw ratio of about 0.995.

**Raw `OCF_TTM_PIT / NI_TTM_PIT` final-representation verdict: NO-GO /
REQUIRES TRANSFORMATION.** The conceptual eligibility boundary is frozen:
if `NI_TTM_PIT <= 0`, any future Cash Realization result is `NA`, because the
concept concerns the cash backing of positive earnings. Yet positive NI does
not make the raw ratio structurally well behaved: representative observations
include JPM -2.59 (negative OCF with positive NI), AAPL 0.995, and XOM/UNH/WMT
about 1.80/1.63/1.90. A 2023-onward descriptive audit found ranges of
0.995–1.262 (AAPL), -2.629–1.053 (JPM), 1.442–1.885 (XOM), 0.681–1.922
(UNH), and 1.789–2.303 (WMT).

Values above one mean more reported OCF than reported Net Income during the
interval; they do not automatically mean proportionally more Quality. Near
`NI = 0+` the quotient can become unstable, while a cap or bounded transform
would sacrifice magnitude and sign information. Choosing such a transformation
is a separate research decision. The current raw evidence does not establish a
complete causal attribution for unusual cases: the WMT high-ratio filing has
reported depreciation/amortization, while JPM has bank-specific cash-flow and
financial-instrument economics, but the aggregate audited facts alone do not
prove a full reconciliation.

Cash Realization, Balance-sheet Resilience, a Quality composite, and `X_char`
remain unimplemented.
### Stage 1C decision — PIT Cash Realization (supersedes its prior audit status)

**FROZEN — Quality / Cash Realization raw component**

Cash Realization asks: given positive accounting earnings, to what extent are
they backed by operating cash generation?

\[
\mathrm{CashRealization}_{i,t} =
\begin{cases}
\min\left(\frac{OCF\_TTM\_PIT_{i,t}}{NI\_TTM\_PIT_{i,t}}, 1.0\right), & NI\_TTM\_PIT_{i,t} > 0, \\
\mathrm{NA}, & NI\_TTM\_PIT_{i,t} \leq 0.
\end{cases}
\]

`OCF_TTM_PIT` uses `us-gaap:NetCashProvidedByUsedInOperatingActivities` and
the frozen compatible PIT-quarter policy: Q2 = H1 − Q1, Q3 = 9M − H1, and
Q4 = FY − 9M. It requires matching issuer/CIK, concept, unit, compatible
period structure, PIT eligibility, accession provenance, and deterministic
vintage selection. Amendments enter prospectively; unresolved duplicates,
incompatibilities, missing required quarters, or invalid NI/OCF TTM return
`NA` with their explicit statuses.

The cap is **upper only**. `OCF < 0` with positive NI remains negative; it is
not floored at zero. `OCF >= NI` maps to exactly one because this component
measures whether positive earnings are fully cash-backed, not progressively
higher quality above that threshold. Thus: `CR = 1` means fully backed or
better; `0 < CR < 1` partially backed; `CR = 0` zero operating cash; and
`CR < 0` positive earnings with negative operating cash.

Cash Realization is a standalone raw component. Balance-sheet Resilience, the
Quality composite, ranks, normalization, and `X_char` remain unimplemented.
### Stage 1C audit — Balance-Sheet Resilience candidates

**Status: PENDING — not frozen or implemented.** Balance-Sheet Resilience is
financial burden relative to capacity to support it; lower reported debt alone
is not a sufficient economic interpretation.

**Observed SEC facts and definitions.** Debt and cash are instant facts, while
Operating Income and Interest Expense are duration facts requiring the same
PIT TTM/vintage discipline as other flows. The audit found materially
different available tags: AAPL `LongTermDebt` (which equals its disclosed
current plus noncurrent LT debt at the inspected boundary), UNH
`DebtLongtermAndShorttermCombinedAmount`, WMT `LongTermDebt` plus separate
`ShortTermBorrowings`, XOM `LongTermDebtAndCapitalLeaseObligations` plus
`DebtCurrent`, and JPM long-term debt, short-term borrowings, deposits, and
other bank funding. These are not demonstrated as one common, non-overlapping
`TotalDebt` definition and must not be silently summed.

The conservative cash candidate is only `CashAndCashEquivalentsAtCarryingValue`.
Restricted cash and short-term investments are not added merely to obtain Net
Debt. It is unavailable at JPM's representative boundary, although a broader
restricted-cash measure exists; therefore JPM Net Debt is `NA` under this
candidate rather than broadened. Net Debt is consequently not cross-sector
comparable in this sample.

`OperatingIncomeLoss` is distinct from EBIT and was not synthesized into EBIT.
Operating-Income / Interest Expense can be constructed with aligned PIT TTM
coverage for UNH (about 4.74) and WMT (about 12.87) in the representative
observations. It cannot be treated as universal: XOM lacks audited
`OperatingIncomeLoss`, JPM lacks a corresponding operating-income series, and
AAPL's available interest TTM is not contemporaneous with its current
Operating Income TTM. Unaligned flows return `NA` rather than a ratio.

Representative diagnostics using single issuer-specific reported debt tags
show OCF/Debt around 1.23 (AAPL), -0.34 (JPM), 1.52 (XOM), 0.25 (UNH), and
1.09 (WMT), but these values do not establish comparability. Net-debt ratios
are available only where the narrow cash fact and selected debt tag coexist.
JPM is the explicit stress test: its deposits, borrowings, interest expense,
cash, and OCF have bank-specific balance-sheet meanings, so none of the three
candidate families is cross-sector coherent as a universal Quality input.

| Candidate | PIT feasibility | Cross-sector coherence | Main defect |
| --- | --- | --- | --- |
| OCF TTM / Debt | Conditional | **NOT CROSS-SECTOR COHERENT** | No universal non-overlapping Debt definition; bank funding differs. |
| OCF TTM / Net Debt | Conditional | **NOT CROSS-SECTOR COHERENT** | Debt and narrow-cash scope differ; JPM cash is unavailable under the conservative definition. |
| Operating Income / Interest Expense | Conditional subset | **CONDITIONALLY COHERENT** | Not EBIT; tag/coverage availability differs and is absent or stale for key issuers. |

**Recommendation: no current candidate is sufficiently coherent** for a
universal Balance-Sheet Resilience component. This is an accounting and
economic conclusion, not a performance conclusion. A later proposal would
need an explicit scope/sector policy or separately justified issuer-class
representations. Balance-Sheet Resilience, the Quality composite, and
`X_char` remain unimplemented.
### Stage 1C audit — PIT Revenue and Revenue-Growth trajectories

**Status: PENDING — Growth is not frozen or implemented.** Growth is sustained
expansion of underlying business activity, with distinct magnitude and
persistence dimensions. Revenue is the audited proxy; it is not Net Income,
EPS, price growth, or an analyst forecast.

**Issuer-specific revenue concepts.** The audited PIT tag map is AAPL
`RevenueFromContractWithCustomerExcludingAssessedTax`; JPM
`RevenuesNetOfInterestExpense`; and XOM, UNH, WMT `Revenues`. All selected
facts retain CIK, start/end, unit, FY/FP, form, accession, acceptance time,
and XNYS eligibility. Their structures include Q1, H1 YTD, 9M YTD, FY, and
in some filings direct Q2/Q3. Compatible reconstruction can therefore use
`Q2 = H1 - Q1`, `Q3 = 9M - H1`, `Q4 = FY - 9M` under the frozen no-look-ahead,
prospective-amendment, duplicate-to-NA, and context-compatibility policies.

**Revenue TTM reconstruction verdict: CONDITIONAL GO.** Four valid PIT
economic quarters form a deterministic `Revenue_TTM_PIT` with preserved
quarter/accession provenance. It remains conditional on tag continuity,
compatible units/periods/vintages, and explicit missingness for conflicts or
unavailable components.

Representative annual PIT TTM paths show the purpose of separating magnitude
and persistence. AAPL's selected path is $394.3bn, $383.3bn, $391.0bn,
$416.2bn: an intermediate contraction followed by expansion. UNH rises from
$324.2bn to $447.6bn and WMT from $611.3bn to $713.2bn, with positive selected
annual intervals. JPM rises from $128.7bn to $182.4bn but annual growth
decelerates from about 22.9% to 12.3% to 2.8%. XOM falls from $413.7bn to
$332.2bn with two contractions. These are descriptive PIT paths, not rankings
or return predictions.

**Magnitude verdict: CONDITIONAL GO.** A ratio such as a current PIT TTM
against a prior PIT TTM is economically transparent, but its horizon, base
effect treatment, acquisition treatment, and minimum history are unresolved.
**Persistence verdict: CONDITIONAL GO.** PIT paths support transparent
diagnostics—positive-growth fraction, contraction count, and growth
dispersion—but no persistence formula, threshold, or history requirement is
selected. Persistence is not merely low revenue volatility: stable negligible
growth is not thereby strong Growth.

Cross-sector status is **CONDITIONALLY COHERENT**. Revenue is broadly a
top-line activity measure for the non-financial issuers. JPM's revenue after
interest expense is a banking activity measure with different economics from
industrial/customer-contract sales; it must not be forced into an identical
interpretation. Unresolved decisions before freezing Growth are the magnitude
horizon, persistence representation, minimum history, tag-change policy,
acquisition/base-effect treatment, and an explicit sector/JPM policy.

Growth, its components, any score, and `X_char` remain unimplemented.

### Stage 1C decision — PIT Revenue Growth (supersedes its prior audit status)

**FROZEN — Growth as two separate raw Revenue components**

Growth represents the magnitude and persistence of reported top-line business
activity expansion. It is represented by issuer-specific, point-in-time (PIT)
Revenue TTM observations, not by earnings, price performance, forecasts, or a
single composite score.

For each issuer and decision time `t`, the repository requires **exactly four**
comparable, consecutive annual `Revenue_TTM_PIT` observations, denoted
`Revenue_TTM_PIT(t-3y)`, `Revenue_TTM_PIT(t-2y)`, `Revenue_TTM_PIT(t-1y)`, and
`Revenue_TTM_PIT(t)`. The frozen raw outputs are:

\[
\mathrm{GrowthMagnitude}_{i,t} =
\left(\frac{Revenue\_TTM\_PIT_{i,t}}
{Revenue\_TTM\_PIT_{i,t-3y}}\right)^{1/3}-1,
\]

\[
g_k=\frac{Revenue\_TTM\_PIT_{i,t-(3-k)y}}
{Revenue\_TTM\_PIT_{i,t-(4-k)y}}}-1,\quad k=1,2,3,
\qquad
\mathrm{GrowthPersistence}_{i,t}=
\frac{\mathbb{1}(g_1>0)+\mathbb{1}(g_2>0)+\mathbb{1}(g_3>0)}{3}.
\]

The components deliberately remain separate: a high CAGR does not establish
uninterrupted expansion, and a high positive-interval fraction does not state
the magnitude of expansion. No scalar `Growth`, rank, normalization, or
`X_char` column is constructed here.

**Frozen Revenue concepts.** The only eligible tags in this experiment are
`RevenueFromContractWithCustomerExcludingAssessedTax` for AAPL,
`RevenuesNetOfInterestExpense` for JPM, and `Revenues` for XOM, UNH, and WMT.
A taxonomy change is not automatically spliced into a history. The result is
`NA` rather than a fabricated continuous history unless a separate audited
policy is approved. JPM's net-interest revenue remains a banking activity
measure and is only conditionally comparable with the customer-sales concepts
of the other issuers.

`Revenue_TTM_PIT` applies the already frozen compatible-quarter reconstruction:
direct standalone quarter when appropriate; otherwise `Q2 = H1 − Q1`,
`Q3 = 9M − H1`, and `Q4 = FY − 9M`. Every selected operand must have the same
issuer/CIK, Revenue concept, unit, economically compatible period structure,
PIT eligibility, accession provenance, and deterministic vintage selection.
The four annual points must be consecutive within the stated annual tolerance.
All four revenue levels must be positive. If any point is unavailable,
ambiguous, incompatible, non-consecutive, or invalid, both Growth components
are `NA`; the horizon is never shortened or imputed.

**Information policy.** Each historical Revenue TTM is rebuilt at the PIT
eligibility of the first original annual filing for that economic annual end.
Thus a later comparative disclosure cannot rewrite an older information set.
An amendment may supersede a fact only prospectively from its own eligibility
date; it is not used retroactively to alter historical Growth inputs. This
preserves original and amendment accessions, forms, acceptance timestamps, and
quarter construction provenance in the reusable result.

The frozen definition is an auditable input to the later experiment, not a
claim of universal cross-sector comparability or a result selected using
returns, Sharpe, backtests, portfolio outcomes, or geometric results.
Balance-sheet Resilience, a Quality composite, all characteristic composites,
normalization/ranking, `X_char`, `w_raw`, `F_risk`, and portfolio construction
remain unimplemented.

### Stage 1C audit — PIT Price Momentum and Recent Persistence

**Status: PENDING — Momentum is not frozen or implemented.** The economic
concept under audit is strength of price movement plus recent persistence. A
portfolio decision is at the XNYS regular-session open on `t`; every price
quantity is therefore limited to the completed close of the preceding XNYS
session, `t-1`.

**Price representation for this audit: adjusted close.** The preserved Alpha
Vantage daily response has raw close, adjusted close, dividend amount, and
split coefficient for all five frozen issuers. Unlike PIT Market Cap,
Momentum has no reported-share quantity whose split basis must match a raw
price. The audit consequently uses one representation consistently:
`adjusted_close`. This makes the return/path interpretation split- and
cash-dividend-adjusted (total-return-like) rather than raw-price-only. The
AAPL 4-for-1 event dated 2020-08-31 is the explicit sanity check: raw close
falls about 74.15% across the split date while adjusted close rises about
3.39%; the raw result would be a mechanical, not economic, Momentum shock.

This is **not** confirmation of fully historical adjusted-price PIT
availability. Stage 0 observed dated sessions but no observation-level bar
publication time, and raw evidence preserves a current provider adjustment
history rather than dated historical adjustment vintages. Later split/dividend
adjustments normally rescale both endpoints of a prior adjusted-price ratio
and cancel algebraically, but that property is an assumption about the
provider's method. The adjustment-vintage assumption remains a decision
required before a Momentum definition can freeze.

**Trading-session convention under audit.** Let `cutoff` be the final XNYS
session strictly before decision date `t`. The 12M and 1M anchors are
`cutoff - DateOffset(years=1)` and `cutoff - DateOffset(months=1)`, mapped to
the last XNYS session on or before each calendar anchor. No fixed 252- or
21-observation shortcut is assumed; holidays, weekends, early closes and
calendar-month differences are handled through the XNYS calendar. With
adjusted price `P`, the descriptive candidates are:

\[
R_{12M}=P_{cutoff}/P_{start,12M}-1,\qquad
R_{1M}=P_{cutoff}/P_{start,1M}-1,
\]

\[
\mathrm{ProximityToHigh}_{12M}=
P_{cutoff}/\max(P_{start,12M},\ldots,P_{cutoff}).
\]

The diagnostic classical-style `R_12_1` is
`P_start,1M / P_start,12M - 1`; it deliberately omits the most recent calendar
month. Missing required session prices remain `NA`; there is no substitution.

**Observed representative cases (descriptive only).** In the restricted five-
issuer historical sample, AAPL at decision 2007-10-09 has `R_12M` 126.23%,
`R_1M` 27.43%, and proximity 1.00: strong movement and continuation near its
high. XOM at 2007-11-06 has `R_12M` 23.60%, but `R_1M` -4.05% and proximity
0.922: strong medium-term movement with recent deterioration. WMT at
2007-10-09 has `R_12M` -4.59% but `R_1M` 6.79% and proximity 0.889: a recent
rebound despite weak 12M movement. On 2008-12-23 AAPL and UNH have similar
12M returns (-55.78% and -56.29%) yet sharply different 1M returns (3.83% and
50.09%), illustrating why a single cumulative return does not describe the
same current trajectory. These facts are not rankings or future-return tests.

`R_1M` measures local final-month direction. Proximity-to-high measures the
location against the entire 12M path. They can disagree: a rebound after a
deep drawdown may have positive `R_1M` but low proximity; a small decline can
remain near a long-window high. Therefore they are not mechanically redundant.
Classical 12-1 removes exactly that recent-month information and is
**MISALIGNED** as the sole measure for this experiment's explicit interest in
recent continuation/deterioration; this is an economic alignment conclusion,
not an observed-performance comparison.

| Candidate | Audit verdict | Rationale / unresolved limitation |
| --- | --- | --- |
| `R_12M` Momentum magnitude | **CONDITIONAL GO** | Transparent session-aware adjusted-price movement; depends on adjustment-vintage assumption. |
| `R_1M` recent continuation | **CONDITIONAL GO** | Distinct local direction; the one-month horizon is not yet frozen. |
| `ProximityToHigh_12M` path information | **CONDITIONAL GO** | Distinct full-window path location; same adjustment-vintage limitation. |
| Classical `R_12_1` as sole representation | **MISALIGNED** | Deliberately drops the recent movement required by the stated concept. |

Before freezing Momentum, research must decide whether the provider-adjustment
vintage assumption is acceptable and whether persistence is represented by
`R_1M`, proximity-to-high, or a separately justified pair. No arbitrary
weighted cocktail is authorized. Momentum, `X_char`, `w_raw`, `F_risk`, and
portfolio construction remain unimplemented.

### Stage 1C decision — PIT Momentum (supersedes its audit status)

**FROZEN — Momentum as two separate raw adjusted-close components**

\[
\mathrm{MomentumRaw}_{i,t}=
[R_{12M,i,t}^{AdjustedClose}, R_{1M,i,t}^{AdjustedClose}].
\]

`R_12M` is medium-term price/total-return-like movement magnitude and `R_1M`
is recent continuation/deterioration. A positive `R_1M` means recent upward
movement; a negative `R_1M` means recent deterioration. Neither is converted
to a binary flag, ranked, normalized, winsorized, or combined into a scalar
Momentum score.

**Frozen representation and timing.** Both components use Alpha Vantage
`adjusted close` exclusively: raw close is forbidden for Momentum. For an
XNYS regular-session opening decision `t`, `cutoff` is the last eligible XNYS
session strictly before `t`; no same-session close or later observation is
used. The 12M and 1M calendar anchors are respectively `cutoff - 1 calendar
year` and `cutoff - 1 calendar month`, each mapped to the last eligible XNYS
session on or before the anchor. The raw results expose the cutoff/current
price date and adjusted close, both reference dates and adjusted closes, and
both returns. Required missing dates, missing/non-positive prices, or
insufficient 12M/1M history return explicit `NA` statuses; no row-count
shortcut, imputation, or future substitution is allowed.

**Adjusted-price historical-vintage limitation.** The currently preserved
historical adjusted-close series does not evidence exactly which historical
adjustment vintage was available at every past decision. This experiment
explicitly accepts that series as its price-history representation, subject to
that limitation; Momentum therefore does not claim the same strict PIT
provenance as filing-linked SEC fundamentals. Its split treatment remains
economically necessary: the audited AAPL 2020-08-31 4-for-1 split produces a
raw-close movement of about -74.15% but an adjusted-close movement of about
+3.39%, so raw close would introduce a mechanical false Momentum signal.

**Investigated but not selected.** `ProximityToHigh_12M` remains documented in
the preceding audit as informative path-location evidence, but is not a
frozen Momentum component. Classical 12-1 is **REJECTED FOR THIS EXPERIMENT'S
DEFINITION** as the final representation because it deliberately removes the
recent month while this economic definition expressly retains recent
continuation/deterioration. Neither statement claims that 12-1 is generally
inferior, nor is either selection based on portfolio returns, Sharpe,
backtests, or downstream geometry.

The final Stage 1C raw-characteristic inventory is seven components under four
economic characteristics:

| Characteristic | Frozen raw components |
| --- | --- |
| Value | `EarningsYield` |
| Quality | `Profitability`, `CashRealization` |
| Growth | `RevenueCAGR_3Y`, `PositiveAnnualGrowthFraction_3Y` |
| Momentum | `R_12M_AdjustedClose`, `R_1M_AdjustedClose` |

`X_char`, `w_raw`, `F_risk`, risk-subspace construction, projection,
portfolio construction, and backtesting remain unimplemented. Stage 1C closes
the characteristic-specification stage only; it does not authorize Stage 2.
