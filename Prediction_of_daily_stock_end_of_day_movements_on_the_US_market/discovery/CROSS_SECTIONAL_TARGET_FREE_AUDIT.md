# Target-Free Cross-Sectional Information Audit

## Scope and boundary

This audit used only `data/discovery/discovery_input_training.csv`, the
physically materialized target-blind Discovery partition:

```text
days 0--352 x E_dev
472,816 rows; 353 days; 1,463 equities
```

No label file was opened, `reod` was not present in the input schema, no
target was constructed, and no model, SHAP calculation, `E_holdout`,
days 353--502, or competition-test row was accessed.

The challenge documentation establishes that `r0,...,r52` are observed inputs
provided before the end-of-day target period. It contains an endpoint-count
inconsistency, so this audit treats the positions as ordered observed inputs
and does not assign additional exact timestamps.

## Prediction-time information set

For a row `(d, i)`:

- **Row-wise information:** its own observed `r_{d,i,0},...,r_{d,i,52}` and
  availability mask.
- **Contemporaneous cross-sectional information:** for each position `t`, all
  observed `r_{d,j,t}` from other rows on the same `day=d`, available at the
  end of the supplied input path. Same-day aggregation is legitimate only if
  the data feed makes the full contemporaneous input cross-section available
  at prediction time.
- **Invalid information:** `reod`; a later day; another position not yet
  available at the intended prediction time; a statistic pooled over future
  days; an OOS-derived preprocessing parameter; or any competition-test data.

## Target-free panel and coverage facts

Daily row counts range from `1,313` to `1,357` (median `1,340`). Across all
`353 x 53 = 18,709` day-position cells, observed cross-sectional coverage has
minimum `0`, 5th percentile `0.86194`, median `0.88814`, 95th percentile
`0.93861`, and maximum `0.98806`.

There are `44` zero-coverage cells and no unit-coverage cells. The zero cells
are exactly:

| Day | Positions with no observed cross-section |
|---:|---|
| 112 | `r42,...,r52` |
| 134 | `r42,...,r52` |
| 229 | `r42,...,r52` |
| 314 | `r42,...,r52` |

The two other previously documented structurally unusual sessions are outside
the authorized Discovery day range. A same-day median and rank are undefined
for these 44 cells; the audit does not choose an encoding or imputation rule.
Every other day-position cell has at least two observed equities (`18,665`
cells), so both a cross-sectional location and a non-degenerate rank scale are
mathematically evaluable there.

Full per-day/per-position coverage, median, MAD, and leave-one-out summaries
are persisted in:

- `discovery/results/CROSS_SECTIONAL_TARGET_FREE_DAY_POSITION.csv`
- `discovery/results/CROSS_SECTIONAL_TARGET_FREE_POSITION_SUMMARY.csv`
- `discovery/results/CROSS_SECTIONAL_TARGET_FREE_AUDIT.json`

For orientation only, `r0` has the largest mean coverage (`0.94443`); the
lowest mean coverage positions are `r2` (`0.87499`) and the late positions
`r42`--`r51` (approximately `0.876`--`0.880`).

## Minimal taxonomy of new contemporaneous information

For an observed return at same-day position `(d,i,t)`, let `O_{d,t}` be the
observed equities and `n_{d,t}=|O_{d,t}|`.

```text
m_{d,t}       = median_{j in O_{d,t}} r_{d,j,t}
u_{d,i,t}     = r_{d,i,t} - m_{d,t}
rank_{d,i,t}  = average-tie rank(r_{d,i,t} among {r_{d,j,t}: j in O_{d,t}})
pct_{d,i,t}   = (rank_{d,i,t} - 1) / (n_{d,t} - 1), when n_{d,t} >= 2
U_{d,i}       = sum_t u_{d,i,t} over a separately specified observed set
```

These are conceptually distinct from the earlier row-only experiments:

1. `m_{d,t}` is a common contemporaneous movement estimate.
2. `u_{d,i,t}` represents movement relative to that common estimate.
3. `pct_{d,i,t}` is scale-robust relative standing at a position.
4. `U_{d,i}` is an aggregate relative path, not merely an own-path sum.

No predictive representation, observed-position rule for `U`, missingness
treatment, or candidate has been frozen by this audit.

## Leave-one-out self-inclusion audit

For observed `i`, compare `m_{d,t}` with the exact
`m_{d,-i,t}=median_{j in O_{d,t}, j != i} r_{d,j,t}`. For a missing `i`, it
does not enter either statistic. Across `22,285,767` observed-row
comparisons, the mean absolute difference was `0.009038` bps, the maximum was
`0.555` bps, and `31.53%` were nonzero at raw numerical precision.

Thus self-inclusion can change a median numerically but is small relative to
the observed cross-sectional dispersion. A future predictive specification
must choose all-row versus leave-one-out construction before evaluation; this
audit does not select one based on a target.

## Dispersion and rank meaning

The pre-specified robust dispersion is:

```text
MAD_{d,t} = median_{j in O_{d,t}} |r_{d,j,t} - m_{d,t}|
```

Across evaluable day-position cells, MAD ranges from `2.10` to `65.205` bps,
with 5th percentile `3.80`, median `7.17`, and 95th percentile `18.00` bps.
No cell has zero MAD. Therefore relative location and percentile rank are not
mathematically degenerate in the evaluable cells.

This does not establish that dispersion, residuals, or ranks predict any
target; dispersion is not proposed as a feature here.

## Missingness implications

- If equity `i` is missing at `t`, `u_{d,i,t}` and its rank are undefined
  without a separately frozen representation decision.
- Missing peers reduce `n_{d,t}` but do not require imputation to compute a
  median, MAD, or rank among observed peers.
- At zero-coverage day-position cells, all cross-sectional statistics are
  undefined. The four cell groups above show why this case cannot be silently
  represented as a numerical median, residual, or rank.
- Availability itself can be cross-sectionally shared; it must remain
  distinct from return-relative information.

## Synthetic same-day examples

| Day | Equity own cumulative input | Common cumulative movement | Relative cumulative movement |
|---|---:|---:|---:|
| A | +35 bps | +30 bps | about +5 bps |
| B | +35 bps | -10 bps | about +45 bps |

The same absolute own-path movement can thus have different relative states.
Likewise, a high absolute return can receive a middle percentile when many
equities moved similarly, or an extreme percentile when peers did not. These
examples clarify information distinctions only; they make no predictive claim.

## Relationship to EXP_003--EXP_007

| Experiment | Row-wise path information | Same-day cross-sectional information | Is `m/u/rank` new? |
|---|---|---|---|
| EXP_003 | cumulative observed own return `R_obs` | none | yes |
| EXP_004 | own positive/negative/zero return counts summarized by `P` | none | yes |
| EXP_005 | own 53 returns plus own masks | none | yes |
| EXP_006 | same own positional path/masks with fixed HGB | none | yes |
| EXP_007 | own 53 return/mask pairs under real/permuted order | none | yes |

The audited relative quantities require grouping contemporaneous rows by
`day`; no cited experiment code performs such a group aggregation.

## Leakage and validation implications

Same-day aggregation is not automatically leakage: it can be part of the
information set if all contemporaneous `r_t` inputs are available when the
row prediction is issued. It becomes invalid if data delivery is asynchronous
or if the intended prediction is made before all peers' relevant `r_t` values
are available; the repository contains no evidence resolving that operational
feed assumption.

The following remain prohibited:

- aggregating a future day;
- using a future return position;
- using `reod` in an aggregate;
- fitting a cross-day learned parameter on OOS rows; or
- pooling Fit and OOS days in a learned preprocessing step.

Cross-sectional features must be computed separately within each day. It is
legitimate to compute them on an OOS day at inference only under the stated
same-day availability assumption, because no future day or target enters the
calculation. Rows within a day are coupled by construction and are not IID;
row-level standard errors or naive independent-row interpretations would be
inappropriate. This does not require redesigning the frozen temporal/equity
validation architecture.

## Candidate questions generated, not frozen

1. Does an equity's cumulative relative observed path add conditional-sign
   information beyond its own cumulative path when contemporaneous inputs are
   available?
2. Does a positionwise relative-state representation add information beyond
   own-path positional returns and masks?
3. Does a scale-robust contemporaneous percentile state differ from a
   median-residual representation under a future, separately frozen test?

## Target-free recommendation

The single most defensible next Discovery question is:

> Using a pre-specified same-day common component, does an equity's
> contemporaneous **relative observed path** contain conditional-sign
> information not represented by its own intraday path?

It is structurally new relative to EXP_003--EXP_007, uses no target-based
selection, and is motivated by the non-degenerate same-day coverage and MAD
audit. It remains contingent on resolving or explicitly freezing the
same-day-data-availability assumption and missing-position representation
before any target-conditioned work.
