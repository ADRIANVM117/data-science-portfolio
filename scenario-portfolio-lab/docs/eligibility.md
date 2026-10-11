# Eligibility boundary

An instrument must have independently verified identity, Alpha Vantage symbol, primary listing market, USD quotation currency, Gold quality approval, valid cut-off coverage, and sufficient common history. ETF records additionally require a current official-source record, verification date, `direction=BULL`, `leverage_multiple=1.0`, and `etf_structure_state=CONFIRMED_DIRECT_UNLEVERAGED`.

Inverse, leveraged, leveraged-inverse, unknown, contradictory, or unverified ETF structures are excluded or remain `REVIEW`. Eligibility is a data and structural gate; it is not an economic portfolio selection or recommendation.

Reasons are intentionally distinct: `IDENTITY_REVIEW`, `MARKET_OR_CURRENCY_REVIEW`, and `ETF_STRUCTURE_REVIEW` are unresolved verification states; `ETF_NOT_DIRECT_UNLEVERAGED` is a structural exclusion; and the accompanying coverage report preserves the concrete data-quality/coverage failure (`INTERIOR_NULL`, `MISSING_CUTOFF`, or `INSUFFICIENT_HISTORY`).
