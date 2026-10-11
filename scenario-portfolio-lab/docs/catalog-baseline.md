# Catalog baseline

`config/universe.csv` is an independent, sanitized public catalog derived from the complete initial reference list. It contains 202 records: 146 equities and 56 ETFs. There are 202 unique reference symbols and 156 non-empty provider-symbol candidates, with no duplicate non-empty candidate symbols. The remaining 46 records have no provider candidate.

Source-state counts retained as context, not as public verification: 56 `CONFIRMED`, 97 `CANDIDATE`, 3 `ADR_PROXY_CANDIDATE`, 4 `SYMBOL_COLLISION`, 38 `LOCAL_ONLY_PENDING`, and 4 explicit source exclusions. The public catalog does not promote any of them: all 202 begin with `provider_mapping_state=PENDING_PROVIDER_VERIFICATION` and `identity_verification_state=PENDING`.

Provider availability is not evidence of USD quotation, market, issuer identity, or economic equivalence. A local share, ADR, and a distinct share class must be reviewed as separate instruments. The real verification record must add public evidence for identity, primary market, USD currency, and, for ETFs, official structure/direction/leverage evidence.
