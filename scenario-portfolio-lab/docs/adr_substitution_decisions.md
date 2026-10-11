# Pending decisions: local instruments / ADS

This UTF-8 document deliberately uses ASCII text to avoid terminal code-page corruption.

| Local catalog instrument | Proposed USD instrument | Relationship and ratio | Relevant differences | Proposed state |
|---|---|---|---|---|
| ASUR B (BMV) | ASR (NYSE) | ASUR 20-F identifies ASUR B Series B shares in Mexico and ASR ADS; 1 ADS = 10 Series B shares. | USD ADS, depositary mechanics, rights and price do not mechanically equal the MXN local share. | REVIEW: explicit decision required. |
| CEMEX CPO (BMV) | CX (NYSE) | CEMEX 20-F identifies CEMEXCPO in BMV and CX ADS in NYSE; 1 ADS = 10 CPO; each CPO represents 2 Series A and 1 Series B share. | Underlying is a CPO certificate, not a simple ordinary share; USD ADS adds depositary structure. | REVIEW: explicit decision required. |
| GAP B (BMV) | PAC (NYSE) | Issuer documentation identifies GAP B and PAC ADS; 1 ADS = 10 Series B shares. | USD ADS, depositary mechanics, rights and economic exposure do not automatically substitute the BMV line. | REVIEW: explicit decision required. |

Fuentes primarias consultadas el 2026-10-11: [ASUR 20-F](https://www.sec.gov/Archives/edgar/data/1123452/000141057825000689/asurb-20241231x20f.htm), [CEMEX 20-F](https://www.sec.gov/Archives/edgar/data/1076378/000119312526177605/d120395d20f.htm), [GAP inversionistas](https://www.aeropuertosgap.com.mx/es/investors/36-inversionistas.html).

If a substitution is approved, the catalog will retain the local `reference_symbol`, use the ADS as `provider_symbol_candidate`, retain `mapping_type=ADR_PROXY`, and add the source URL, ratio/class and dated decision. The local identity will never be overwritten.
