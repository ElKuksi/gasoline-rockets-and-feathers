# Rockets and Feathers: Do Pump Prices Rise Faster Than They Fall?

> *Work in progress — sections are filled in as the analysis proceeds.*

## The question
Do US retail gasoline prices respond **faster to crude-oil increases than to decreases** — the "rockets and feathers" asymmetry? And how big is the effect, including around the 2026 crude spike?

## Approach
_(fill in: data → lag analysis → distributed-lag asymmetry model → asymmetric error-correction model → 2026 event study → regional comparison)_

## Key finding
_(fill in — the headline number, honestly stated)_

## Data

**Source**: [FRED](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis). Crude and wholesale-spot series are FRED's own; retail gasoline/diesel series are EIA (U.S. Energy Information Administration) data, redistributed through FRED — no direct EIA API call is made (see `decisions.md`).

**Licence**: Both FRED and the EIA publish their data as **"Public Domain: Citation Requested"** — free to use and redistribute, with attribution requested but not legally required.

**Coverage**: 2010-01-01 → present, refreshed on each run of `src/fetch_series.py`.

**Frequency, by level**:
- **Crude** — daily and weekly (Friday-ending)
- **Wholesale spot** — daily and weekly (Friday-ending)
- **Retail** — weekly only (Monday-ending; an 8am Monday survey snapshot — no daily equivalent exists)

Full series catalogue: `data/processed/series_manifest.csv` (generated from `src/series_manifest.py`).

## Reproduce
_(fill in: `.env` with API keys → `pip install -r requirements.txt` → one command)_

## Limitations
_(fill in: gas taxes, summer-blend seasonality, refinery outages, ethanol — confounders not fully controlled)_

## How this transfers
Asymmetric price pass-through is exactly how you'd analyze pricing response in any marketplace — surge pricing, fee changes, competitor moves.
