# MSME Financial Health Card

**IDBI Innovate 2026 — Track 03: Financial Inclusion / Digital Lending / Credit Decisioning**

An AI/ML-driven credit scoring prototype that helps New-to-Credit (NTC) and
New-to-Bank (NTB) MSMEs get evaluated fairly using alternate data (GST, UPI,
Account Aggregator, EPFO, utility consumption) instead of traditional
financial documents alone.

## The idea

Instead of a single opaque credit score, this outputs a **multidimensional
financial health score with an explicit confidence band** — so a thin-file
business isn't rejected outright, but is honestly flagged as "promising but
low-confidence" rather than scored as if it had years of data behind it.

Four core mechanisms:
1. **Data Maturity Index** — each score comes with a confidence band that
   narrows as more months of real data accumulate.
2. **Promoter–Business Dual Score** — `Final = α(age)×Business + (1-α(age))×Promoter`,
   so a new business with a financially stable owner is still bankable.
3. **Sector-conditioned features** — utility consumption, margins, and
   revenue-per-employee are benchmarked against sector-specific norms, not
   absolute thresholds.
4. **Explainable radar visualization** — every score is broken into 6
   inspectable sub-components, not a black-box number.

## Try it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Two modes:
- **Browse existing MSMEs** — 60 synthetic profiles across 11 sectors and
  4 risk personas. Try MSME003 (thin-file, risky) vs MSME036 (established,
  stable) to see the contrast. Drag the "months of data available" slider
  to watch the confidence band shrink independent of the score.
- **Build your own MSME** — construct a business from scratch with sliders,
  including a "no CIBIL history" toggle, to test the cold-start handling
  live.

## Data note

This prototype runs on **synthetic data** generated to mirror the schema of
real GST/UPI/AA/EPFO/utility data (see `generate_data.py`), since IDBI's
official sandbox datasets are only available to shortlisted teams. Swapping
to live data requires changing only `data_loader.py` — the scoring engine
(`scoring.py`) and UI (`app.py`) are already built against the same schema
and require no changes.

## Files

| File | Purpose |
|---|---|
| `generate_data.py` | Synthetic MSME dataset generator (sector + persona conditioned) |
| `synthetic_msme_data.csv` | 60 generated synthetic MSME profiles |
| `scoring.py` | Core scoring engine (business/promoter/blend/confidence) |
| `data_loader.py` | Data source abstraction — only file to change for real APIs |
| `app.py` | Streamlit interactive prototype |
| `financial_health_cards.png` | Static snapshot for presentation |

## Roadmap

- **Phase 1 (this prototype):** explainable, rule-based scoring on synthetic
  data.
- **Phase 2 (sandbox access):** calibrate weights against IDBI's real
  synthetic datasets; integrate with ULI/OCEN/AA rails.
- **Phase 3 (production):** train a supervised ML model (e.g. XGBoost with
  SHAP) once real repayment/default outcome labels exist; upgrade the
  "confidence band" from a heuristic to a calibrated prediction interval.
  Introduce the "sachet-to-scale" dynamic credit line that recomputes
  monthly as more data accrues.