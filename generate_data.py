"""
Synthetic MSME dataset generator for the Financial Health Card prototype.
Schema follows the fields identified in deep-research-report.md.
Produces sector-conditioned, persona-conditioned synthetic MSME profiles.
No real/proprietary data used anywhere.
"""

import numpy as np
import pandas as pd
import json

rng = np.random.default_rng(42)

# ---- Sector benchmark table (indicative ranges, from research report) ----
# electricity: kWh per lakh turnover ; water: kL per employee ; revenue/employee in lakh ; profit margin range
SECTOR_BENCHMARKS = {
    "Textile Garments":    {"elec": (2.0, 3.0),  "water": (50, 100), "rev_emp": (5, 8),   "margin": (0.08, 0.12), "gas": (1.5, 3.0)},
    "Food Processing":     {"elec": (2.5, 4.0),  "water": (100, 150),"rev_emp": (6, 10),  "margin": (0.07, 0.15), "gas": (2.0, 4.0)},
    "Chemicals":           {"elec": (3.0, 5.0),  "water": (10, 50),  "rev_emp": (15, 25), "margin": (0.05, 0.12), "gas": (4.0, 6.0)},
    "Iron & Steel":        {"elec": (4.0, 6.0),  "water": (10, 20),  "rev_emp": (20, 30), "margin": (0.05, 0.10), "gas": (6.0, 8.0)},
    "Engineering Mfg.":    {"elec": (3.0, 5.0),  "water": (20, 50),  "rev_emp": (10, 15), "margin": (0.07, 0.12), "gas": (2.0, 4.0)},
    "Electronics":         {"elec": (3.0, 4.5),  "water": (5, 15),   "rev_emp": (8, 12),  "margin": (0.06, 0.10), "gas": (1.0, 2.5)},
    "Plastics & Polymers": {"elec": (2.5, 4.0),  "water": (5, 20),   "rev_emp": (10, 15), "margin": (0.05, 0.12), "gas": (3.5, 5.0)},
    "Paper & Packaging":   {"elec": (4.0, 6.0),  "water": (20, 40),  "rev_emp": (12, 18), "margin": (0.06, 0.10), "gas": (2.5, 4.0)},
    "Leather Goods":       {"elec": (2.0, 3.5),  "water": (20, 40),  "rev_emp": (5, 8),   "margin": (0.10, 0.15), "gas": (2.0, 3.5)},
    "Retail/Trade":        {"elec": (0.5, 1.0),  "water": (30, 60),  "rev_emp": (3, 5),   "margin": (0.15, 0.20), "gas": (0.0, 0.0)},
    "IT/Services":         {"elec": (0.3, 0.8),  "water": (5, 15),   "rev_emp": (8, 20),  "margin": (0.15, 0.25), "gas": (0.0, 0.0)},
}

SECTORS = list(SECTOR_BENCHMARKS.keys())

# Persona definitions control how "bankable" a synthetic MSME is
PERSONAS = {
    "thin_file_good":     dict(weight=0.30, years_range=(0, 1.5), gst_ontime=(0.85, 1.0), cashflow_vol=(0.05, 0.15), promoter_cibil=(700, 800), bounce_rate=(0.0, 0.02)),
    "thin_file_risky":    dict(weight=0.20, years_range=(0, 1.5), gst_ontime=(0.4, 0.7),  cashflow_vol=(0.30, 0.55), promoter_cibil=(550, 680), bounce_rate=(0.05, 0.15)),
    "established_stable": dict(weight=0.30, years_range=(3, 15), gst_ontime=(0.85, 1.0), cashflow_vol=(0.05, 0.20), promoter_cibil=(680, 800), bounce_rate=(0.0, 0.03)),
    "established_declining": dict(weight=0.20, years_range=(3, 15), gst_ontime=(0.5, 0.8), cashflow_vol=(0.25, 0.45), promoter_cibil=(600, 720), bounce_rate=(0.04, 0.12)),
}

N = 60

def sample_range(r):
    return rng.uniform(r[0], r[1])

def gen_monthly_cashflow(months, base, volatility, seasonal=True):
    t = np.arange(months)
    trend = base * (1 + rng.uniform(-0.02, 0.03)) ** t
    seasonality = 1 + (0.15 * np.sin(2 * np.pi * t / 12) if seasonal else 0)
    noise = rng.normal(1.0, volatility, months)
    series = trend * seasonality * noise
    return np.round(series, 0)

rows = []
persona_names = list(PERSONAS.keys())
persona_weights = [PERSONAS[p]["weight"] for p in persona_names]

for i in range(N):
    persona_name = rng.choice(persona_names, p=persona_weights)
    persona = PERSONAS[persona_name]
    sector = rng.choice(SECTORS)
    bench = SECTOR_BENCHMARKS[sector]

    years_in_business = round(sample_range(persona["years_range"]), 1)
    data_history_months = int(min(years_in_business * 12, rng.uniform(3, 36)))
    data_history_months = max(data_history_months, 2)

    annual_turnover_lakh = round(rng.uniform(20, 300) * (1.3 if "established" in persona_name else 1.0), 1)
    margin = round(sample_range(bench["margin"]), 3)
    net_profit_lakh = round(annual_turnover_lakh * margin, 2)

    employees = max(1, int(rng.uniform(1, 25) * (annual_turnover_lakh / max(*bench["rev_emp"], 1))**0 * (annual_turnover_lakh / 50)))
    employees = max(1, min(employees, 60))

    elec_intensity = sample_range(bench["elec"])
    water_intensity = sample_range(bench["water"])
    gas_intensity = sample_range(bench["gas"])

    utility_kwh_year = round(elec_intensity * annual_turnover_lakh * 12, 0)
    utility_water_year = round(water_intensity * employees, 0)
    utility_gas_year = round(gas_intensity * annual_turnover_lakh * 12, 0)

    cashflow_vol = round(sample_range(persona["cashflow_vol"]), 3)
    monthly_avg = annual_turnover_lakh * 100000 / 12  # convert lakh to rupees /12
    bank_balance_monthly = gen_monthly_cashflow(min(data_history_months, 12), monthly_avg * 0.3, cashflow_vol).tolist()

    gst_ontime_rate = round(sample_range(persona["gst_ontime"]), 2)
    gst_return_flag = 1 if rng.random() < gst_ontime_rate else 0

    bounce_rate = round(sample_range(persona["bounce_rate"]), 3)

    promoter_age = int(rng.uniform(24, 58))
    promoter_cibil = int(sample_range(persona["promoter_cibil"])) if data_history_months >= 6 or "established" in persona_name else None
    promoter_income_lakh = round(rng.uniform(3, 18), 1)
    outstanding_loans = round(rng.uniform(0, annual_turnover_lakh * 0.3) * 100000, 0)

    rows.append({
        "business_id": f"MSME{i+1:03d}",
        "sector": sector,
        "persona": persona_name,
        "years_in_business": years_in_business,
        "data_history_months": data_history_months,
        "annual_turnover_lakh": annual_turnover_lakh,
        "net_profit_margin": margin,
        "net_profit_lakh": net_profit_lakh,
        "employees": employees,
        "gst_return_flag": gst_return_flag,
        "gst_ontime_rate": gst_ontime_rate,
        "cashflow_volatility": cashflow_vol,
        "bank_balance_monthly": json.dumps(bank_balance_monthly),
        "bounce_rate": bounce_rate,
        "utility_kwh_year": utility_kwh_year,
        "utility_water_kl_year": utility_water_year,
        "utility_gas_year": utility_gas_year,
        "promoter_age": promoter_age,
        "promoter_cibil": promoter_cibil,
        "promoter_income_lakh": promoter_income_lakh,
        "outstanding_loans_rupee": outstanding_loans,
    })

df = pd.DataFrame(rows)
df.to_csv("synthetic_msme_data.csv", index=False)
print(df.shape)
print(df.head(10).to_string())
print("\nPersona distribution:\n", df["persona"].value_counts())
print("\nSector distribution:\n", df["sector"].value_counts())
