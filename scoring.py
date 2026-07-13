"""
Financial Health Card scoring engine.
Implements:
  - Data Maturity Index (confidence weighting)
  - Business Score (sector-normalized)
  - Promoter Score
  - Age-weighted blend: Final = alpha(age)*Business + (1-alpha(age))*Promoter
  - Confidence band width driven by maturity
"""

import json
import numpy as np
import pandas as pd

# elec: kWh/lakh-turnover/month equiv ; water: kL/employee/year ; gas: units/lakh-turnover/year
# rev_emp: revenue (lakh) per employee ; margin: typical net profit margin range
SECTOR_BENCHMARKS = {
    "Textile Garments":    {"elec": (2.0, 3.0),  "water": (50, 100),  "gas": (1.5, 3.0), "rev_emp": (5, 8),   "margin": (0.08, 0.12)},
    "Food Processing":     {"elec": (2.5, 4.0),  "water": (100, 150), "gas": (2.0, 4.0), "rev_emp": (6, 10),  "margin": (0.07, 0.15)},
    "Chemicals":           {"elec": (3.0, 5.0),  "water": (10, 50),   "gas": (4.0, 6.0), "rev_emp": (15, 25), "margin": (0.05, 0.12)},
    "Iron & Steel":        {"elec": (4.0, 6.0),  "water": (10, 20),   "gas": (6.0, 8.0), "rev_emp": (20, 30), "margin": (0.05, 0.10)},
    "Engineering Mfg.":    {"elec": (3.0, 5.0),  "water": (20, 50),   "gas": (2.0, 4.0), "rev_emp": (10, 15), "margin": (0.07, 0.12)},
    "Electronics":         {"elec": (3.0, 4.5),  "water": (5, 15),    "gas": (1.0, 2.5), "rev_emp": (8, 12),  "margin": (0.06, 0.10)},
    "Plastics & Polymers": {"elec": (2.5, 4.0),  "water": (5, 20),    "gas": (3.5, 5.0), "rev_emp": (10, 15), "margin": (0.05, 0.12)},
    "Paper & Packaging":   {"elec": (4.0, 6.0),  "water": (20, 40),   "gas": (2.5, 4.0), "rev_emp": (12, 18), "margin": (0.06, 0.10)},
    "Leather Goods":       {"elec": (2.0, 3.5),  "water": (20, 40),   "gas": (2.0, 3.5), "rev_emp": (5, 8),   "margin": (0.10, 0.15)},
    "Retail/Trade":        {"elec": (0.5, 1.0),  "water": (30, 60),   "gas": (0.0, 0.0), "rev_emp": (3, 5),   "margin": (0.15, 0.20)},
    "IT/Services":         {"elec": (0.3, 0.8),  "water": (5, 15),    "gas": (0.0, 0.0), "rev_emp": (8, 20),  "margin": (0.15, 0.25)},
}


def _mid(rng):
    return float(np.mean(rng))


def _safe_get(row, key, default=None):
    """Handles both dict and pandas Series rows, and missing/NaN values."""
    val = row.get(key, default) if hasattr(row, "get") else row[key]
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def _parse_bank_series(row):
    """Returns a numpy array of the monthly bank balance series, or None if unavailable."""
    raw = _safe_get(row, "bank_balance_monthly")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            arr = np.array(json.loads(raw), dtype=float)
        else:
            arr = np.array(raw, dtype=float)
        return arr if len(arr) >= 3 else None
    except (ValueError, TypeError):
        return None


def data_maturity_index(data_history_months, cap_months=12):
    """Confidence in the data backing the score. 0 to 1."""
    return float(min(1.0, data_history_months / cap_months))


def business_score(row):
    """0-100. Higher is better. Uses 7 sub-components across GST, cash flow,
    profitability, utility (electricity+water+gas), liquidity, growth, and leverage."""
    bench = SECTOR_BENCHMARKS[row["sector"]]
    sector_mid_margin = _mid(bench["margin"])
    sector_mid_elec = _mid(bench["elec"])
    sector_mid_water = _mid(bench["water"])
    sector_mid_gas = _mid(bench["gas"])
    sector_mid_rev_emp = _mid(bench["rev_emp"])

    turnover_lakh = max(row["annual_turnover_lakh"], 1)
    employees = max(_safe_get(row, "employees", 1) or 1, 1)

    # 1. Compliance: on-time filing rate blended with whether returns were complete
    gst_return_flag = _safe_get(row, "gst_return_flag", 1)
    compliance = 0.7 * row["gst_ontime_rate"] * 100 + 0.3 * float(gst_return_flag) * 100

    # 2. Cash-flow stability + trend: derived from the RAW monthly bank balance
    #    series when available (real feature engineering), falling back to the
    #    pre-summarized cashflow_volatility column otherwise.
    series = _parse_bank_series(row)
    if series is not None and series.mean() > 0:
        measured_cv = float(series.std() / series.mean())
        t = np.arange(len(series))
        slope = float(np.polyfit(t, series, 1)[0])
        trend_rate = slope / series.mean()  # per-month growth rate, roughly
    else:
        measured_cv = float(_safe_get(row, "cashflow_volatility", 0.2))
        trend_rate = float(_safe_get(row, "manual_trend_rate", 0.0))
    cashflow_stability = float(np.clip(100 - measured_cv * 200, 0, 100))

    # 3. Profitability: sector-relative margin
    margin_ratio = row["net_profit_margin"] / sector_mid_margin if sector_mid_margin else 1
    profitability = float(np.clip(margin_ratio * 60, 0, 100))

    # 4. Utility consistency: electricity + water + gas, each checked against
    #    sector-typical intensity. Sectors with ~0 gas/water use (e.g. IT/Retail)
    #    are auto-excluded and weights renormalized.
    actual_elec = (row["utility_kwh_year"] / 12) / turnover_lakh
    elec_dev = abs(actual_elec - sector_mid_elec) / max(sector_mid_elec, 0.1)
    elec_score = float(np.clip(100 - elec_dev * 40, 0, 100))

    utility_parts = [(elec_score, 0.5)]
    if sector_mid_water > 0:
        actual_water = _safe_get(row, "utility_water_kl_year", sector_mid_water * employees) / employees
        water_dev = abs(actual_water - sector_mid_water) / max(sector_mid_water, 0.1)
        utility_parts.append((float(np.clip(100 - water_dev * 40, 0, 100)), 0.25))
    if sector_mid_gas > 0:
        actual_gas = (_safe_get(row, "utility_gas_year", sector_mid_gas * turnover_lakh * 12) / 12) / turnover_lakh
        gas_dev = abs(actual_gas - sector_mid_gas) / max(sector_mid_gas, 0.1)
        utility_parts.append((float(np.clip(100 - gas_dev * 40, 0, 100)), 0.25))
    total_w = sum(w for _, w in utility_parts)
    utility_consistency = sum(s * w for s, w in utility_parts) / total_w

    # 5. Liquidity: transaction bounce rate
    liquidity = float(np.clip(100 - row["bounce_rate"] * 500, 0, 100))

    # 6. Growth / formalization: staffing efficiency (EPFO-style proxy via
    #    revenue-per-employee vs sector norm) blended with the real cash-flow trend
    rev_per_emp = turnover_lakh / employees
    staffing_dev = abs(rev_per_emp - sector_mid_rev_emp) / max(sector_mid_rev_emp, 0.1)
    staffing_efficiency = float(np.clip(100 - staffing_dev * 40, 0, 100))
    trend_score = float(np.clip(50 + trend_rate * 300, 0, 100))
    growth = 0.5 * staffing_efficiency + 0.5 * trend_score

    # 7. Leverage: existing debt relative to turnover
    outstanding = _safe_get(row, "outstanding_loans_rupee", 0) or 0
    debt_to_turnover = outstanding / (turnover_lakh * 100000)
    leverage = float(np.clip(100 - debt_to_turnover * 150, 0, 100))

    weights = dict(compliance=0.18, cashflow_stability=0.20, profitability=0.12,
                    utility_consistency=0.12, liquidity=0.12, growth=0.13, leverage=0.13)
    components = dict(compliance=compliance, cashflow_stability=cashflow_stability,
                       profitability=profitability, utility_consistency=utility_consistency,
                       liquidity=liquidity, growth=growth, leverage=leverage)
    score = sum(components[k] * weights[k] for k in weights)

    return round(score, 1), {k: round(v, 1) for k, v in components.items()}


def promoter_score(row):
    """0-100. Higher is better. Handles missing CIBIL (thin-file promoter)."""
    if row["promoter_cibil"] is not None and not (isinstance(row["promoter_cibil"], float) and np.isnan(row["promoter_cibil"])):
        cibil_component = float(np.clip((row["promoter_cibil"] - 300) / (900 - 300) * 100, 0, 100))
        cibil_confidence = 1.0
    else:
        cibil_component = 50.0  # neutral prior when no bureau footprint exists
        cibil_confidence = 0.4

    income_component = float(np.clip(row["promoter_income_lakh"] / 20 * 100, 0, 100))

    experience_years = max(row["promoter_age"] - 22, 0)
    experience_component = float(np.clip(experience_years / 30 * 100, 0, 100))

    score = 0.5 * cibil_component + 0.3 * income_component + 0.2 * experience_component
    return round(score, 1), cibil_confidence, dict(
        cibil_component=round(cibil_component, 1),
        income_component=round(income_component, 1),
        experience_component=round(experience_component, 1),
    )


def alpha_from_age(years_in_business, saturate_years=5.0):
    """Weight given to Business Score as firm ages. New firms lean on promoter."""
    return float(np.clip(years_in_business / saturate_years, 0.15, 0.85))


def blend_final_score(b_score, p_score, alpha, gate_threshold=30, gate_margin=20):
    """
    Combines business and promoter scores, then applies a hard gate: if the
    business score itself is catastrophic (below gate_threshold), the promoter
    cannot fully mask it. This reflects real underwriting practice — promoter
    strength should compensate for MISSING business data (cold start), not
    override business data that IS present and shows genuine distress.
    """
    final = alpha * b_score + (1 - alpha) * p_score
    if b_score < gate_threshold:
        final = min(final, b_score + gate_margin)
    return round(final, 1)


def confidence_band(maturity_index, promoter_cibil_confidence):
    """Wider band = less confidence. Returns +/- points on a 0-100 score scale."""
    combined_confidence = 0.7 * maturity_index + 0.3 * promoter_cibil_confidence
    band = 25 * (1 - combined_confidence) + 3  # ranges roughly 3 to 28 points
    label = "High" if combined_confidence > 0.75 else ("Medium" if combined_confidence > 0.45 else "Low")
    return round(band, 1), label, round(combined_confidence, 2)


def score_business(row):
    maturity = data_maturity_index(row["data_history_months"])
    b_score, b_components = business_score(row)
    p_score, cibil_conf, p_components = promoter_score(row)
    alpha = alpha_from_age(row["years_in_business"])
    final = blend_final_score(b_score, p_score, alpha)
    band, conf_label, combined_conf = confidence_band(maturity, cibil_conf)

    return {
        "business_id": row["business_id"],
        "sector": row["sector"],
        "persona": row["persona"],
        "data_maturity_index": round(maturity, 2),
        "business_score": b_score,
        "promoter_score": p_score,
        "alpha_business_weight": round(alpha, 2),
        "final_health_score": round(final, 1),
        "confidence_band": band,
        "confidence_label": conf_label,
        "combined_confidence": combined_conf,
        **{f"biz_{k}": v for k, v in b_components.items()},
        **{f"prom_{k}": v for k, v in p_components.items()},
    }


def apply_hard_gates(final_score: float, b_components: dict):
    """Catastrophic business fundamentals should cap the score regardless of
    promoter strength -- a blend alone can let a strong promoter mask a
    business that's severely over-leveraged or has near-zero compliance.
    Returns (possibly-capped score, list of triggered flags)."""
    flags = []
    if b_components["leverage"] <= 10:
        flags.append("Severe over-leverage — outstanding debt far exceeds turnover")
    if b_components["compliance"] <= 20:
        flags.append("Very poor GST compliance history")
    if b_components["cashflow_stability"] <= 15:
        flags.append("Highly unstable cash flow")

    if flags:
        return round(min(final_score, 40.0), 1), flags
    return round(final_score, 1), flags


def project_trajectory(row: dict, months_ahead: int = 6):
    """Projects the score forward assuming the current cash-flow trend
    continues linearly. Naive by design -- genuinely seasonal businesses
    need a full 12-month cycle before a linear trend is trustworthy, so
    this is meant to be read as an illustrative trajectory, not a forecast."""
    series = _parse_bank_series(row)
    if series is not None and series.mean() > 0:
        t = np.arange(len(series))
        slope = float(np.polyfit(t, series, 1)[0])
        trend = slope / series.mean()
    else:
        trend = float(_safe_get(row, "manual_trend_rate", 0.0))

    trajectory = []
    for m in range(0, months_ahead + 1):
        proj_row = dict(row)
        proj_row["data_history_months"] = row["data_history_months"] + m
        proj_row["annual_turnover_lakh"] = row["annual_turnover_lakh"] * (1 + trend) ** m

        b_score, b_components = business_score(proj_row)
        p_score, cibil_conf, _ = promoter_score(proj_row)
        alpha = alpha_from_age(row["years_in_business"] + m / 12)
        final = alpha * b_score + (1 - alpha) * p_score
        final, flags = apply_hard_gates(final, b_components)
        maturity = data_maturity_index(proj_row["data_history_months"])
        band, label, _ = confidence_band(maturity, cibil_conf)
        trajectory.append(dict(month=m, score=final, band=band, label=label, flagged=bool(flags)))
    return trajectory, trend


COMPONENT_LABELS = {
    "compliance": "GST compliance",
    "cashflow_stability": "Cash flow stability",
    "profitability": "Profitability (sector-adjusted)",
    "utility_consistency": "Utility consistency",
    "liquidity": "Liquidity (low bounce rate)",
    "growth": "Growth & formalization",
    "leverage": "Low leverage (debt load)",
    "promoter_strength": "Promoter strength",
}


def strengths_and_watchouts(b_components: dict, p_score: float, n: int = 3):
    """Returns (strengths, watchouts) each a list of (label, score) tuples,
    ranked from all 8 scored dimensions. Max n each, no overlap since 2n < 8."""
    all_scores = dict(b_components)
    all_scores["promoter_strength"] = p_score
    ranked = sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)
    strengths = [(COMPONENT_LABELS[k], v) for k, v in ranked[:n]]
    watchouts = [(COMPONENT_LABELS[k], v) for k, v in ranked[-n:][::-1]]
    return strengths, watchouts


if __name__ == "__main__":
    df = pd.read_csv("synthetic_msme_data.csv")
    results = [score_business(row) for _, row in df.iterrows()]
    out = pd.DataFrame(results)
    out.to_csv("scored_msme_data.csv", index=False)

    print(out[["business_id", "sector", "persona", "data_maturity_index",
               "business_score", "promoter_score", "alpha_business_weight",
               "final_health_score", "confidence_band", "confidence_label"]].to_string())

    print("\nScore distribution by persona:")
    print(out.groupby("persona")["final_health_score"].agg(["mean", "std", "count"]).round(1))
