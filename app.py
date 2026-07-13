"""
MSME Financial Health Card — interactive prototype.

Run with: streamlit run app.py
"""

import json
import streamlit as st
import plotly.graph_objects as go

import data_loader
from scoring import (
    SECTOR_BENCHMARKS,
    data_maturity_index,
    business_score,
    promoter_score,
    alpha_from_age,
    blend_final_score,
    confidence_band,
    strengths_and_watchouts,
)

st.set_page_config(page_title="MSME Financial Health Card", layout="wide")

AXES = ["Compliance", "Cash flow stability", "Profitability", "Utility consistency",
        "Liquidity", "Growth", "Leverage", "Promoter strength"]


def render_card(row: dict, maturity_override=None, key=""):
    """Runs the live scoring engine on a row dict and renders score + radar + breakdown."""
    b_score, b_components = business_score(row)
    p_score, cibil_conf, p_components = promoter_score(row)
    alpha = alpha_from_age(row["years_in_business"])
    final = blend_final_score(b_score, p_score, alpha)

    months = maturity_override if maturity_override is not None else row["data_history_months"]
    maturity = data_maturity_index(months)
    band, conf_label, combined_conf = confidence_band(maturity, cibil_conf)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        badge_color = {"High": "green", "Medium": "orange", "Low": "red"}[conf_label]
        st.metric("Financial health score", f"{final:.0f} / 100", delta=None)
        st.markdown(f"**Confidence:** :{badge_color}[{conf_label}]  (±{band:.0f} points, based on {months} months of data)")

        if final >= 70 and conf_label == "High":
            verdict = "Bankable — high confidence. Standard underwriting."
        elif final >= 70 and conf_label != "High":
            verdict = "Promising, but thin file — recommend small-ticket pilot exposure."
        elif final >= 50:
            verdict = "Borderline — manual review recommended."
        else:
            verdict = "High risk — recommend decline or heavy collateral."
        st.info(verdict)

        st.markdown("##### Why this score")
        st.markdown(f"`Final = α×Business + (1-α)×Promoter`")
        st.markdown(f"- Business score: **{b_score}**")
        st.markdown(f"- Promoter score: **{p_score}**")
        st.markdown(f"- α (business weight, from firm age): **{alpha}**")

        with st.expander("Raw sub-component breakdown"):
            st.json({"business": b_components, "promoter": p_components})

        st.markdown("##### Strengths & watch-outs")
        strengths, watchouts = strengths_and_watchouts(b_components, p_score)
        for label, val in strengths:
            st.markdown(f"✅ **{label}** — {val:.0f}/100")
        for label, val in watchouts:
            st.markdown(f"⚠️ **{label}** — {val:.0f}/100")

    with col2:
        values = [b_components["compliance"], b_components["cashflow_stability"],
                  b_components["profitability"], b_components["utility_consistency"],
                  b_components["liquidity"], b_components["growth"],
                  b_components["leverage"], p_score]
        upper = [min(100, v + band) for v in values]
        lower = [max(0, v - band) for v in values]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=upper + upper[:1], theta=AXES + AXES[:1],
                                       line=dict(color="rgba(216,90,48,0.4)", dash="dash"),
                                       name="Upper confidence"))
        fig.add_trace(go.Scatterpolar(r=lower + lower[:1], theta=AXES + AXES[:1],
                                       line=dict(color="rgba(216,90,48,0.4)", dash="dash"),
                                       fill="tonext", fillcolor="rgba(216,90,48,0.08)",
                                       name="Lower confidence"))
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=AXES + AXES[:1],
                                       fill="toself", fillcolor="rgba(29,158,117,0.25)",
                                       line=dict(color="rgb(29,158,117)", width=2),
                                       name="Score"))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), showlegend=False,
                           height=450, margin=dict(l=90, r=90, t=40, b=40))
        st.plotly_chart(fig, width="stretch", key=f"radar_{key}")

        bar_labels = AXES
        bar_values = values
        bar_colors = ["#1D9E75" if v >= 70 else "#E8A33D" if v >= 50 else "#D8453A" for v in bar_values]
        order = sorted(range(len(bar_values)), key=lambda i: bar_values[i])
        bar_fig = go.Figure(go.Bar(
            x=[bar_values[i] for i in order],
            y=[bar_labels[i] for i in order],
            orientation="h",
            marker_color=[bar_colors[i] for i in order],
            text=[f"{bar_values[i]:.0f}" for i in order],
            textposition="outside",
        ))
        bar_fig.update_layout(xaxis=dict(range=[0, 110], title="Score"), height=320,
                               margin=dict(l=10, r=10, t=10, b=10))
        st.caption("Green ≥ 70 (healthy) · Amber 50–69 (watch) · Red < 50 (risk)")
        st.plotly_chart(bar_fig, width="stretch", key=f"bar_{key}")

    return months, band, conf_label


st.title("MSME Financial Health Card")
st.caption("Track 03 — Financial Inclusion / Digital Lending / Credit Decisioning · IDBI Innovate 2026")

mode = st.radio("Mode", ["Browse existing MSMEs", "Build your own MSME"], horizontal=True)

if mode == "Browse existing MSMEs":
    ids = data_loader.list_business_ids()
    biz_id = st.selectbox("Select an MSME", ids)
    row = data_loader.get_business_record(biz_id)
    st.caption(f"Sector: {row['sector']} · Persona (synthetic label, hidden from model): {row['persona']} · "
               f"Years in business: {row['years_in_business']}")

    months, band, conf_label = render_card(row, key=biz_id)

    st.markdown("---")
    st.markdown("##### Cash flow trajectory — is it declining, or just seasonal?")
    raw_series = row.get("bank_balance_monthly")
    try:
        series = json.loads(raw_series) if isinstance(raw_series, str) else raw_series
    except (TypeError, ValueError):
        series = None

    if series and len(series) >= 3:
        n = len(series)
        window = 3
        moving_avg = [
            sum(series[max(0, i - window + 1):i + 1]) / len(series[max(0, i - window + 1):i + 1])
            for i in range(n)
        ]
        traj_fig = go.Figure()
        traj_fig.add_trace(go.Scatter(x=list(range(1, n + 1)), y=series, mode="lines+markers",
                                       name="Monthly balance", line=dict(color="#3B82C4")))
        traj_fig.add_trace(go.Scatter(x=list(range(1, n + 1)), y=moving_avg, mode="lines",
                                       name="3-month trend", line=dict(color="#D8453A", dash="dash")))
        traj_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title="Month", yaxis_title="Bank balance (₹)",
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(traj_fig, width="stretch", key=f"traj_{biz_id}")
        st.caption("A single down month against the raw line doesn't mean real decline — check "
                   "whether the 3-month trend line is still falling before concluding this business "
                   "is genuinely shrinking rather than seasonal.")
    else:
        st.caption("No monthly time series available for this profile.")

    st.markdown("---")
    st.markdown("##### Data maturity simulator — drag independent of the business above")
    sim_months = st.slider("Months of data available", 1, 24, int(row["data_history_months"]))
    st.caption("Watch the confidence band change while the score stays roughly fixed — "
               "this is the cold-start handling in action.")
    render_card(row, maturity_override=sim_months, key=f"sim_{biz_id}")

else:
    st.markdown("##### Construct a synthetic MSME")
    c1, c2, c3 = st.columns(3)
    with c1:
        sector = st.selectbox("Sector", list(SECTOR_BENCHMARKS.keys()))
        years_in_business = st.slider("Years in business", 0.0, 15.0, 1.0, 0.5)
        data_history_months = st.slider("Months of data available", 1, 36, 6)
        annual_turnover_lakh = st.slider("Annual turnover (₹ lakh)", 5, 400, 60)
        employees = st.slider("Number of employees", 1, 60, 10)
    with c2:
        net_profit_margin = st.slider("Net profit margin", 0.02, 0.30, 0.10, 0.01)
        gst_ontime_rate = st.slider("GST on-time filing rate", 0.0, 1.0, 0.8, 0.05)
        gst_return_flag = st.checkbox("GST returns filed complete (not just on-time)", value=True)
        cashflow_volatility = st.slider("Cash flow volatility", 0.0, 0.6, 0.15, 0.01)
        trend_choice = st.select_slider("Cash flow trend", ["Declining", "Flat", "Growing"], value="Flat")
        bounce_rate = st.slider("Transaction bounce rate", 0.0, 0.2, 0.02, 0.01)
        outstanding_loans_lakh = st.slider("Outstanding loans (₹ lakh)", 0.0, 150.0, 10.0)
    with c3:
        has_cibil = st.checkbox("Promoter has CIBIL history", value=True)
        promoter_cibil = st.slider("Promoter CIBIL", 300, 900, 700) if has_cibil else None
        promoter_age = st.slider("Promoter age", 21, 65, 35)
        promoter_income_lakh = st.slider("Promoter annual income (₹ lakh)", 1.0, 25.0, 8.0)

    bench = SECTOR_BENCHMARKS[sector]
    elec_default = int(sum(bench["elec"]) / 2 * annual_turnover_lakh * 12)
    water_default = int(sum(bench["water"]) / 2 * employees)
    gas_default = int(sum(bench["gas"]) / 2 * annual_turnover_lakh * 12)

    st.markdown("##### Resource consumption (actual quantities — reflects scale + equipment efficiency)")
    u1, u2, u3 = st.columns(3)
    with u1:
        utility_kwh_year = st.slider("Electricity (kWh/year)", 0, max(elec_default * 3, 100),
                                      elec_default, help="Sector-typical default shown; drag to simulate more/less efficient machinery or a mismatch with declared turnover.")
    with u2:
        utility_water_kl_year = st.slider("Water (kL/year)", 0, max(water_default * 3, 100), water_default)
    with u3:
        utility_gas_year = st.slider("Gas (units/year)", 0, max(gas_default * 3, 100), gas_default)

    trend_map = {"Declining": -0.02, "Flat": 0.0, "Growing": 0.03}
    row = dict(
        business_id="CUSTOM", sector=sector, years_in_business=years_in_business,
        data_history_months=data_history_months, annual_turnover_lakh=annual_turnover_lakh,
        net_profit_margin=net_profit_margin, gst_ontime_rate=gst_ontime_rate,
        gst_return_flag=int(gst_return_flag),
        cashflow_volatility=cashflow_volatility, manual_trend_rate=trend_map[trend_choice],
        bounce_rate=bounce_rate,
        utility_kwh_year=utility_kwh_year, utility_water_kl_year=utility_water_kl_year,
        utility_gas_year=utility_gas_year,
        employees=employees, outstanding_loans_rupee=outstanding_loans_lakh * 100000,
        promoter_cibil=promoter_cibil, promoter_age=promoter_age,
        promoter_income_lakh=promoter_income_lakh,
    )
    st.markdown("---")
    render_card(row, key="CUSTOM")

st.markdown("---")
st.caption("Prototype runs on synthetic data mirroring GST/UPI/AA/EPFO schemas. "
           "Production swaps only data_loader.py for live sandbox APIs — scoring logic and UI are unchanged.")
