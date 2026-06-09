"""Streamlit dashboard for the temporal medication–lab conflict detector.

Run: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app_data import get_data, get_model
from src.config import OUTPUTS_DIR, SCENARIOS
from src.preprocessing.build_timeline import labs_for
from src.risk.assess import assess_patient
from src.rules.baseline_rules import first_static_alert_date
from src.rules.temporal_rules import first_temporal_alert_date

st.set_page_config(page_title="Temporal Med–Lab Conflict Detector", layout="wide")

RISK_COLOR = {"Low": "#2ca02c", "Moderate": "#ff7f0e",
              "High": "#d62728", "Critical": "#7d0a0a"}


@st.cache_data(show_spinner="Scoring patients ...")
def score_scenario(scenario_key: str) -> pd.DataFrame:
    d = get_data()
    model = get_model(scenario_key)
    rows = []
    for _, patient in d["patients"].iterrows():
        a = assess_patient(scenario_key, patient, d["medications"], d["labs"],
                           d["knowledge"], model, explain_shap=False)
        if a is None:
            continue
        rows.append({
            "patient_id": a["patient_id"], "drug": a["features"].get("drug_name"),
            "risk_score": a["risk_score"], "risk_level": a["risk_level"],
            "static_alert": a["static_alert"], "temporal_alert": a["temporal_alert"],
            "ml_probability": round(a["ml_probability"], 3) if a["ml_probability"] else None,
        })
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False)


def timeline_chart(scenario_key: str, pid: int):
    d = get_data()
    s = SCENARIOS[scenario_key]
    med = d["medications"][(d["medications"]["patient_id"] == pid)
                           & (d["medications"]["drug_class"] == s.drug_class)]
    if med.empty:
        return None
    med = med.iloc[0]
    drug_start = med["start_date"]
    dose_change = med.get("dose_change_date")
    dose_change = dose_change if pd.notna(dose_change) else None

    series = labs_for(d["labs"], pid, s.primary_lab)
    days = [(pd.Timestamp(x) - pd.Timestamp(drug_start)).days for x in series["lab_date"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days, y=series["value"], mode="lines+markers",
                             name=s.primary_lab))
    fig.add_hline(y=s.static_threshold, line_dash="dash", line_color="red",
                  annotation_text="static threshold")
    fig.add_hline(y=s.temporal_current_gate, line_dash="dot", line_color="orange",
                  annotation_text="temporal gate")
    fig.add_vline(x=0, line_color="green", annotation_text="drug start")

    s_date = first_static_alert_date(s, d["labs"], pid, drug_start)
    t_date = first_temporal_alert_date(s, d["labs"], med_patient(d, pid), drug_start, dose_change)
    if s_date:
        fig.add_vline(x=(s_date - drug_start).days, line_color="red", opacity=0.4,
                      annotation_text="static alert")
    if t_date:
        fig.add_vline(x=(t_date - drug_start).days, line_color="orange", opacity=0.5,
                      annotation_text="temporal alert")
    fig.update_layout(height=420, xaxis_title="Days since drug start",
                      yaxis_title=s.primary_lab, margin=dict(t=30))
    return fig


def med_patient(d, pid):
    return d["patients"][d["patients"]["patient_id"] == pid].iloc[0]


# --------------------------------------------------------------------------- #
st.title("Explainable Temporal Medication–Laboratory Conflict Detector")
st.caption("Research prototype — supports, does not replace, clinical judgment. "
           "Synthetic proof-of-concept data.")

with st.sidebar:
    st.header("Filters")
    scenario_key = st.selectbox("Scenario", list(SCENARIOS),
                                format_func=lambda k: SCENARIOS[k].name)
    levels = st.multiselect("Risk level", list(RISK_COLOR),
                            default=list(RISK_COLOR))
    high_only = st.checkbox("High-risk only (alerts firing)", value=False)

scored = score_scenario(scenario_key)
view = scored[scored["risk_level"].isin(levels)]
if high_only:
    view = view[view["temporal_alert"] | view["static_alert"]]

# summary metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", len(scored))
c2.metric("Static alerts", int(scored["static_alert"].sum()))
c3.metric("Temporal alerts", int(scored["temporal_alert"].sum()))
c4.metric("High/Critical", int(scored["risk_level"].isin(["High", "Critical"]).sum()))

left, right = st.columns([1, 1.3])

with left:
    st.subheader("Patients by risk")
    st.dataframe(view, use_container_width=True, height=420,
                 column_config={"risk_score": st.column_config.ProgressColumn(
                     "risk_score", min_value=0, max_value=100, format="%d")})
    options = view["patient_id"].tolist()
    selected = st.selectbox("Inspect patient", options) if options else None

with right:
    if selected is not None:
        d = get_data()
        a = assess_patient(scenario_key, med_patient(d, selected), d["medications"],
                           d["labs"], d["knowledge"], get_model(scenario_key),
                           explain_shap=True)
        color = RISK_COLOR.get(a["risk_level"], "#444")
        st.markdown(f"### Patient {selected} — "
                    f"<span style='color:{color}'>{a['risk_level']} "
                    f"({a['risk_score']})</span>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        b1.metric("ML probability", f"{a['ml_probability']:.2f}" if a['ml_probability'] else "n/a")
        b2.metric("Static rule", "FIRED" if a["static_alert"] else "—")
        b3.metric("Temporal rule", "FIRED" if a["temporal_alert"] else "—")

        fig = timeline_chart(scenario_key, selected)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Explanation**")
        st.info(a["explanation"])
        st.markdown("**Suggested review action**")
        st.warning(a["suggested_action"])

        with st.expander("Risk-score components"):
            st.bar_chart(pd.Series(a["risk_components"]))
        if a["shap_top"]:
            with st.expander("Top model drivers (SHAP)"):
                st.table(pd.DataFrame(a["shap_top"], columns=["feature", "shap_value"]))

        st.download_button(
            "Export patient report (JSON)",
            data=json.dumps(a, indent=2, default=str),
            file_name=f"patient_{selected}_{scenario_key}_report.json",
            mime="application/json")

st.divider()
st.subheader("Rule-based vs temporal model — performance")
metrics_path = OUTPUTS_DIR / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    tab1, tab2, tab3 = st.tabs(["Rules vs Temporal", "ML models", "Early detection"])
    tab1.dataframe(pd.DataFrame(metrics["exp1_rules_vs_temporal"]), use_container_width=True)
    tab2.dataframe(pd.DataFrame(metrics["exp2_model_comparison"]), use_container_width=True)
    tab3.dataframe(pd.DataFrame(metrics["exp3_early_detection"]), use_container_width=True)
else:
    st.info("Run `python run_pipeline.py` to populate performance metrics.")
