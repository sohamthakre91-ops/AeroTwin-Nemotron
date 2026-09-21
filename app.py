"""
AEROTWIN-NEMOTRON: Tactical Digital Twin & Mission Assurance Dashboard
Autonomous UAV Propulsion Health & Mission Assurance Platform
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from src.mission import MissionSimulator, MISSION_CSV_PATH, REQUIRED_TELEMETRY_COLUMNS
from src.diagnostics import (
    train_fault_classifier,
    predict_fault,
    load_metadata,
    FEATURE_COLUMNS,
    DEFAULT_DATASET_PATH,
)
from src.baseline import compare_with_baseline
from src.trends import analyze_trends, assess_system_status
from src.risk import assess_mission_risk
from src.tools import run_what_if_scenario, compare_scenarios

# =====================================================================
# PAGE CONFIGURATION & TACTICAL STYLING
# =====================================================================

st.set_page_config(
    page_title="AeroTwin-Nemotron | UAV Propulsion Health",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TACTICAL_CSS = """
<style>
    /* Dark tactical aeronautical styling */
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background: linear-gradient(135deg, #131c2e 0%, #0d1524 100%);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
    }
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .badge-tag {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 4px;
        background-color: #1e293b;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .badge-accent {
        background-color: #0c4a6e;
        color: #38bdf8;
        border: 1px solid #0284c7;
    }
    .badge-warn {
        background-color: #451a03;
        color: #fb923c;
        border: 1px solid #b45309;
    }
    .status-card {
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 5px solid;
    }
    .status-NORMAL {
        background-color: #052e16;
        border-color: #22c55e;
        color: #86efac;
    }
    .status-MONITORING {
        background-color: #082f49;
        border-color: #0ea5e9;
        color: #7dd3fc;
    }
    .status-WARNING {
        background-color: #422006;
        border-color: #f59e0b;
        color: #fde047;
    }
    .status-CRITICAL {
        background-color: #450a0a;
        border-color: #ef4444;
        color: #fca5a5;
    }
    .nemotron-card {
        background: linear-gradient(135deg, #091e14 0%, #061510 100%);
        border: 1px dashed #10b981;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 15px;
    }
</style>
"""
st.markdown(TACTICAL_CSS, unsafe_allow_html=True)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

defaults = {
    "mission_running": False,
    "mission_complete": False,
    "mission_history": [],
    "current_point": None,
    "simulator": None,
    "diagnosis": None,
    "baseline": None,
    "trends": None,
    "anomaly_status": None,
    "risk": None,
    "training_in_progress": False,
    "training_logs": [],
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# If mission history is empty but mission_telemetry.csv exists, load it
if not st.session_state.mission_history and MISSION_CSV_PATH.exists():
    try:
        cached_df = pd.read_csv(MISSION_CSV_PATH)
        if len(cached_df) > 0:
            st.session_state.mission_history = cached_df.to_dict(orient="records")
            st.session_state.current_point = cached_df.iloc[-1].to_dict()
            st.session_state.mission_complete = True
    except Exception:
        pass

# =====================================================================
# HEADER & PROTOCOL BADGES
# =====================================================================

st.title("✈️ AEROTWIN-NEMOTRON")
st.subheader("Autonomous UAV Propulsion Health & Mission Assurance")

st.markdown("""
<div class="badge-bar">
    <span class="badge-tag badge-accent">SYNTHETIC TELEMETRY</span>
    <span class="badge-tag badge-accent">PHYSICS-INFORMED SIMULATION</span>
    <span class="badge-tag badge-warn">RESEARCH PROTOTYPE</span>
    <span class="badge-tag badge-warn">NOT FLIGHT CERTIFIED</span>
    <span class="badge-tag">TARGET: ROTAX 915 iS-CLASS (104 kW TURBO)</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================================
# SIDEBAR: MISSION CONTROLS
# =====================================================================

st.sidebar.header("🎯 MISSION CONFIGURATION")

alt_val = st.sidebar.slider(
    "Target Cruise Altitude (m)",
    min_value=1000,
    max_value=8000,
    value=3000,
    step=250,
    help="Rotax 915 iS class operating ceiling is 12,000m. Nominal cruise is 3,000m.",
)

thr_val = st.sidebar.slider(
    "Cruise Throttle Position",
    min_value=0.45,
    max_value=0.90,
    value=0.75,
    step=0.05,
    help="Continuous maximum cruise throttle is approximately 0.75 to 0.85.",
)

dur_val = st.sidebar.slider(
    "Mission Duration (min)",
    min_value=20,
    max_value=60,
    value=60,
    step=10,
    help="Default mission is 60 minutes.",
)

smp_val = st.sidebar.slider(
    "Sampling Rate (samples/min)",
    min_value=10,
    max_value=20,
    value=20,
    step=5,
    help="20 samples/min yields 1,200 samples for a 60-min mission.",
)

total_target_samples = int(dur_val * smp_val)
st.sidebar.caption(f"📊 Total Mission Telemetry: **{total_target_samples:,} samples** (dt = {60.0/smp_val:.1f}s)")

st.sidebar.divider()
st.sidebar.markdown("**Simulation Control**")
sim_step_delay = st.sidebar.slider("Playback Speed (batch update delay)", 0.0, 0.20, 0.02, 0.01)

st.sidebar.info(
    "🛡️ **Autonomous Stress Degradation**\n\n"
    "Faults are NEVER manually injected. The engine begins 100% pristine. "
    "Internal degradation accumulates gradually from operating stress over the mission."
)

# Start / Reset Buttons
col_start, col_reset = st.sidebar.columns(2)
start_clicked = col_start.button("🚀 START MISSION", type="primary", use_container_width=True)
reset_clicked = col_reset.button("↻ RESET", use_container_width=True)

if reset_clicked:
    st.session_state.mission_running = False
    st.session_state.mission_complete = False
    st.session_state.mission_history = []
    st.session_state.current_point = None
    st.session_state.diagnosis = None
    st.session_state.baseline = None
    st.session_state.trends = None
    st.session_state.anomaly_status = None
    st.session_state.risk = None
    st.rerun()

# =====================================================================
# CONTINUOUS MISSION EXECUTION LOGIC
# =====================================================================

if start_clicked:
    st.session_state.mission_running = True
    st.session_state.mission_complete = False
    st.session_state.mission_history = []
    st.session_state.current_point = None
    st.session_state.diagnosis = None
    st.session_state.baseline = None
    st.session_state.trends = None
    st.session_state.anomaly_status = None
    st.session_state.risk = None

    sim = MissionSimulator(
        target_altitude_m=alt_val,
        base_throttle=thr_val,
        duration_minutes=dur_val,
        samples_per_minute=smp_val,
        seed=42,
    )
    st.session_state.simulator = sim

# If mission is active, run the simulation loop
if st.session_state.mission_running:
    sim: MissionSimulator = st.session_state.simulator
    clock_placeholder = st.empty()
    progress_placeholder = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Step through mission with smooth visualization batches
    batch_size = max(1, smp_val)  # Update UI every 1 simulated minute
    total_steps = int(dur_val * smp_val)

    step_count = 0
    while step_count < total_steps:
        # Step the backend simulation
        for _ in range(batch_size):
            if step_count >= total_steps:
                break
            point = sim.step()
            st.session_state.mission_history.append(point)
            st.session_state.current_point = point
            step_count += 1

        curr_t = st.session_state.current_point["mission_time_min"]
        pct = min(1.0, step_count / total_steps)

        # Update clock & progress
        mins = int(curr_t)
        secs = int((curr_t - mins) * 60)
        clock_placeholder.markdown(f"### 🔴 MISSION LIVE — Clock: `T+{mins:02d}:{secs:02d}` / `T+{dur_val:02d}:00` ({step_count}/{total_steps} samples)")
        progress_placeholder.progress(pct)

        # Update Live Telemetry Metrics
        p = st.session_state.current_point
        with metrics_placeholder.container():
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("RPM", f"{p['rpm']:,.0f}")
            c2.metric("MAP", f"{p['map_kpa']:.1f} kPa")
            c3.metric("Power", f"{p['power_kw']:.1f} kW")
            c4.metric("CHT", f"{p['cht_c']:.1f} °C")
            c5.metric("EGT", f"{p['egt_c']:.1f} °C")
            c6.metric("Oil Press", f"{p['oil_pressure_kpa']:.1f} kPa")
            c7.metric("Vibration", f"{p['vibration_g']:.3f} g")

        # Update live line chart
        df_hist = pd.DataFrame(st.session_state.mission_history)
        if len(df_hist) > 2 and step_count % (batch_size * 2) == 0:
            sub_df = df_hist[["mission_time_min", "cht_c", "egt_c", "oil_pressure_kpa", "vibration_g"]].copy()
            chart_placeholder.line_chart(sub_df.set_index("mission_time_min"), height=250)

        if sim_step_delay > 0:
            time.sleep(sim_step_delay)

    # Export authoritative telemetry CSV (1,200 rows)
    sim.export_csv(MISSION_CSV_PATH)

    st.session_state.mission_running = False
    st.session_state.mission_complete = True

    # Authoritative Post-Mission Analytics
    final_point = st.session_state.current_point
    st.session_state.baseline = compare_with_baseline(final_point)
    st.session_state.diagnosis = predict_fault(final_point)
    st.session_state.trends = analyze_trends(st.session_state.mission_history)
    st.session_state.anomaly_status = assess_system_status(
        st.session_state.mission_history,
        baseline_result=st.session_state.baseline,
        ml_diagnosis=st.session_state.diagnosis,
    )
    st.session_state.risk = assess_mission_risk(
        telemetry=final_point,
        baseline_result=st.session_state.baseline,
        ml_diagnosis=st.session_state.diagnosis,
        trend_result=st.session_state.trends,
        mission_duration_min=dur_val,
    )
    st.rerun()

# =====================================================================
# DASHBOARD TABS & MAIN DISPLAY
# =====================================================================

tab_live, tab_analysis, tab_baseline, tab_whatif, tab_ml_train, tab_raw, tab_nemotron = st.tabs([
    "📡 Telemetry & Mission State",
    "🔍 Anomaly & Diagnostics",
    "⚖️ Healthy Baseline Comparison",
    "🧪 What-If Counterfactual",
    "🤖 Model Training & Evaluation",
    "📊 Raw Telemetry (1200+ Samples)",
    "⚡ NVIDIA Nemotron (Future)",
])

# Ensure analysis is populated if mission history exists
if st.session_state.mission_history and st.session_state.current_point is not None:
    if st.session_state.baseline is None:
        p = st.session_state.current_point
        st.session_state.baseline = compare_with_baseline(p)
        st.session_state.diagnosis = predict_fault(p)
        st.session_state.trends = analyze_trends(st.session_state.mission_history)
        st.session_state.anomaly_status = assess_system_status(
            st.session_state.mission_history,
            st.session_state.baseline,
            st.session_state.diagnosis,
        )
        st.session_state.risk = assess_mission_risk(
            p,
            st.session_state.baseline,
            st.session_state.diagnosis,
            st.session_state.trends,
            mission_duration_min=dur_val,
        )

# ---------------------------------------------------------------------
# TAB 1: TELEMETRY & MISSION STATE
# ---------------------------------------------------------------------
with tab_live:
    if st.session_state.current_point is not None:
        p = st.session_state.current_point
        hist_len = len(st.session_state.mission_history)
        last_t = p.get("mission_time_min", 0.0)

        st.markdown(f"#### 🛰️ Propulsion Telemetry Status — `T+{last_t:.2f} min` ({hist_len:,} samples recorded)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Altitude", f"{p['altitude_m']:,.0f} m", delta=f"{p['throttle']*100:.0f}% Throttle")
        c2.metric("Engine RPM", f"{p['rpm']:,.0f} RPM", delta="Governed")
        c3.metric("MAP (Boosted)", f"{p['map_kpa']:.1f} kPa", delta=f"{p['power_kw']:.1f} kW Power")
        c4.metric("Fuel Flow", f"{p['fuel_flow_lph']:.1f} L/h", delta=f"{p['engine_load']*100:.0f}% Load")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("CHT (Cylinder Head)", f"{p['cht_c']:.1f} °C", delta=f"Base: 140.0°C", delta_color="inverse")
        c6.metric("EGT (Exhaust Gas)", f"{p['egt_c']:.1f} °C", delta="Nominal ~720°C")
        c7.metric("Oil Pressure", f"{p['oil_pressure_kpa']:.1f} kPa", delta="Normal: 380 kPa")
        c8.metric("Vibration", f"{p['vibration_g']:.4f} g", delta="Baseline: 0.025 g", delta_color="inverse")

        st.divider()
        st.markdown("#### 📈 Multi-Channel Mission Trajectories")

        df_h = pd.DataFrame(st.session_state.mission_history)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_thermal = go.Figure()
            fig_thermal.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["cht_c"], name="CHT (°C)", line=dict(color="#f97316", width=2)))
            fig_thermal.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["oil_temperature_c"], name="Oil Temp (°C)", line=dict(color="#eab308", width=1.5)))
            fig_thermal.update_layout(title="Thermal Response (CHT & Oil Temp)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_thermal, use_container_width=True)

        with col_c2:
            fig_lub = go.Figure()
            fig_lub.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["oil_pressure_kpa"], name="Oil Pressure (kPa)", line=dict(color="#38bdf8", width=2)))
            fig_lub.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["power_kw"], name="Power (kW)", line=dict(color="#4ade80", width=1.5)))
            fig_lub.update_layout(title="Lubrication Pressure & Engine Power", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lub, use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_vibe = go.Figure()
            fig_vibe.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["vibration_g"], name="Vibration (g)", line=dict(color="#ec4899", width=2)))
            fig_vibe.update_layout(title="Mechanical Vibration Spectrum", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_vibe, use_container_width=True)

        with col_c4:
            fig_egt = go.Figure()
            fig_egt.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["egt_c"], name="EGT (°C)", line=dict(color="#f43f5e", width=2)))
            fig_egt.update_layout(title="Exhaust Gas Temperature Trajectory", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_egt, use_container_width=True)

    else:
        st.info("No mission currently active. Configure flight parameters in the sidebar and click **'🚀 START MISSION'** to begin.")

# ---------------------------------------------------------------------
# TAB 2: ANOMALY & DIAGNOSTICS
# ---------------------------------------------------------------------
with tab_analysis:
    if st.session_state.current_point is not None:
        st.markdown("### 🔬 System Health & Anomaly Assessment")

        # 1. Anomaly Status Card
        status_info = st.session_state.anomaly_status or {"status": "NORMAL", "evidence": []}
        curr_status = status_info.get("status", "NORMAL")
        st.markdown(f"""
        <div class="status-card status-{curr_status}">
            <h3 style="margin:0;">SYSTEM STATUS: {curr_status}</h3>
            <p style="margin: 4px 0 0 0;">Severity Score: <strong>{status_info.get('severity_score', 0.0)} / 10.0</strong></p>
        </div>
        """, unsafe_allow_html=True)

        col_ev, col_diag = st.columns([1.2, 1.0])

        with col_ev:
            st.markdown("#### 📋 Quantitative Supporting Evidence")
            evidence_list = status_info.get("evidence", [])
            if evidence_list:
                for ev in evidence_list:
                    st.markdown(f"- 🔸 {ev}")
            else:
                st.markdown("- No abnormal trends detected. All signals nominal.")

            # Trend Window Details
            if st.session_state.trends and "signals" in st.session_state.trends:
                st.markdown("#### 📊 Early (First 10%) vs Late (Last 10%) Mission Window Comparison")
                trend_rows = []
                for sig, sdata in st.session_state.trends["signals"].items():
                    trend_rows.append({
                        "Signal": sig,
                        "Early Window Mean": sdata["early_window_mean"],
                        "Late Window Mean": sdata["late_window_mean"],
                        "Delta": f"{sdata['window_delta']:+.2f}",
                        "Change %": f"{sdata['percent_change']:+.1f}%",
                        "Slope/min": f"{sdata['slope_per_min']:+.4f}",
                        "Persistent?": "YES" if sdata["is_persistent"] else "NO",
                    })
                st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

        with col_diag:
            st.markdown("#### 🤖 ML Diagnostic Fault Classification")
            diag = st.session_state.diagnosis or {}
            pred_fault = diag.get("predicted_fault", "normal")
            conf = diag.get("confidence", 0.0)

            c_p1, c_p2 = st.columns(2)
            c_p1.metric("Predicted Condition", pred_fault.replace("_", " ").upper())
            c_p2.metric("Classification Confidence", f"{conf*100:.1f}%")

            # Ranked Hypotheses
            st.markdown("**Ranked Diagnostic Hypotheses:**")
            ranked = diag.get("ranked_hypotheses", [])
            if ranked:
                df_ranked = pd.DataFrame(ranked)
                df_ranked["probability_pct"] = df_ranked["probability"].apply(lambda x: f"{x*100:.1f}%")
                df_ranked["fault"] = df_ranked["fault"].str.replace("_", " ").str.title()
                fig_bar = px.bar(
                    df_ranked,
                    x="probability",
                    y="fault",
                    orientation="h",
                    color="probability",
                    color_continuous_scale="Blues",
                    labels={"probability": "Posterior Probability", "fault": "Fault Hypothesis"},
                    template="plotly_dark",
                    height=280,
                )
                fig_bar.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        # Mission Assurance Risk Score Section
        st.markdown("### 🛡️ Mission Assurance & Risk Quantification")
        risk = st.session_state.risk or {}
        r_col1, r_col2, r_col3 = st.columns([1, 1, 2])
        r_col1.metric("Mission Risk Score", f"{risk.get('risk_score', 0.0):.1f} / 100")
        r_col2.metric("Assurance Level", risk.get("risk_level", "LOW"))
        r_col3.info(f"**Operational Impact:**\n{risk.get('mission_impact', 'Normal envelope')}")

        if "indicators" in risk:
            ind = risk["indicators"]
            st.caption(f"Risk Sub-Scores — Thermal: {ind.get('thermal_stress_score', 0)} | Lubrication: {ind.get('lubrication_stress_score', 0)} | Vibration: {ind.get('vibration_stress_score', 0)} | ML Factor: {ind.get('ml_diagnostic_score', 0)}")
    else:
        st.info("Start mission to observe live anomaly detection and diagnostic hypotheses.")

# ---------------------------------------------------------------------
# TAB 3: HEALTHY BASELINE COMPARISON
# ---------------------------------------------------------------------
with tab_baseline:
    if st.session_state.baseline:
        base_data = st.session_state.baseline
        st.markdown("### ⚖️ Healthy Twin vs Observed Operating State")
        st.caption("Baseline values computed dynamically using pristine Digital Twin physics at identical altitude & throttle.")

        dev_list = []
        for param, d in base_data.get("deviations", {}).items():
            dev_list.append({
                "Parameter": param,
                "Observed Telemetry": d["observed"],
                "Healthy Reference": d["healthy_baseline"],
                "Delta": f"{d['delta']:+.3f}",
                "Deviation %": f"{d['percent_deviation']:+.2f}%",
                "Significant (>3%)": "⚠️ YES" if abs(d["percent_deviation"]) >= 3.0 else "NOMINAL",
            })

        df_base = pd.DataFrame(dev_list)
        st.dataframe(df_base, use_container_width=True, hide_index=True)

        sig_devs = base_data.get("significant_deviations", [])
        if sig_devs:
            st.warning(f"⚠️ **{len(sig_devs)} Significant Deviations Detected:**")
            for item in sig_devs:
                st.markdown(f"- **{item['parameter']}**: Observed `{item['observed']}` vs Baseline `{item['healthy_baseline']}` (`{item['percent_deviation']:+.2f}%`)")
        else:
            st.success("✅ All observable parameters are tracking within ±3.0% of the healthy reference twin.")
    else:
        st.info("Execute a mission run to compare telemetry against the healthy baseline.")

# ---------------------------------------------------------------------
# TAB 4: WHAT-IF COUNTERFACTUAL
# ---------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🧪 Counterfactual What-If Simulation")
    st.caption("Test engineering hypotheses: Simulate how the Digital Twin behaves under custom degraded health states and compare with observed telemetry.")

    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        st.markdown("**Hypothetical Subsystem Health**")
        hypo_alt = st.number_input("Hypothetical Altitude (m)", 0.0, 10000.0, 3000.0, 250.0)
        hypo_thr = st.slider("Hypothetical Throttle", 0.20, 1.00, 0.75, 0.05)
        hypo_cooling = st.slider("Cooling Health", 0.0, 1.0, 0.75, 0.05)
        hypo_fuel = st.slider("Fuel Delivery Health", 0.0, 1.0, 1.0, 0.05)
        hypo_oil = st.slider("Oil/Lubrication Health", 0.0, 1.0, 1.0, 0.05)
        hypo_bearing = st.slider("Bearing Mechanical Health", 0.0, 1.0, 1.0, 0.05)
        hypo_sensor = st.slider("Sensor Integrity", 0.0, 1.0, 1.0, 0.05)

        run_hypo = st.button("Simulate Hypothesis", type="primary")

    with col_w2:
        if run_hypo or st.session_state.current_point is not None:
            observed_target = st.session_state.current_point
            whatif_res = run_what_if_scenario(
                altitude_m=hypo_alt,
                throttle=hypo_thr,
                cooling_health=hypo_cooling,
                fuel_health=hypo_fuel,
                oil_health=hypo_oil,
                bearing_health=hypo_bearing,
                sensor_health=hypo_sensor,
                target_telemetry=observed_target,
            )

            st.markdown("#### Hypothetical State vs Observed Mission Telemetry")
            if "comparison" in whatif_res:
                comp = whatif_res["comparison"]
                c_s1, c_s2 = st.columns(2)
                c_s1.metric("Hypothesis Match Similarity", f"{comp['similarity_score_percent']:.1f}%")
                c_s2.metric("Normalized MSE", f"{comp['mean_squared_normalized_error']:.4f}")

                st.markdown("**Signal Deltas (Hypothetical - Observed):**")
                st.json(comp["deltas"])

            st.markdown("**Predicted Telemetry Under Hypothesis:**")
            hypo_t = whatif_res["hypothetical_telemetry"]
            st.dataframe(pd.DataFrame([hypo_t]), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 5: MODEL TRAINING & EVALUATION
# ---------------------------------------------------------------------
with tab_ml_train:
    st.markdown("### 🤖 Model Training, Evaluation & Version Registry")
    st.caption("Train a fresh RandomForestClassifier using GroupShuffleSplit to guarantee zero scenario overlap.")

    meta = load_metadata()
    latest_v = meta.get("latest_version")
    versions = meta.get("versions", {})

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        include_new_mission = st.checkbox("Retrain using Latest Mission Telemetry", value=False)
        train_button = st.button("🚀 TRAIN NEW MODEL", type="primary", use_container_width=True)

    with col_t2:
        if latest_v and latest_v in versions:
            cur_metrics = versions[latest_v]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Active Model", latest_v)
            m2.metric("Accuracy", f"{cur_metrics['accuracy']*100:.2f}%")
            m3.metric("Macro F1", f"{cur_metrics['macro_f1']:.4f}")
            m4.metric("Dataset Size", f"{cur_metrics['dataset_size']:,}")

    if train_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def gui_progress(stage, msg):
            pct = stage / 7.0
            progress_bar.progress(pct)
            status_text.markdown(f"**Stage {stage}/7:** {msg}")

        try:
            extra_df = None
            if include_new_mission and st.session_state.mission_history:
                extra_df = pd.DataFrame(st.session_state.mission_history)
                # Ensure training required columns
                extra_df["scenario_id"] = 9999
                extra_df["fault"] = "normal"  # Autonomous mission starts healthy
                extra_df["fault_severity"] = 0.0

            model, new_metrics = train_fault_classifier(progress_callback=gui_progress)
            progress_bar.progress(1.0)
            status_text.success(f"✅ Model {new_metrics['version']} trained successfully!")
            st.rerun()
        except Exception as e:
            status_text.error(f"❌ Training failed: {str(e)}")

    st.divider()

    # Display Active Model Details
    if latest_v and latest_v in versions:
        active_m = versions[latest_v]
        st.markdown(f"#### 📊 Performance Evaluation for `{latest_v}`")

        c_d1, c_d2, c_d3, c_d4, c_d5, c_d6, c_d7 = st.columns(7)
        c_d1.metric("Dataset Size", f"{active_m['dataset_size']:,}")
        c_d2.metric("Train Size", f"{active_m['training_size']:,}")
        c_d3.metric("Test Size", f"{active_m['testing_size']:,}")
        c_d4.metric("Features", f"{active_m['feature_count']}")
        c_d5.metric("Accuracy", f"{active_m['accuracy']*100:.2f}%")
        c_d6.metric("Macro F1", f"{active_m['macro_f1']:.4f}")
        c_d7.metric("Weighted F1", f"{active_m['weighted_f1']:.4f}")

        # Confusion Matrix Heatmap
        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.markdown("**Confusion Matrix (7 Classes)**")
            cm_data = active_m.get("confusion_matrix", [])
            class_labels = active_m.get("class_list", [])
            if cm_data and class_labels:
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Samples"),
                    x=[c.replace("_", " ").title() for c in class_labels],
                    y=[c.replace("_", " ").title() for c in class_labels],
                    text_auto=True,
                    template="plotly_dark",
                    color_continuous_scale="Viridis",
                    height=360,
                )
                fig_cm.update_layout(margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_fi:
            st.markdown("**Derived Feature Importance**")
            fi_list = active_m.get("feature_importance", [])
            if fi_list:
                df_fi = pd.DataFrame(fi_list)
                fig_fi = px.bar(
                    df_fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    template="plotly_dark",
                    color="importance",
                    color_continuous_scale="Purples",
                    height=360,
                )
                fig_fi.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_fi, use_container_width=True)

        # Model Version Registry Table
        st.markdown("#### 📜 Model Version Registry")
        hist_rows = []
        for v_name, v_data in versions.items():
            hist_rows.append({
                "Version": v_name,
                "Timestamp": v_data.get("timestamp", "N/A"),
                "Dataset Size": f"{v_data.get('dataset_size', 0):,}",
                "Accuracy": f"{v_data.get('accuracy', 0)*100:.2f}%",
                "Macro F1": f"{v_data.get('macro_f1', 0):.4f}",
                "Weighted F1": f"{v_data.get('weighted_f1', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 6: RAW TELEMETRY
# ---------------------------------------------------------------------
with tab_raw:
    st.markdown("### 📊 Mission Telemetry History")
    if st.session_state.mission_history:
        df_raw = pd.DataFrame(st.session_state.mission_history)
        st.caption(f"Total Telemetry Records: **{len(df_raw):,} rows** | Authoritative CSV location: `{MISSION_CSV_PATH}`")

        # Column selection
        selected_cols = st.multiselect(
            "Select Columns to Display",
            options=list(df_raw.columns),
            default=list(df_raw.columns)[:10],
        )

        st.dataframe(df_raw[selected_cols] if selected_cols else df_raw, use_container_width=True, height=400)

        csv_data = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Mission Telemetry (CSV)",
            data=csv_data,
            file_name="mission_telemetry.csv",
            mime="text/csv",
        )
    else:
        st.info("No mission telemetry generated yet. Run a mission from the sidebar.")

# ---------------------------------------------------------------------
# TAB 7: FUTURE NEMOTRON PLACEHOLDER
# ---------------------------------------------------------------------
with tab_nemotron:
    st.markdown("""
    <div class="nemotron-card">
        <h3 style="color:#10b981; margin:0 0 8px 0;">⚡ NVIDIA NEMOTRON — AUTONOMOUS INVESTIGATION AGENT</h3>
        <p style="font-size:0.95rem; color:#a7f3d0; margin-bottom:12px;">
            <strong>STATUS: NOT CONNECTED</strong> &nbsp;|&nbsp;
            <strong>NEXT DEVELOPMENT PHASE:</strong> Nebius Token Factory + NVIDIA Nemotron
        </p>
        <p style="font-size:0.85rem; color:#94a3b8; line-height:1.6;">
            In this local engineering core build, all physical digital twin models, autonomous degradation pipelines,
            sensor models, trend analysis algorithms, healthy baseline comparators, and counterfactual what-if simulation
            tools are completely built and verified locally.
            <br/><br/>
            In the upcoming manual integration phase, NVIDIA Nemotron will be connected to query these exact local tools:
            <code>get_engine_state()</code>, <code>run_fault_detection()</code>, <code>compare_with_baseline()</code>,
            <code>run_what_if_scenario()</code>, and <code>assess_mission_risk()</code>.
            <br/><br/>
            <em>Note: No external API requests, OpenAI clients, or Nebius token authorizations are executed during this phase.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)