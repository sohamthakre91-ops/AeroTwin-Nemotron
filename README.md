# AeroTwin-Nemotron: Autonomous UAV Propulsion Health & Mission Assurance

> **Local Engineering Core Build (Phase 1)**  
> **Physics-Informed Digital Twin • ML Diagnostics • Mission Assurance Platform**  
> *Target Class: Rotax 915 iS-class Turbocharged UAV Piston Engine*

---

## 1. Project Overview

**AeroTwin-Nemotron** is a physics-informed digital twin and autonomous mission assurance platform developed for long-endurance unmanned aerial vehicles (UAVs). It provides continuous propulsion health monitoring, progressive stress-induced degradation modeling, real-time multi-channel telemetry synthesis, machine learning fault diagnosis, healthy baseline tracking, counterfactual "what-if" testing, and quantitative mission risk estimation.

The system is designed to answer three central operational questions:
1. **What is the current condition of the propulsion system?**
2. **What physical and statistical evidence supports that assessment?**
3. **How does that condition impact mission assurance and flight survivability?**

> [!IMPORTANT]
> **Safety & Research Disclaimer**: This software is a physics-informed research prototype. It is **NOT** flight-certified, is **NOT** connected to real aircraft systems, and must **NOT** be used to make operational flight or engine control decisions. All telemetry is synthetic.

> [!NOTE]
> **Future Nemotron Integration**: NVIDIA Nemotron and Nebius Token Factory APIs are **intentionally excluded** from this Local Core Build phase. This build establishes the rigorous deterministic engineering and diagnostic infrastructure that a future Nemotron autonomous agent will inspect through clean, deterministic local tools.

---

## 2. Problem Statement & Operational Significance

High-altitude, long-endurance (HALE) and tactical UAVs rely heavily on turbocharged internal combustion engines for propulsion. In uncrewed operations:
- **Subtle Degradation Precedes Failure**: Failures rarely happen instantaneously. Blocked cooling channels, fuel filter clogging, oil foaming/pressure drop, and bearing race fatigue exhibit slow, subtle trends hidden under atmospheric variations and autopilot throttle adjustments.
- **Flight Risk from Premature Aborts or Catastrophic Loss**: Unnecessary aborts jeopardize mission objectives, while undetected propulsion degradation risks hull loss or forced landing in contested terrain.
- **Explainability is Essential**: Flight controllers and autonomous decision-support systems require transparent, quantitative evidence—not black-box assertions.

---

## 3. Target System Architecture

```
                       +-------------------------------+
                       |     Mission Configuration     |
                       | (Altitude, Throttle, Duration)|
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |       Mission Profile         |
                       |  (Climb, Cruise, Ingress/RTB) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |     ISA Atmosphere Model      |
                       |    (T_amb, P_amb, Density)    |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       | Turbocharged Aero Digital Twin|
                       |  (Rotax 915 iS Physics Model) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       | Autonomous Hidden Degradation |
                       | (Stress Wear: Cooling, Oil,   |
                       |   Fuel, Bearing, Sensors)     |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |   Synthetic Sensor Modeling   |
                       | (Gaussian Noise, Bias, Drift) |
                       +---------------+---------------+
                                       |
                                       v
                       +-------------------------------+
                       |  ONE AUTHORITATIVE TELEMETRY  |
                       |         MISSION STATE         |
                       +-------+---+-------+---+-------+
                               |   |       |   |
            +------------------+   |       |   +------------------+
            |                      |       |                      |
            v                      v       v                      v
     +--------------+      +-----------+ +------------+   +---------------+
     | Trend Engine |      | Baseline  | | ML Fault   |   | Mission Risk  |
     | (Slopes,     |      | Comparator| | Classifier |   | Assurance     |
     |  Persistence)|      | (Healthy) | | (14 Feats) |   | (0-100 Score) |
     +-------+------+      +-----+-----+ +-----+------+   +-------+-------+
             |                   |             |                  |
             +-------------------+------+------+------------------+
                                        |
                                        v
                       +-------------------------------+
                       |  Streamlit Tactical Dashboard |
                       |  & Local Tools (Future API)   |
                       +-------------------------------+
```

---

## 4. Engineering & Physical Models

All engine constants are centralized in `config/engine_config.json` and loaded through `src/config.py`. Constants are never hardcoded or scattered across modules.

### 4.1 Engine Specification (Rotax 915 iS-Class)
- **Engine Type**: Turbocharged, fuel-injected, 4-cylinder, 4-stroke aero piston engine
- **Displacement**: 1,352 cc
- **Nominal Max Power**: 104 kW (140 hp) @ 5,800 RPM
- **Idle / Max RPM**: 1,800 RPM / 6,000 RPM
- **Compression Ratio**: 8.2 : 1
- **Service Ceiling / Range**: Up to 12,000 m (nominal cruise 1,000–8,000 m)

### 4.2 International Standard Atmosphere (ISA) (`src/atmosphere.py`)
Computes true ambient temperature, ambient pressure, and air density using standard barometric equations and tropospheric temperature lapse rate ($L = 0.0065\text{ K/m}$):
$$T_{\text{std}}(h) = T_0 - L \cdot h$$
$$P_{\text{std}}(h) = P_0 \left( \frac{T_{\text{std}}}{T_0} \right)^{\frac{g}{R \cdot L}}$$
$$\rho(h) = \frac{P_{\text{std}}}{R \cdot (T_{\text{std}} + \Delta T_{\text{offset}})}$$
The model prevents unphysical independent random variations: altitude rise strictly lowers pressure and density.

### 4.3 Turbocharger & Manifold Absolute Pressure (MAP) (`src/twin.py`)
Turbo boost actively compensates for altitude density loss up to the wastegate limit (target boost ~155 kPa, maximum 200 kPa). MAP is dynamically constrained by throttle position and ambient density:
$$\text{MAP} = \text{clamp}\left( P_{\text{amb}} \cdot (0.35 + 0.65 \cdot \text{Throttle}) + P_{\text{boost}} \cdot (0.20 + 0.80 \cdot \text{Throttle}), 25.0, 200.0 \right)$$

### 4.4 Thermal Dynamics & Inertia
Cylinder Head Temperature (CHT) and Oil Temperature exhibit continuous first-order thermal inertia:
$$\text{CHT}_{t} = \text{CHT}_{t-1} + \left(1 - e^{-\Delta t / \tau_{\text{cht}}}\right) \left(\text{CHT}_{\text{target}} - \text{CHT}_{t-1}\right)$$
Target temperatures scale with engine load, ambient temperature, altitude heat dissipation loss, and cooling health degradation.

### 4.5 Lubrication & Vibration
- **Oil Pressure**: Driven by positive-displacement mechanical pump (proportional to RPM), tempered by thermal viscosity thinnings and lubrication health degradation.
- **Vibration**: Base structural vibration scales with RPM and load ($0.025\text{–}0.035\text{ g}$). Bearing degradation adds persistent harmonic vibration ($+0.11\text{ g}$ at severe wear), and misfires add cyclic combustion imbalance spikes.

---

## 5. Autonomous Hidden Degradation (`src/degradation.py`)

In real operations, faults are not selected from dropdowns; they evolve organically.
- The mission always begins in a **pristine healthy state** ($1.0$ health across all subsystems).
- Degradation is driven by an **operating stress function**:
  $$\text{Stress} = 0.40 \cdot \text{ThrottleStress} + 0.25 \cdot \text{RPMStress} + 0.20 \cdot \text{AltitudeStress} + 0.15 \cdot \text{TempStress}$$
- Wear accumulates continuously at physical rates calibrated so that trends manifest naturally across the 60-minute mission duration.
- Observable consequences of specific degradation modes:
  - **Cooling Degradation**: CHT climbs progressively; thermal recovery after climb slows down.
  - **Oil Pressure Degradation**: Oil pressure decays while oil temperature rises.
  - **Bearing Degradation**: High-frequency vibration increases persistently.
  - **Fuel Restriction**: Delivered fuel flow drops, restricting available power.
  - **Sensor Drift**: Sensors accumulate bias and drift while underlying physics remains pristine.
  - **Misfire**: Correlated oscillations in RPM, power dips, and vibration surges.

---

## 6. Sensor Simulation Suite (`src/sensors.py`)

No ML model or flight controller observes pure ground truth. The sensor model injects:
- Realistic Gaussian instrument noise (e.g. $\pm 8\text{ RPM}$, $\pm 0.8^\circ\text{C CHT}$, $\pm 1.5\text{ kPa Oil P}$)
- Systematic calibration bias
- Uncalibrated thermal drift accumulation
- Transient dropout probability
- Full reproducibility via random seeds

All 14 observable features consumed by the diagnostics pipeline originate from this sensor suite.

---

## 7. Machine Learning Diagnostics (`src/diagnostics.py`)

### 7.1 Observable Feature Space (14 Observable Features)
The ML model strictly receives observable sensor telemetry. Hidden health states are never exposed:
1. `altitude_m`
2. `ambient_temp_c`
3. `ambient_pressure_kpa`
4. `throttle`
5. `rpm`
6. `map_kpa`
7. `engine_load`
8. `power_kw`
9. `fuel_flow_lph`
10. `cht_c`
11. `egt_c`
12. `oil_pressure_kpa`
13. `oil_temperature_c`
14. `vibration_g`

### 7.2 Fault Classes (7 Classes)
1. `normal`
2. `cooling_degradation`
3. `fuel_restriction`
4. `oil_pressure_degradation`
5. `bearing_degradation`
6. `sensor_drift`
7. `misfire`

### 7.3 Model Architecture & Scenario-Wise Validation
- **Algorithm**: `RandomForestClassifier` (250 estimators, max depth 14, min samples leaf 3, class_weight="balanced")
- **Scenario-Grouped Split**: `GroupShuffleSplit` (80% train / 20% test by `scenario_id`). Zero scenario overlap prevents time-series data leakage between training and testing folds.
- **Evaluation**: Held-out accuracy, precision, recall, macro F1, weighted F1, 7x7 confusion matrix, and feature importance rankings.
- **Model Versioning**: Serialized as `models/fault_classifier_v{N}.joblib` with evaluation metrics recorded in `models/model_metadata.json`.

---

## 8. Analytical & Decision Support Engines

### 8.1 Healthy Baseline Comparator (`src/baseline.py`)
Generates an ideal reference state for the exact current flight condition ($\text{Alt}, \text{Throttle}$) and computes absolute and percentage deviations for CHT, EGT, Oil Pressure, Oil Temp, Vibration, and Power.

### 8.2 Trend & Anomaly Engine (`src/trends.py`)
Computes linear regression slopes and statistical window deltas (first 10% vs. last 10% of mission). Signals are flagged as persistent if deviations exceed $2\sigma$ over sustained samples. Outputs system health status:
- **`NORMAL`**: All parameters tracking within limits.
- **`MONITORING`**: Early deviations detected; increased observation recommended.
- **`WARNING`**: Multiple deviations exhibiting statistical persistence.
- **`CRITICAL`**: Safety thresholds breached (e.g. CHT $> 165^\circ\text{C}$ or Oil Pressure $< 200\text{ kPa}$).

### 8.3 Mission Assurance & Risk Engine (`src/risk.py`)
Fuses thermal stress (30%), lubrication stress (25%), vibration stress (25%), and ML fault probability (20%) into a transparent 0–100 index:
- **`LOW`** (0–25)
- **`MODERATE`** (25–50)
- **`HIGH`** (50–75)
- **`CRITICAL`** (75–100)

### 8.4 Counterfactual "What-If" Explorer (`src/tools.py`)
Enables virtual counterfactual experiments: What if cooling health is 0.75? The simulator computes hypothetical telemetry and compares it against observed telemetry, calculating a normalized similarity score and Mean Squared Error.

---

## 9. Deterministic Local Tool Interface (`src/tools.py`)

These functions operate entirely locally with no external dependencies or LLM calls:
```python
from src.tools import (
    get_engine_state,
    get_sensor_history,
    run_fault_detection,
    compare_with_baseline,
    assess_mission_risk,
    run_what_if_scenario,
    compare_scenarios,
)

# 1. Query observable engine state
telemetry = get_engine_state(altitude_m=3000.0, throttle=0.75)

# 2. Run local ML fault diagnosis
diagnosis = run_fault_detection(telemetry)

# 3. Compare with healthy baseline
baseline_diff = compare_with_baseline(telemetry)

# 4. Assess mission assurance risk
risk = assess_mission_risk(telemetry, mission_duration_min=60.0)

# 5. Run counterfactual hypothesis test
what_if = run_what_if_scenario(
    altitude_m=3000.0,
    throttle=0.75,
    cooling_health=0.65,
    target_telemetry=telemetry,
)
```

---

## 10. Installation & Usage

### 10.1 Environment Setup
```powershell
# Navigate to project directory
cd D:\aerotwin\AeroTwin-Nemotron

# Activate existing virtual environment
.\.venv\Scripts\Activate.ps1

# Verify dependencies
pip install -r requirements.txt
```

### 10.2 Run Comprehensive Test Suite
```powershell
python -m unittest discover tests -v
```
All 16 unit tests covering atmosphere physics, digital twin dynamics, sensor noise, 1,200 sample generation, CSV export, dataset integrity, and ML inference pass with zero failures.

### 10.3 Launch the Streamlit Tactical Dashboard
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 11. Verification Checklist

- [x] Default mission produces $\ge 1,200$ samples (60 min @ 20 samples/min)
- [x] Baseline synthetic dataset contains $\ge 42,000$ rows across 7 classes
- [x] Exactly 14 observable features utilized (no hidden health leakage)
- [x] Scenario-wise GroupShuffleSplit prevents train/test data leakage
- [x] Actual held-out metrics calculated (Accuracy, Macro F1, Weighted F1)
- [x] Real confusion matrix and feature importances derived from classifier
- [x] Model versioning registry active in `models/model_metadata.json`
- [x] Authoritative mission telemetry exported to `data/mission_telemetry.csv`
- [x] Trend analysis, baseline comparison, and what-if simulation operational
- [x] Streamlit tactical dark UI launches cleanly with live interactive charts
- [x] Retraining pipeline supports adding new mission telemetry
- [x] Zero LLM calls, zero OpenAI/Nebius clients, zero external inference requests
