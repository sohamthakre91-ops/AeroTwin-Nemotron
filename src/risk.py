"""
AeroTwin-Nemotron: Mission Risk & Assurance Engine
Synthesizes ML fault probabilities, baseline deviations, thermal stress,
lubrication integrity, and mechanical vibration into an interpretable 0-100 risk score.

DISCLAIMER: This is a physics-informed research prototype and does NOT constitute
certified aviation flight safety risk determination.
"""

from typing import Dict, Any, Optional, List


def assess_mission_risk(
    telemetry: Dict[str, Any],
    baseline_result: Optional[Dict[str, Any]] = None,
    ml_diagnosis: Optional[Dict[str, Any]] = None,
    trend_result: Optional[Dict[str, Any]] = None,
    mission_duration_min: float = 60.0,
) -> Dict[str, Any]:
    """
    Computes local mission risk score (0-100) and risk level:
    LOW: 0 - 25
    MODERATE: 25 - 50
    HIGH: 50 - 75
    CRITICAL: 75 - 100
    """
    # 1. Thermal Stress Component (0 - 30 points)
    cht = float(telemetry.get("cht_c", 140.0))
    egt = float(telemetry.get("egt_c", 720.0))
    thermal_excess = max(0.0, cht - 145.0) / 35.0  # Normalized over [145, 180]
    egt_excess = max(0.0, egt - 760.0) / 100.0     # Normalized over [760, 860]
    thermal_score = min(30.0, (thermal_excess * 20.0 + egt_excess * 10.0))

    # 2. Lubrication System Stress (0 - 25 points)
    oil_p = float(telemetry.get("oil_pressure_kpa", 380.0))
    oil_t = float(telemetry.get("oil_temperature_c", 85.0))
    oil_p_loss = max(0.0, 320.0 - oil_p) / 170.0   # Normalized drop towards 150 kPa
    oil_t_excess = max(0.0, oil_t - 95.0) / 35.0   # Normalized rise towards 130 C
    lubrication_score = min(25.0, (oil_p_loss * 18.0 + oil_t_excess * 7.0))

    # 3. Mechanical Vibration Stress (0 - 25 points)
    vibe = float(telemetry.get("vibration_g", 0.025))
    vibe_excess = max(0.0, vibe - 0.040) / 0.080   # Normalized above 0.040 g
    vibration_score = min(25.0, vibe_excess * 25.0)

    # 4. ML Fault Hypothesis Contribution (0 - 20 points)
    ml_score = 0.0
    active_hypotheses = []
    if ml_diagnosis and "error" not in ml_diagnosis:
        pred_fault = ml_diagnosis.get("predicted_fault", "normal")
        conf = ml_diagnosis.get("confidence", 0.0)
        if pred_fault != "normal":
            ml_score = min(20.0, conf * 20.0)
            active_hypotheses.append(f"{pred_fault.replace('_', ' ').title()} ({conf * 100:.1f}%)")

    # 5. Baseline Deviation & Trend Persistence Multipliers
    persistence_factor = 1.0
    if trend_result:
        signals = trend_result.get("signals", {})
        persistent_count = sum(1 for s in signals.values() if s.get("is_persistent", False))
        if persistent_count >= 2:
            persistence_factor = 1.15

    duration_factor = min(1.20, 0.85 + 0.15 * (mission_duration_min / 60.0))

    # Raw risk summation (max 100)
    raw_risk = (thermal_score + lubrication_score + vibration_score + ml_score) * persistence_factor * duration_factor
    risk_score = round(max(0.0, min(100.0, raw_risk)), 1)

    # Risk level classification
    if risk_score < 25.0:
        risk_level = "LOW"
        mission_impact = "Propulsion system operating within normal envelope. Mission continuation recommended."
    elif risk_score < 50.0:
        risk_level = "MODERATE"
        mission_impact = "Mild thermal or lubrication degradation detected. Recommend heightened monitoring and altitude/speed optimization."
    elif risk_score < 75.0:
        risk_level = "HIGH"
        mission_impact = "Substantial subsystem degradation observed. Plan contingency ingress path or power reduction."
    else:
        risk_level = "CRITICAL"
        mission_impact = "Severe propulsion anomaly detected (potential thermal runaway, loss of oil pressure, or high vibration). Immediate RTB or forced descent advisory."

    # Uncertainty estimation based on noise and evidence agreement
    uncertainty_pct = 12.0
    if ml_diagnosis and ml_diagnosis.get("confidence", 0.0) > 0.85:
        uncertainty_pct = 6.5
    elif ml_diagnosis and ml_diagnosis.get("confidence", 0.0) < 0.40:
        uncertainty_pct = 22.0

    indicators = {
        "thermal_stress_score": round(thermal_score, 1),
        "lubrication_stress_score": round(lubrication_score, 1),
        "vibration_stress_score": round(vibration_score, 1),
        "ml_diagnostic_score": round(ml_score, 1),
        "active_hypotheses": active_hypotheses,
    }

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "mission_impact": mission_impact,
        "uncertainty_percent": uncertainty_pct,
        "indicators": indicators,
        "disclaimer": "RESEARCH PROTOTYPE ONLY: Not certified for real-world aircraft operations.",
    }
