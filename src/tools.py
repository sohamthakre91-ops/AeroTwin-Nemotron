"""
AeroTwin-Nemotron: Local Engineering Tool Interface
Provides clean, deterministic, local inspection tools designed for future autonomous
investigation by NVIDIA Nemotron.

CRITICAL ARCHITECTURAL CONSTRAINTS:
- No LLM API calls.
- No Nebius Token Factory calls.
- No external network requests.
- Purely local deterministic physics and numerical computation.
"""

from pathlib import Path
import math
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .twin import AeroTwin, EngineState
from .sensors import SensorModel
from .diagnostics import predict_fault, FEATURE_COLUMNS
from .baseline import compare_with_baseline as baseline_compare
from .risk import assess_mission_risk as risk_assess
from .trends import analyze_trends, assess_system_status

MISSION_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "mission_telemetry.csv"


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely converts string or number to float."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def get_engine_state(
    altitude_m: float = 3000.0,
    throttle: float = 0.75,
    ambient_temp_offset_c: float = 0.0,
    mission_time_min: float = 0.0,
    cooling_health: float = 1.0,
    fuel_health: float = 1.0,
    oil_health: float = 1.0,
    bearing_health: float = 1.0,
    sensor_health: float = 1.0,
) -> Dict[str, float]:
    """
    Generate current Digital Twin telemetry under specified flight conditions.
    Internal health variables can be specified for testing, but the returned
    dictionary strictly contains ONLY observable telemetry features.
    """
    twin = AeroTwin()
    sensors = SensorModel(seed=42)

    state = twin.simulate(
        altitude_m=_safe_float(altitude_m, 3000.0),
        throttle=_safe_float(throttle, 0.75),
        ambient_temp_offset_c=_safe_float(ambient_temp_offset_c, 0.0),
        cooling_health=_safe_float(cooling_health, 1.0),
        fuel_health=_safe_float(fuel_health, 1.0),
        oil_health=_safe_float(oil_health, 1.0),
        bearing_health=_safe_float(bearing_health, 1.0),
        sensor_health=_safe_float(sensor_health, 1.0),
        mission_time_min=_safe_float(mission_time_min, 0.0),
    )

    measured = sensors.measure_state(state, sensor_health=_safe_float(sensor_health, 1.0))
    return measured


def get_sensor_history(
    limit: int = 50,
    file_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve recent authoritative mission telemetry history.
    """
    path = file_path or MISSION_CSV_PATH
    if not path.exists():
        return []

    try:
        df = pd.read_csv(path)
        records = df.tail(limit).to_dict(orient="records")
        return records
    except Exception:
        return []


def run_fault_detection(
    telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute ML fault diagnosis on observable telemetry.
    If no telemetry is provided, pulls the latest row from mission history.
    """
    if telemetry is None:
        hist = get_sensor_history(limit=1)
        if not hist:
            return {"error": "No telemetry provided and no mission history available."}
        telemetry = hist[-1]

    return predict_fault(telemetry)


def compare_with_baseline(
    telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare observable telemetry against healthy Digital Twin reference at same altitude & throttle.
    """
    if telemetry is None:
        hist = get_sensor_history(limit=1)
        if not hist:
            return {"error": "No telemetry provided and no mission history available."}
        telemetry = hist[-1]

    return baseline_compare(telemetry)


def assess_mission_risk(
    telemetry: Optional[Dict[str, Any]] = None,
    mission_duration_min: float = 60.0,
) -> Dict[str, Any]:
    """
    Evaluate overall propulsion mission assurance risk (0-100 score and risk level).
    """
    if telemetry is None:
        hist = get_sensor_history(limit=1)
        if not hist:
            return {"error": "No telemetry provided and no mission history available."}
        telemetry = hist[-1]

    baseline = baseline_compare(telemetry)
    diagnosis = predict_fault(telemetry)
    trends = None

    history = get_sensor_history(limit=1200)
    if history:
        trends = analyze_trends(history)

    return risk_assess(
        telemetry=telemetry,
        baseline_result=baseline,
        ml_diagnosis=diagnosis,
        trend_result=trends,
        mission_duration_min=mission_duration_min,
    )


def run_what_if_scenario(
    altitude_m: float = 3000.0,
    throttle: float = 0.75,
    cooling_health: float = 1.0,
    fuel_health: float = 1.0,
    oil_health: float = 1.0,
    bearing_health: float = 1.0,
    sensor_health: float = 1.0,
    target_telemetry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Counterfactual simulation: evaluate hypothetical subsystem health states
    and optionally calculate similarity error against an observed target telemetry.
    """
    hypo_telemetry = get_engine_state(
        altitude_m=altitude_m,
        throttle=throttle,
        cooling_health=cooling_health,
        fuel_health=fuel_health,
        oil_health=oil_health,
        bearing_health=bearing_health,
        sensor_health=sensor_health,
    )

    result = {
        "inputs": {
            "altitude_m": altitude_m,
            "throttle": throttle,
            "cooling_health": cooling_health,
            "fuel_health": fuel_health,
            "oil_health": oil_health,
            "bearing_health": bearing_health,
            "sensor_health": sensor_health,
        },
        "hypothetical_telemetry": hypo_telemetry,
    }

    if target_telemetry:
        # Compare key observable signals
        comparison_keys = ["cht_c", "egt_c", "oil_pressure_kpa", "oil_temperature_c", "vibration_g", "power_kw"]
        deltas = {}
        sq_errs = []
        for k in comparison_keys:
            if k in hypo_telemetry and k in target_telemetry:
                diff = hypo_telemetry[k] - float(target_telemetry[k])
                deltas[k] = round(diff, 3)
                norm_scale = max(1.0, abs(float(target_telemetry[k])))
                sq_errs.append((diff / norm_scale) ** 2)

        mse = float(np.mean(sq_errs)) if sq_errs else 0.0
        similarity_score = max(0.0, 100.0 * (1.0 - math.sqrt(mse)))

        result["comparison"] = {
            "deltas": deltas,
            "similarity_score_percent": round(similarity_score, 1),
            "mean_squared_normalized_error": round(mse, 4),
        }

    return result


def compare_scenarios(
    scenario_a: Dict[str, Any],
    scenario_b: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compares two telemetry dictionaries or scenario states, highlighting deltas.
    """
    all_keys = sorted(list(set(scenario_a.keys()).union(set(scenario_b.keys()))))
    comparison: Dict[str, Dict[str, Any]] = {}

    for k in all_keys:
        val_a = scenario_a.get(k)
        val_b = scenario_b.get(k)
        if val_a is not None and val_b is not None:
            try:
                fa = float(val_a)
                fb = float(val_b)
                comparison[k] = {
                    "scenario_a": fa,
                    "scenario_b": fb,
                    "delta": round(fb - fa, 3),
                    "percent_change": round((fb - fa) / abs(fa) * 100.0, 2) if abs(fa) > 1e-6 else 0.0,
                }
            except (ValueError, TypeError):
                comparison[k] = {"scenario_a": val_a, "scenario_b": val_b}

    return {"differences": comparison}