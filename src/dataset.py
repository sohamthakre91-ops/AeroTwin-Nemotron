"""
AeroTwin-Nemotron: Synthetic Training Dataset Generator
Generates >= 42,000 rows across 7 propulsion fault classes with temporal correlation
and scenario grouping to prevent train/test data leakage.
"""

from pathlib import Path
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd

from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .twin import AeroTwin, EngineState
from .sensors import SensorModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TELEMETRY_CSV_PATH = DATA_DIR / "telemetry.csv"
TRAINING_CSV_PATH = DATA_DIR / "model_training_data.csv"

FAULT_CLASSES = [
    "normal",
    "cooling_degradation",
    "fuel_restriction",
    "oil_pressure_degradation",
    "bearing_degradation",
    "sensor_drift",
    "misfire",
]

FEATURE_COLUMNS = [
    "altitude_m",
    "ambient_temp_c",
    "ambient_pressure_kpa",
    "throttle",
    "rpm",
    "map_kpa",
    "engine_load",
    "power_kw",
    "fuel_flow_lph",
    "cht_c",
    "egt_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
    "vibration_g",
]

TARGET_COLUMN = "fault"


def generate_scenario_data(
    scenario_id: int,
    fault_name: str,
    steps: int = 120,
    seed: int = 42,
    config: Optional[EngineConfig] = None,
) -> pd.DataFrame:
    """
    Generate one cohesive time-series flight scenario (120 timesteps).
    Fault severity smoothly evolves over time (e.g. 0.05 -> 0.10 -> 0.15 -> ...).
    Preserves temporal correlations and realistic operating point variations.
    """
    rng = random.Random(seed)
    cfg = config or DEFAULT_ENGINE_CONFIG

    twin = AeroTwin(config=cfg)
    sensors = SensorModel(seed=seed)

    # Initial flight conditions for this scenario
    base_altitude = rng.uniform(1500.0, 5500.0)
    base_throttle = rng.uniform(0.55, 0.85)
    temp_offset = rng.uniform(-8.0, 8.0)

    # Max severity for degraded scenarios [0.35, 0.85]
    target_severity = 0.0 if fault_name == "normal" else rng.uniform(0.35, 0.85)

    # If sensor drift fault, configure sensor channel drifts
    if fault_name == "sensor_drift":
        sensors.configure_channel_fault(
            "cht_c",
            bias=rng.uniform(4.0, 12.0) * target_severity,
            drift_per_min=rng.uniform(0.25, 0.60) * target_severity,
        )
        sensors.configure_channel_fault(
            "egt_c",
            bias=rng.uniform(10.0, 25.0) * target_severity,
            drift_per_min=rng.uniform(0.40, 0.90) * target_severity,
        )
        sensors.configure_channel_fault(
            "oil_pressure_kpa",
            bias=-rng.uniform(15.0, 40.0) * target_severity,
            drift_per_min=-rng.uniform(0.20, 0.50) * target_severity,
        )

    rows: List[Dict[str, Any]] = []
    prev_state: Optional[EngineState] = None

    for step in range(steps):
        time_min = step * 0.05  # 0.05 min timestep
        progress = step / max(1, steps - 1)

        # Realistic smooth autopilot altitude and throttle variations
        alt = base_altitude + 80.0 * math.sin(step / 14.0) + rng.uniform(-10.0, 10.0)
        thr = base_throttle + 0.02 * math.cos(step / 11.0) + rng.uniform(-0.005, 0.005)

        # Clamping
        alt = max(500.0, min(10000.0, alt))
        thr = max(0.35, min(0.95, thr))

        # Temporal evolution of fault severity within the scenario
        # Mild onset evolving to pronounced severity
        curr_severity = target_severity * (0.15 + 0.85 * (progress ** 1.3))

        # Map current fault to twin health arguments
        cooling_h = 1.0
        fuel_h = 1.0
        oil_h = 1.0
        bearing_h = 1.0
        sensor_h = 1.0
        misfire_sev = 0.0

        if fault_name == "cooling_degradation":
            cooling_h = 1.0 - curr_severity
        elif fault_name == "fuel_restriction":
            fuel_h = 1.0 - curr_severity
        elif fault_name == "oil_pressure_degradation":
            oil_h = 1.0 - curr_severity
        elif fault_name == "bearing_degradation":
            bearing_h = 1.0 - curr_severity
        elif fault_name == "sensor_drift":
            sensor_h = 1.0 - curr_severity
        elif fault_name == "misfire":
            misfire_sev = curr_severity

        # Simulate Digital Twin physics
        true_state = twin.simulate(
            altitude_m=alt,
            throttle=thr,
            ambient_temp_offset_c=temp_offset,
            cooling_health=cooling_h,
            fuel_health=fuel_h,
            oil_health=oil_h,
            bearing_health=bearing_h,
            sensor_health=sensor_h,
            misfire_severity=misfire_sev,
            mission_time_min=time_min,
            dt_min=0.05,
            prev_state=prev_state,
        )
        prev_state = true_state

        # Apply realistic sensor model
        measured = sensors.measure_state(
            true_state,
            dt_minutes=0.05,
            sensor_health=sensor_h,
        )

        row = {
            "scenario_id": scenario_id,
            "step": step,
            "mission_time_min": round(time_min, 3),
            "altitude_m": measured["altitude_m"],
            "ambient_temp_c": measured["ambient_temp_c"],
            "ambient_pressure_kpa": measured["ambient_pressure_kpa"],
            "throttle": measured["throttle"],
            "rpm": measured["rpm"],
            "map_kpa": measured["map_kpa"],
            "engine_load": measured["engine_load"],
            "power_kw": measured["power_kw"],
            "fuel_flow_lph": measured["fuel_flow_lph"],
            "cht_c": measured["cht_c"],
            "egt_c": measured["egt_c"],
            "oil_pressure_kpa": measured["oil_pressure_kpa"],
            "oil_temperature_c": measured["oil_temperature_c"],
            "vibration_g": measured["vibration_g"],
            "fault": fault_name,
            "fault_severity": round(curr_severity, 4),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_full_dataset(
    scenarios_per_class: int = 50,
    steps_per_scenario: int = 120,
    output_path: Optional[Path] = None,
    seed_base: int = 1000,
) -> pd.DataFrame:
    """
    Generate synthetic dataset across all 7 fault classes.
    7 classes * 50 scenarios * 120 steps = 42,000 rows.
    """
    all_dfs = []
    current_scenario_id = 1

    for fault_class in FAULT_CLASSES:
        for i in range(scenarios_per_class):
            scenario_seed = seed_base + current_scenario_id
            df_scenario = generate_scenario_data(
                scenario_id=current_scenario_id,
                fault_name=fault_class,
                steps=steps_per_scenario,
                seed=scenario_seed,
            )
            all_dfs.append(df_scenario)
            current_scenario_id += 1

    full_df = pd.concat(all_dfs, ignore_index=True)

    dest_path = output_path or TELEMETRY_CSV_PATH
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(dest_path, index=False)

    # Also save as model_training_data.csv for convenience
    copy_path = DATA_DIR / "model_training_data.csv"
    full_df.to_csv(copy_path, index=False)

    return full_df


if __name__ == "__main__":
    print("Generating 42,000-row synthetic dataset...")
    df = generate_full_dataset()
    print(f"Generated {len(df):,} rows with {df['scenario_id'].nunique()} scenarios.")
    print(df["fault"].value_counts())