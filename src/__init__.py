"""
AeroTwin-Nemotron: Autonomous UAV Propulsion Health & Mission Assurance
Physics-informed Digital Twin, ML Diagnostics & Mission Assurance Platform.
"""

from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .atmosphere import isa_atmosphere
from .twin import AeroTwin, EngineState
from .sensors import SensorModel
from .degradation import AutonomousDegradationSimulator, DegradationState
from .mission import MissionSimulator
from .dataset import generate_full_dataset, FAULT_CLASSES, FEATURE_COLUMNS
from .diagnostics import train_fault_classifier, predict_fault, load_metadata
from .baseline import compare_with_baseline, BaselineComparator
from .trends import analyze_trends, assess_system_status
from .risk import assess_mission_risk
from .tools import (
    get_engine_state,
    get_sensor_history,
    run_fault_detection,
    run_what_if_scenario,
    compare_scenarios,
)

__all__ = [
    "EngineConfig",
    "DEFAULT_ENGINE_CONFIG",
    "isa_atmosphere",
    "AeroTwin",
    "EngineState",
    "SensorModel",
    "AutonomousDegradationSimulator",
    "DegradationState",
    "MissionSimulator",
    "generate_full_dataset",
    "FAULT_CLASSES",
    "FEATURE_COLUMNS",
    "train_fault_classifier",
    "predict_fault",
    "load_metadata",
    "compare_with_baseline",
    "BaselineComparator",
    "analyze_trends",
    "assess_system_status",
    "assess_mission_risk",
    "get_engine_state",
    "get_sensor_history",
    "run_fault_detection",
    "run_what_if_scenario",
    "compare_scenarios",
]
