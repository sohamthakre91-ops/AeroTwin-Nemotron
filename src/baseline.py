"""
AeroTwin-Nemotron: Healthy Baseline Comparison Module
Evaluates observed propulsion telemetry against a simulated healthy Digital Twin
operating under identical flight conditions (altitude, throttle, ambient).
"""

from typing import Dict, Any, Optional, List
from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .twin import AeroTwin, EngineState


class BaselineComparator:
    """
    Simulates a pristine healthy engine reference at the exact same operating point
    and computes absolute and percentage deviations for key propulsion parameters.
    """

    COMPARISON_FIELDS = [
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

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or DEFAULT_ENGINE_CONFIG
        self.twin = AeroTwin(config=self.config)

    def generate_baseline(
        self,
        altitude_m: float,
        throttle: float,
        ambient_temp_offset_c: float = 0.0,
        mission_time_min: float = 0.0,
    ) -> Dict[str, float]:
        """Generate healthy Digital Twin reference at given operating condition."""
        state: EngineState = self.twin.simulate(
            altitude_m=altitude_m,
            throttle=throttle,
            ambient_temp_offset_c=ambient_temp_offset_c,
            cooling_health=1.0,
            fuel_health=1.0,
            oil_health=1.0,
            bearing_health=1.0,
            sensor_health=1.0,
            misfire_severity=0.0,
            mission_time_min=mission_time_min,
        )
        return state.to_dict()

    def compare(
        self,
        telemetry: Dict[str, Any],
        threshold_percent: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Compare observed telemetry with healthy baseline.
        Returns:
            operating_point: altitude, throttle
            baseline: healthy dictionary
            deviations: map of field -> {observed, healthy_baseline, delta, percent_deviation}
            significant_deviations: list of deviations exceeding threshold_percent
        """
        try:
            altitude_m = float(telemetry.get("altitude_m", 3000.0))
            throttle = float(telemetry.get("throttle", 0.75))
            ambient_offset = float(telemetry.get("ambient_temp_offset_c", 0.0))
            mission_time = float(telemetry.get("mission_time_min", 0.0))
        except (ValueError, TypeError):
            altitude_m = 3000.0
            throttle = 0.75
            ambient_offset = 0.0
            mission_time = 0.0

        baseline = self.generate_baseline(
            altitude_m=altitude_m,
            throttle=throttle,
            ambient_temp_offset_c=ambient_offset,
            mission_time_min=mission_time,
        )

        deviations: Dict[str, Dict[str, float]] = {}
        significant: List[Dict[str, Any]] = []

        for field in self.COMPARISON_FIELDS:
            obs = telemetry.get(field)
            exp = baseline.get(field)

            if obs is None or exp is None:
                continue

            try:
                obs_f = float(obs)
                exp_f = float(exp)
            except (ValueError, TypeError):
                continue

            delta = obs_f - exp_f
            pct = (delta / abs(exp_f) * 100.0) if abs(exp_f) > 1e-6 else 0.0

            dev_info = {
                "observed": round(obs_f, 3),
                "healthy_baseline": round(exp_f, 3),
                "delta": round(delta, 3),
                "percent_deviation": round(pct, 2),
            }
            deviations[field] = dev_info

            if abs(pct) >= threshold_percent:
                significant.append({
                    "parameter": field,
                    "observed": round(obs_f, 3),
                    "healthy_baseline": round(exp_f, 3),
                    "delta": round(delta, 3),
                    "percent_deviation": round(pct, 2),
                })

        significant.sort(key=lambda x: abs(x["percent_deviation"]), reverse=True)

        return {
            "operating_point": {
                "altitude_m": altitude_m,
                "throttle": throttle,
            },
            "baseline": baseline,
            "deviations": deviations,
            "significant_deviations": significant,
        }


# Module level convenience function
_DEFAULT_COMPARATOR = BaselineComparator()

def compare_with_baseline(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    return _DEFAULT_COMPARATOR.compare(telemetry)
