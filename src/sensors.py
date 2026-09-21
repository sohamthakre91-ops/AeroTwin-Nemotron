"""
AeroTwin-Nemotron: Synthetic Sensor Simulation Model
Converts true physics states to realistic sensor telemetry by introducing
Gaussian noise, systematic bias, time-accumulated drift, and dropout probabilities.
"""

from dataclasses import dataclass
import random
from typing import Dict, Any, Optional

from .twin import EngineState


@dataclass
class SensorChannelConfig:
    noise_std: float = 0.01
    bias: float = 0.0
    drift_per_min: float = 0.0
    dropout_probability: float = 0.0


class SensorModel:
    """
    Sensor suite modeling measurement imperfections for all 14 observable telemetry parameters.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

        # Baseline realistic instrument noise configurations
        self.configs: Dict[str, SensorChannelConfig] = {
            "rpm": SensorChannelConfig(noise_std=8.0, bias=0.0),
            "map_kpa": SensorChannelConfig(noise_std=0.4, bias=0.0),
            "engine_load": SensorChannelConfig(noise_std=0.005, bias=0.0),
            "power_kw": SensorChannelConfig(noise_std=0.5, bias=0.0),
            "fuel_flow_lph": SensorChannelConfig(noise_std=0.18, bias=0.0),
            "cht_c": SensorChannelConfig(noise_std=0.8, bias=0.0),
            "egt_c": SensorChannelConfig(noise_std=2.5, bias=0.0),
            "oil_pressure_kpa": SensorChannelConfig(noise_std=1.5, bias=0.0),
            "oil_temperature_c": SensorChannelConfig(noise_std=0.6, bias=0.0),
            "vibration_g": SensorChannelConfig(noise_std=0.002, bias=0.0),
            "altitude_m": SensorChannelConfig(noise_std=2.0, bias=0.0),
            "ambient_temp_c": SensorChannelConfig(noise_std=0.3, bias=0.0),
            "ambient_pressure_kpa": SensorChannelConfig(noise_std=0.15, bias=0.0),
            "throttle": SensorChannelConfig(noise_std=0.001, bias=0.0),
        }

        # Accumulated drift states (per channel)
        self.accumulated_drift: Dict[str, float] = {k: 0.0 for k in self.configs}

    def reset(self):
        """Reset sensor drifts and RNG."""
        self.rng = random.Random(self.seed)
        self.accumulated_drift = {k: 0.0 for k in self.configs}

    def configure_channel_fault(
        self,
        channel: str,
        bias: float = 0.0,
        drift_per_min: float = 0.0,
        dropout_probability: float = 0.0,
    ):
        """Inject specific sensor channel fault (bias, drift rate, or dropout)."""
        if channel in self.configs:
            cfg = self.configs[channel]
            self.configs[channel] = SensorChannelConfig(
                noise_std=cfg.noise_std,
                bias=bias,
                drift_per_min=drift_per_min,
                dropout_probability=dropout_probability,
            )

    def measure(
        self,
        channel: str,
        true_value: float,
        dt_minutes: float = 0.05,
        sensor_health: float = 1.0,
    ) -> Optional[float]:
        """
        Convert ground-truth value to synthetic sensor reading.
        Applies health degradation (increased drift/bias if sensor_health < 1.0).
        """
        if channel not in self.configs:
            return true_value

        cfg = self.configs[channel]

        # Accumulate systematic drift
        drift_rate = cfg.drift_per_min
        # When sensor health is degraded, additional uncalibrated thermal drift emerges
        if sensor_health < 0.99:
            degradation_drift = (1.0 - sensor_health) * 0.15
            drift_rate += degradation_drift

        self.accumulated_drift[channel] += drift_rate * dt_minutes

        # Dropout check
        if cfg.dropout_probability > 0.0 and self.rng.random() < cfg.dropout_probability:
            return None

        # Gaussian measurement noise
        noise = self.rng.gauss(0.0, cfg.noise_std)

        # Additional degraded health noise
        if sensor_health < 0.99:
            noise *= (1.0 + (1.0 - sensor_health) * 2.0)

        measured = true_value + cfg.bias + self.accumulated_drift[channel] + noise
        return measured

    def measure_state(
        self,
        state: EngineState,
        dt_minutes: float = 0.05,
        sensor_health: float = 1.0,
    ) -> Dict[str, float]:
        """
        Produce noisy, drifting observable telemetry dictionary from EngineState.
        Guarantees physical non-negative boundaries for appropriate quantities.
        """
        raw = state.to_dict()
        measured: Dict[str, float] = {}

        for key, val in raw.items():
            if key in self.configs:
                m_val = self.measure(key, val, dt_minutes, sensor_health)
                if m_val is None:
                    m_val = val  # Fallback if dropped
                measured[key] = round(m_val, 4 if "g" in key or "load" in key or "throttle" in key or "density" in key else 2)
            else:
                # E.g. mission_time_min, air_density_kg_m3
                measured[key] = val

        # Boundary protections on noisy measurements
        if "rpm" in measured:
            measured["rpm"] = max(0.0, measured["rpm"])
        if "map_kpa" in measured:
            measured["map_kpa"] = max(10.0, measured["map_kpa"])
        if "oil_pressure_kpa" in measured:
            measured["oil_pressure_kpa"] = max(0.0, measured["oil_pressure_kpa"])
        if "vibration_g" in measured:
            measured["vibration_g"] = max(0.005, measured["vibration_g"])
        if "power_kw" in measured:
            measured["power_kw"] = max(0.0, measured["power_kw"])
        if "fuel_flow_lph" in measured:
            measured["fuel_flow_lph"] = max(0.0, measured["fuel_flow_lph"])

        return measured