"""
AeroTwin-Nemotron: Autonomous Hidden Degradation Engine
Simulates stress-induced mechanical and thermal wear.
The user does NOT manually pick a fault; degradation evolves autonomously.
"""

from dataclasses import dataclass, asdict
import random
import math
from typing import Dict, Any, Optional


@dataclass
class DegradationState:
    cooling_health: float = 1.0
    fuel_health: float = 1.0
    oil_health: float = 1.0
    bearing_health: float = 1.0
    sensor_health: float = 1.0
    misfire_severity: float = 0.0
    elapsed_minutes: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class AutonomousDegradationSimulator:
    """
    Evolves internal health states based on operating stress factors:
    - Throttle stress (40%)
    - RPM stress (25%)
    - Altitude stress (20%)
    - Ambient Temperature stress (15%)
    """

    def __init__(self, seed: int = 42, active_mode: Optional[str] = None, fault_rate_multiplier: float = 1.0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.state = DegradationState()
        self.active_mode = active_mode  # Can be None (natural autonomous) or specific fault type for training
        self.fault_rate_multiplier = fault_rate_multiplier

    def reset(self):
        """Reset degradation state to healthy pristine engine condition."""
        self.state = DegradationState()
        self.rng = random.Random(self.seed)

    def calculate_operating_stress(
        self,
        throttle: float,
        rpm: float,
        ambient_temp_c: float,
        altitude_m: float,
    ) -> float:
        """
        Computes composite operating stress [0.0 - 1.0].
        Weights: throttle (40%), RPM (25%), altitude (20%), ambient temp (15%).
        """
        throttle_stress = max(0.0, min(1.0, (throttle - 0.20) / 0.80))
        rpm_stress = max(0.0, min(1.0, (rpm - 1800.0) / (6000.0 - 1800.0)))
        altitude_stress = max(0.0, min(1.0, altitude_m / 10000.0))
        temp_stress = max(0.0, min(1.0, (ambient_temp_c - 15.0) / 35.0))

        stress = (
            0.40 * throttle_stress
            + 0.25 * rpm_stress
            + 0.20 * altitude_stress
            + 0.15 * temp_stress
        )
        return max(0.0, min(1.0, stress))

    def step(
        self,
        throttle: float,
        rpm: float,
        ambient_temp_c: float,
        altitude_m: float,
        dt_minutes: float = 0.05,
    ) -> DegradationState:
        """
        Advance degradation by dt_minutes based on operating stress.
        Degradation is deliberately calibrated to be slow and realistic.
        """
        stress = self.calculate_operating_stress(throttle, rpm, ambient_temp_c, altitude_m)
        self.state.elapsed_minutes += dt_minutes

        # Base wear rate per minute under nominal stress
        base_rate = (dt_minutes / 60.0) * self.fault_rate_multiplier

        if self.active_mode is None or self.active_mode == "natural":
            # Natural autonomous mission degradation:
            # Under high sustained cruise stress (e.g. throttle 0.75+, 3000m+ altitude),
            # cooling and oil systems accumulate mild, observable wear over 60 mins.
            cooling_damage = base_rate * 0.08 * (0.3 + 1.2 * stress)
            oil_damage = base_rate * 0.06 * (0.2 + 1.0 * stress)
            bearing_damage = base_rate * 0.04 * (0.1 + 0.8 * stress)
            fuel_damage = base_rate * 0.02 * (0.1 + 0.5 * stress)
            sensor_damage = base_rate * 0.01 * (0.05 + 0.3 * stress)

            self.state.cooling_health -= cooling_damage
            self.state.oil_health -= oil_damage
            self.state.bearing_health -= bearing_damage
            self.state.fuel_health -= fuel_damage
            self.state.sensor_health -= sensor_damage

        elif self.active_mode == "cooling_degradation":
            # Focused cooling degradation scenario
            damage = base_rate * 0.45 * (0.6 + 0.9 * stress)
            self.state.cooling_health -= damage
            self.state.oil_health -= damage * 0.25

        elif self.active_mode == "oil_pressure_degradation":
            # Focused lubrication degradation scenario
            damage = base_rate * 0.42 * (0.5 + 0.9 * stress)
            self.state.oil_health -= damage

        elif self.active_mode == "bearing_degradation":
            # Focused mechanical bearing degradation scenario
            damage = base_rate * 0.40 * (0.5 + 1.0 * stress)
            self.state.bearing_health -= damage

        elif self.active_mode == "fuel_restriction":
            # Focused fuel restriction scenario
            damage = base_rate * 0.38 * (0.4 + 1.1 * stress)
            self.state.fuel_health -= damage

        elif self.active_mode == "sensor_drift":
            # Focused sensor degradation
            damage = base_rate * 0.40 * (0.5 + 0.8 * stress)
            self.state.sensor_health -= damage

        elif self.active_mode == "misfire":
            # Intermittent combustion degradation
            progress = min(1.0, self.state.elapsed_minutes / 40.0)
            self.state.misfire_severity = min(0.85, progress * 0.65)
            self.state.fuel_health -= base_rate * 0.15

        # Clamp all health bounds strictly within [0.0, 1.0]
        self.state.cooling_health = max(0.0, min(1.0, self.state.cooling_health))
        self.state.oil_health = max(0.0, min(1.0, self.state.oil_health))
        self.state.bearing_health = max(0.0, min(1.0, self.state.bearing_health))
        self.state.fuel_health = max(0.0, min(1.0, self.state.fuel_health))
        self.state.sensor_health = max(0.0, min(1.0, self.state.sensor_health))
        self.state.misfire_severity = max(0.0, min(1.0, self.state.misfire_severity))

        return self.state

    def hidden_state(self) -> Dict[str, float]:
        """Internal evaluator only. Never expose to public telemetry."""
        return self.state.to_dict()