"""
AeroTwin-Nemotron: Continuous Mission Simulator
Generates continuous high-fidelity UAV propulsion telemetry across realistic mission profiles.
Default: 60 minutes @ 20 samples/min = 1,200 samples.
"""

from pathlib import Path
import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .twin import AeroTwin, EngineState
from .sensors import SensorModel
from .degradation import AutonomousDegradationSimulator, DegradationState

MISSION_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "mission_telemetry.csv"

REQUIRED_TELEMETRY_COLUMNS = [
    "mission_time_min",
    "altitude_m",
    "ambient_temp_c",
    "ambient_pressure_kpa",
    "air_density_kg_m3",
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


class MissionSimulator:
    """
    Executes continuous mission simulation incorporating:
    Mission Profile -> ISA Atmosphere -> Digital Twin -> Autonomous Degradation -> Sensor Model
    """

    def __init__(
        self,
        target_altitude_m: float = 3000.0,
        base_throttle: float = 0.75,
        duration_minutes: float = 60.0,
        samples_per_minute: int = 20,
        seed: int = 42,
        config: Optional[EngineConfig] = None,
        fault_rate_multiplier: float = 1.0,
    ):
        self.target_altitude_m = target_altitude_m
        self.base_throttle = base_throttle
        self.duration_minutes = duration_minutes
        self.samples_per_minute = samples_per_minute
        self.timestep_min = 1.0 / float(samples_per_minute)  # 0.05 min
        self.seed = seed
        self.config = config or DEFAULT_ENGINE_CONFIG

        self.twin = AeroTwin(config=self.config)
        self.sensor_model = SensorModel(seed=seed)
        self.degradation = AutonomousDegradationSimulator(
            seed=seed,
            active_mode="natural",
            fault_rate_multiplier=fault_rate_multiplier,
        )

        self.history: List[Dict[str, float]] = []
        self.current_time_min: float = 0.0
        self.prev_state: Optional[EngineState] = None

    def reset(self):
        """Reset mission simulator, twin thermal states, degradation, and history."""
        self.twin.reset()
        self.sensor_model.reset()
        self.degradation.reset()
        self.history = []
        self.current_time_min = 0.0
        self.prev_state = None

    def _get_mission_profile_point(self, t_min: float) -> Tuple[float, float]:
        """
        Calculates altitude and throttle for the current mission time according to
        four distinct phases:
        0-5 min: Initialization / Initial Climb
        5-15 min: Climb / Transition to cruise altitude
        15-50 min: Sustained Cruise with realistic atmospheric/navigation variations
        50-60 min: Extended Cruise / Ingress Return / Descent
        """
        total = max(1.0, self.duration_minutes)
        rel = t_min / total

        # Smooth cyclic atmospheric and autopilot variations
        alt_osc = 180.0 * math.sin(t_min / 4.2) + 70.0 * math.cos(t_min / 8.5)
        thr_osc = 0.025 * math.sin(t_min / 3.5) + 0.012 * math.cos(t_min / 6.8)

        if t_min <= 5.0:
            # Phase 1: Ground rollout & initial climb
            progress = t_min / 5.0
            alt = 300.0 + progress * (self.target_altitude_m * 0.40 - 300.0)
            thr = 0.65 + 0.20 * progress
        elif t_min <= 15.0:
            # Phase 2: Climb transition to cruise altitude
            progress = (t_min - 5.0) / 10.0
            alt_start = self.target_altitude_m * 0.40
            alt = alt_start + progress * (self.target_altitude_m - alt_start)
            thr = 0.85 - 0.10 * progress
        elif t_min <= 50.0:
            # Phase 3: Cruise (near target altitude, nominal throttle)
            alt = self.target_altitude_m + alt_osc
            thr = self.base_throttle + thr_osc
        else:
            # Phase 4: Extended cruise / return / descent
            progress = (t_min - 50.0) / 10.0
            alt = (self.target_altitude_m + alt_osc) - progress * 800.0
            thr = self.base_throttle - 0.12 * progress + thr_osc

        # Enforce bounds
        alt = max(0.0, min(12000.0, alt))
        thr = max(0.20, min(1.00, thr))
        return alt, thr

    def step(self, dt_minutes: Optional[float] = None) -> Dict[str, float]:
        """
        Advance the simulation by dt_minutes (default 0.05 min).
        Simulates: Profile -> Twin -> Degradation -> Sensors -> History record.
        """
        dt = dt_minutes if dt_minutes is not None else self.timestep_min
        altitude, throttle = self._get_mission_profile_point(self.current_time_min)

        # 1. Preliminary estimate to determine operating stress on engine
        temp_state = self.twin.simulate(
            altitude_m=altitude,
            throttle=throttle,
            mission_time_min=self.current_time_min,
            dt_min=dt,
            prev_state=self.prev_state,
        )

        # 2. Autonomous stress-induced degradation updates hidden health variables
        deg_state = self.degradation.step(
            throttle=throttle,
            rpm=temp_state.rpm,
            ambient_temp_c=temp_state.ambient_temp_c,
            altitude_m=altitude,
            dt_minutes=dt,
        )

        # 3. Ground-truth Digital Twin evaluation reflecting updated degradation
        true_state = self.twin.simulate(
            altitude_m=altitude,
            throttle=throttle,
            cooling_health=deg_state.cooling_health,
            fuel_health=deg_state.fuel_health,
            oil_health=deg_state.oil_health,
            bearing_health=deg_state.bearing_health,
            sensor_health=deg_state.sensor_health,
            misfire_severity=deg_state.misfire_severity,
            mission_time_min=self.current_time_min,
            dt_min=dt,
            prev_state=self.prev_state,
        )
        self.prev_state = true_state

        # 4. Sensor Model applies instrumentation noise, bias, and drift
        measured = self.sensor_model.measure_state(
            true_state,
            dt_minutes=dt,
            sensor_health=deg_state.sensor_health,
        )

        # 5. Build authoritative telemetry record
        record: Dict[str, float] = {
            "mission_time_min": round(self.current_time_min, 3),
            "altitude_m": measured["altitude_m"],
            "ambient_temp_c": measured["ambient_temp_c"],
            "ambient_pressure_kpa": measured["ambient_pressure_kpa"],
            "air_density_kg_m3": true_state.air_density_kg_m3,
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
        }

        self.history.append(record)
        self.current_time_min += dt

        return record

    def run(
        self,
        duration_minutes: Optional[float] = None,
        samples_per_minute: Optional[int] = None,
        save_csv: bool = True,
    ) -> List[Dict[str, float]]:
        """
        Execute full mission run. Default: 60 minutes @ 20 samples/min = 1,200 samples.
        Saves authoritative telemetry to data/mission_telemetry.csv.
        """
        if duration_minutes is not None:
            self.duration_minutes = duration_minutes
        if samples_per_minute is not None:
            self.samples_per_minute = samples_per_minute
            self.timestep_min = 1.0 / float(samples_per_minute)

        self.reset()
        total_steps = int(round(self.duration_minutes * self.samples_per_minute))

        for _ in range(total_steps):
            self.step()

        if save_csv:
            self.export_csv(MISSION_CSV_PATH)

        return self.history

    def export_csv(self, file_path: Path):
        """Export current mission history to CSV."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history)
        # Ensure exact required column order
        cols = [c for c in REQUIRED_TELEMETRY_COLUMNS if c in df.columns]
        df = df[cols]
        df.to_csv(file_path, index=False)

    def get_dataframe(self) -> pd.DataFrame:
        """Return history as pandas DataFrame."""
        return pd.DataFrame(self.history)

    def latest(self) -> Optional[Dict[str, float]]:
        """Return most recent telemetry sample."""
        if not self.history:
            return None
        return self.history[-1]