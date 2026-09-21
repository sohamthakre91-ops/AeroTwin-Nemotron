"""
AeroTwin-Nemotron: Centralized Engine & Mission Configuration
Models a turbocharged, fuel-injected, 4-cylinder aero piston engine
representative of the Rotax 915 iS class.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Optional

CONFIG_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "engine_config.json"


@dataclass
class EngineConfig:
    # Core Engine Parameters
    engine_name: str = "Rotax 915 iS-class"
    engine_type: str = "Turbocharged fuel-injected four-cylinder piston engine"
    cylinders: int = 4
    displacement_cc: float = 1352.0
    nominal_power_kw: float = 104.0
    nominal_power_hp: float = 140.0
    nominal_rpm: float = 5800.0
    idle_rpm: float = 1800.0
    maximum_rpm: float = 6000.0
    compression_ratio: float = 8.2
    fuel_type: str = "aviation gasoline / compatible fuel model"

    # Mission Operating Limits
    minimum_altitude_m: float = 0.0
    maximum_altitude_m: float = 12000.0
    default_altitude_m: float = 3000.0
    minimum_throttle: float = 0.20
    maximum_throttle: float = 1.00
    default_throttle: float = 0.75
    reference_ambient_temperature_c: float = 15.0
    ambient_temperature_min_c: float = -40.0
    ambient_temperature_max_c: float = 50.0

    # Mission Timings
    mission_duration_default_min: float = 60.0
    samples_per_minute: int = 20
    timestep_min: float = 0.05

    # Turbocharger Parameters
    target_map_kpa: float = 155.0
    maximum_map_kpa: float = 200.0
    turbo_efficiency: float = 0.75
    boost_response: float = 0.85
    ambient_pressure_ratio_critical: float = 0.55

    # Thermal Parameters
    base_cht_c: float = 140.0
    max_cht_c: float = 180.0
    base_egt_c: float = 720.0
    max_egt_c: float = 880.0
    base_oil_temp_c: float = 85.0
    max_oil_temp_c: float = 130.0

    # Lubrication System
    base_oil_pressure_kpa: float = 380.0
    min_oil_pressure_kpa: float = 150.0
    max_oil_pressure_kpa: float = 550.0

    # Vibration Characteristics
    base_vibration_g: float = 0.025
    warning_vibration_g: float = 0.065
    critical_vibration_g: float = 0.120

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "EngineConfig":
        path = config_path or CONFIG_FILE_PATH
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception:
                return cls()
        return cls()


# Default singleton instance
DEFAULT_ENGINE_CONFIG = EngineConfig.load()
