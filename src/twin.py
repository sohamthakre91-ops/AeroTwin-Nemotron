"""
AeroTwin-Nemotron: Physics-Informed Digital Twin
Models a turbocharged, fuel-injected, four-cylinder aero piston engine
representative of the Rotax 915 iS class.
"""

from dataclasses import dataclass, asdict
import math
from typing import Optional, Dict, Any

from .config import EngineConfig, DEFAULT_ENGINE_CONFIG
from .atmosphere import isa_atmosphere, AtmosphereState


@dataclass
class EngineState:
    """
    Public observable telemetry state of the engine.
    Hidden health variables are NEVER exposed here.
    """
    altitude_m: float
    ambient_temp_c: float
    ambient_pressure_kpa: float
    air_density_kg_m3: float

    throttle: float
    rpm: float
    map_kpa: float
    engine_load: float

    power_kw: float
    fuel_flow_lph: float

    cht_c: float
    egt_c: float

    oil_pressure_kpa: float
    oil_temperature_c: float

    vibration_g: float

    mission_time_min: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class AeroTwin:
    """
    Physics-informed Digital Twin for a Rotax 915 iS-class UAV propulsion system.
    Maintains internal thermal states and dynamic responses.
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or DEFAULT_ENGINE_CONFIG
        # Internal thermal states for continuous first-order inertia
        self._cht_internal: float = self.config.base_cht_c
        self._oil_temp_internal: float = self.config.base_oil_temp_c

    def reset(self):
        """Reset internal thermal history to baseline defaults."""
        self._cht_internal = self.config.base_cht_c
        self._oil_temp_internal = self.config.base_oil_temp_c

    def simulate(
        self,
        altitude_m: float = 3000.0,
        throttle: float = 0.75,
        ambient_temp_offset_c: float = 0.0,
        cooling_health: float = 1.0,
        fuel_health: float = 1.0,
        oil_health: float = 1.0,
        bearing_health: float = 1.0,
        sensor_health: float = 1.0,
        misfire_severity: float = 0.0,
        mission_time_min: float = 0.0,
        dt_min: float = 0.05,
        prev_state: Optional[EngineState] = None,
    ) -> EngineState:
        """
        Simulate engine operation at a specific operating point and health condition.

        Parameters:
            altitude_m: Flight altitude in meters [0, 12000].
            throttle: Commanded throttle position [0.20, 1.00].
            ambient_temp_offset_c: Deviation from ISA standard temperature.
            cooling_health: Internal cooling health [0.0 - 1.0].
            fuel_health: Internal fuel system health [0.0 - 1.0].
            oil_health: Internal lubrication health [0.0 - 1.0].
            bearing_health: Internal mechanical bearing health [0.0 - 1.0].
            sensor_health: Sensor measurement integrity [0.0 - 1.0].
            misfire_severity: Internal combustion misfire severity [0.0 - 1.0].
            mission_time_min: Elapsed mission time in minutes.
            dt_min: Timestep in minutes for dynamic integration.
            prev_state: Optional previous engine state for smooth thermal inertia.
        """
        cfg = self.config

        # ------------------------------------------------------------------
        # 1. Bounded Input Clamping
        # ------------------------------------------------------------------
        altitude_m = max(cfg.minimum_altitude_m, min(cfg.maximum_altitude_m, float(altitude_m)))
        throttle = max(cfg.minimum_throttle, min(cfg.maximum_throttle, float(throttle)))

        cooling_health = max(0.0, min(1.0, float(cooling_health)))
        fuel_health = max(0.0, min(1.0, float(fuel_health)))
        oil_health = max(0.0, min(1.0, float(oil_health)))
        bearing_health = max(0.0, min(1.0, float(bearing_health)))
        sensor_health = max(0.0, min(1.0, float(sensor_health)))
        misfire_severity = max(0.0, min(1.0, float(misfire_severity)))

        # ------------------------------------------------------------------
        # 2. Atmospheric Calculations (ISA)
        # ------------------------------------------------------------------
        atmos: AtmosphereState = isa_atmosphere(altitude_m, ambient_temp_offset_c)
        ambient_temp_c = atmos.ambient_temperature_c
        ambient_pressure_kpa = atmos.ambient_pressure_kpa
        air_density_kg_m3 = atmos.air_density_kg_m3

        # Standard sea level reference values
        P0_KPA = 101.325
        RHO0 = 1.225
        density_ratio = air_density_kg_m3 / RHO0
        ambient_pressure_ratio = ambient_pressure_kpa / P0_KPA

        # ------------------------------------------------------------------
        # 3. Turbocharger Model & Manifold Absolute Pressure (MAP)
        # ------------------------------------------------------------------
        # Turbo compensates for altitude pressure loss up to critical pressure ratio
        # Turbo boost capability increases as ambient pressure decreases until wastegate limit
        boost_demand = (P0_KPA - ambient_pressure_kpa) * cfg.turbo_efficiency * cfg.boost_response
        max_boost_allowed = cfg.maximum_map_kpa - ambient_pressure_kpa
        boost_kpa = max(0.0, min(max_boost_allowed, boost_demand))

        # Full throttle target MAP is ~155 kPa at sea level, maintained up to ~4500m
        base_map = ambient_pressure_kpa + boost_kpa
        map_effective = (
            ambient_pressure_kpa * (0.35 + 0.65 * throttle)
            + boost_kpa * (0.20 + 0.80 * throttle)
        )

        # Enforce realistic manifold pressure boundaries [25 kPa, 200 kPa]
        map_kpa = max(25.0, min(cfg.maximum_map_kpa, map_effective))

        # ------------------------------------------------------------------
        # 4. RPM Dynamics
        # ------------------------------------------------------------------
        # Rotax 915 iS has constant speed propeller governor logic,
        # but RPM responds dynamically to throttle setting, MAP, and mechanical drag
        rpm_range = cfg.maximum_rpm - cfg.idle_rpm
        governed_rpm = cfg.idle_rpm + (throttle ** 0.85) * rpm_range

        # Degradation drag and misfire effects on RPM
        rpm_penalty = (1.0 - fuel_health) * 120.0 + (1.0 - bearing_health) * 150.0 + misfire_severity * 280.0
        # Correlated misfire oscillation
        misfire_rpm_jitter = 0.0
        if misfire_severity > 0.01:
            misfire_rpm_jitter = misfire_severity * 65.0 * math.sin(mission_time_min * 24.0)

        rpm = governed_rpm - rpm_penalty + misfire_rpm_jitter
        rpm = max(cfg.idle_rpm, min(cfg.maximum_rpm, rpm))

        # ------------------------------------------------------------------
        # 5. Engine Load & Power Delivery
        # ------------------------------------------------------------------
        # Load reflects throttle, available manifold pressure, and fuel system health
        engine_load = (0.20 + 0.80 * throttle) * (map_kpa / cfg.target_map_kpa)
        engine_load *= (0.75 + 0.25 * fuel_health)
        engine_load = max(0.15, min(1.0, engine_load))

        # Power output (kW): nominal 104 kW (140 hp) at full power
        # Derived from displacement, RPM, MAP, air density, and combustion efficiency
        combustion_eff = (0.85 + 0.15 * fuel_health) * (1.0 - 0.35 * misfire_severity)
        power_kw = (
            cfg.nominal_power_kw
            * (rpm / cfg.nominal_rpm)
            * (map_kpa / cfg.target_map_kpa)
            * (0.85 + 0.15 * density_ratio)
            * combustion_eff
        )
        power_kw = max(5.0, min(120.0, power_kw))

        # ------------------------------------------------------------------
        # 6. Fuel Flow (L/h)
        # ------------------------------------------------------------------
        # Specific Fuel Consumption (BSFC) ~ 0.28 L/kWh at nominal operating points
        bsfc = 0.285 + 0.04 * (1.0 - throttle)  # Slightly higher BSFC at low throttle
        ideal_fuel_flow = 3.5 + power_kw * bsfc
        # Fuel health directly constrains delivered fuel flow
        fuel_flow_lph = ideal_fuel_flow * (0.60 + 0.40 * fuel_health)
        if misfire_severity > 0.05:
            # Unburnt fuel loss / erratic delivery
            fuel_flow_lph += misfire_severity * 1.5 * math.sin(mission_time_min * 18.0)
        fuel_flow_lph = max(2.5, min(48.0, fuel_flow_lph))

        # ------------------------------------------------------------------
        # 7. Thermal Dynamics: CHT & EGT with First-Order Inertia
        # ------------------------------------------------------------------
        # Target steady-state CHT
        cht_target = (
            cfg.base_cht_c
            + 35.0 * (engine_load ** 1.2)
            + 0.5 * (ambient_temp_c - cfg.reference_ambient_temperature_c)
            + 55.0 * (1.0 - cooling_health)
            + (altitude_m / 6000.0) * 8.0  # Thinner air reduces radiator cooling
        )

        # Thermal inertia integration for CHT (time constant tau ~ 2.5 min)
        tau_cht = 2.5
        alpha_cht = 1.0 - math.exp(-max(0.001, dt_min) / tau_cht)
        if prev_state is not None:
            self._cht_internal = prev_state.cht_c + alpha_cht * (cht_target - prev_state.cht_c)
        else:
            self._cht_internal = self._cht_internal + alpha_cht * (cht_target - self._cht_internal)
        cht_c = max(70.0, min(cfg.max_cht_c + 25.0, self._cht_internal))

        # Exhaust Gas Temperature (EGT)
        # Higher load increases EGT; lean mixture (fuel restriction) elevates EGT up to peak;
        # Severe misfire creates erratic drops and unburnt pulses
        lean_egt_boost = 60.0 * (1.0 - fuel_health) if fuel_health > 0.5 else 20.0
        cooling_egt_effect = 25.0 * (1.0 - cooling_health)
        misfire_egt_dip = -90.0 * misfire_severity + (misfire_severity * 30.0 * math.cos(mission_time_min * 20.0))

        egt_c = (
            cfg.base_egt_c
            + 110.0 * engine_load
            + 0.3 * (ambient_temp_c - cfg.reference_ambient_temperature_c)
            + lean_egt_boost
            + cooling_egt_effect
            + misfire_egt_dip
        )
        egt_c = max(500.0, min(cfg.max_egt_c + 30.0, egt_c))

        # ------------------------------------------------------------------
        # 8. Lubrication System: Oil Pressure & Oil Temperature
        # ------------------------------------------------------------------
        # Target steady-state Oil Temperature (tau ~ 3.5 min)
        oil_temp_target = (
            cfg.base_oil_temp_c
            + 28.0 * engine_load
            + 0.4 * (ambient_temp_c - cfg.reference_ambient_temperature_c)
            + 30.0 * (1.0 - cooling_health)
            + 22.0 * (1.0 - oil_health)
        )
        tau_oil = 3.5
        alpha_oil = 1.0 - math.exp(-max(0.001, dt_min) / tau_oil)
        if prev_state is not None:
            self._oil_temp_internal = prev_state.oil_temperature_c + alpha_oil * (oil_temp_target - prev_state.oil_temperature_c)
        else:
            self._oil_temp_internal = self._oil_temp_internal + alpha_oil * (oil_temp_target - self._oil_temp_internal)
        oil_temperature_c = max(40.0, min(cfg.max_oil_temp_c + 15.0, self._oil_temp_internal))

        # Oil Pressure: driven by positive displacement pump (proportional to RPM)
        # Reduced by high oil temperature (thinned oil viscosity) and oil_health degradation
        temp_viscosity_factor = max(0.70, 1.0 - 0.004 * (oil_temperature_c - cfg.base_oil_temp_c))
        oil_pressure_kpa = (
            cfg.base_oil_pressure_kpa
            * (rpm / cfg.nominal_rpm) ** 0.8
            * (0.35 + 0.65 * oil_health)
            * temp_viscosity_factor
        )
        # Strictly enforce non-negative pressure
        oil_pressure_kpa = max(10.0, min(cfg.max_oil_pressure_kpa, oil_pressure_kpa))

        # ------------------------------------------------------------------
        # 9. Vibration Dynamics
        # ------------------------------------------------------------------
        # Baseline vibration scales with RPM and load
        vibe_rpm = cfg.base_vibration_g * (rpm / cfg.nominal_rpm)
        vibe_load = 0.012 * engine_load
        # Bearing degradation produces pronounced and persistent harmonic vibration
        vibe_bearing = 0.110 * (1.0 - bearing_health) ** 1.3
        # Misfire creates cyclic combustion pulse imbalance
        vibe_misfire = misfire_severity * (0.055 + 0.025 * abs(math.sin(mission_time_min * 30.0)))

        vibration_g = vibe_rpm + vibe_load + vibe_bearing + vibe_misfire
        vibration_g = max(0.010, min(0.350, vibration_g))

        return EngineState(
            altitude_m=round(altitude_m, 1),
            ambient_temp_c=round(ambient_temp_c, 2),
            ambient_pressure_kpa=round(ambient_pressure_kpa, 2),
            air_density_kg_m3=round(air_density_kg_m3, 4),
            throttle=round(throttle, 3),
            rpm=round(rpm, 1),
            map_kpa=round(map_kpa, 2),
            engine_load=round(engine_load, 3),
            power_kw=round(power_kw, 2),
            fuel_flow_lph=round(fuel_flow_lph, 2),
            cht_c=round(cht_c, 2),
            egt_c=round(egt_c, 2),
            oil_pressure_kpa=round(oil_pressure_kpa, 2),
            oil_temperature_c=round(oil_temperature_c, 2),
            vibration_g=round(vibration_g, 4),
            mission_time_min=round(mission_time_min, 2),
        )