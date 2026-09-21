"""
Tests for ISA Atmosphere, AeroTwin Digital Twin physics, and EngineState.
"""

import unittest
from src.atmosphere import isa_atmosphere, AtmosphereState
from src.twin import AeroTwin, EngineState
from src.config import EngineConfig, DEFAULT_ENGINE_CONFIG


class TestAtmosphereAndTwin(unittest.TestCase):

    def test_isa_atmosphere_physics(self):
        """Verify altitude increases -> pressure and density decrease."""
        sl = isa_atmosphere(0.0)
        h3k = isa_atmosphere(3000.0)
        h6k = isa_atmosphere(6000.0)

        # Standard sea level checks
        self.assertAlmostEqual(sl.ambient_pressure_kpa, 101.33, delta=0.5)
        self.assertAlmostEqual(sl.ambient_temperature_c, 15.0, delta=0.5)
        self.assertAlmostEqual(sl.air_density_kg_m3, 1.225, delta=0.05)

        # Pressure monotonic decrease
        self.assertGreater(sl.ambient_pressure_kpa, h3k.ambient_pressure_kpa)
        self.assertGreater(h3k.ambient_pressure_kpa, h6k.ambient_pressure_kpa)

        # Density monotonic decrease
        self.assertGreater(sl.air_density_kg_m3, h3k.air_density_kg_m3)
        self.assertGreater(h3k.air_density_kg_m3, h6k.air_density_kg_m3)

        # Temperature decreases in troposphere
        self.assertGreater(sl.ambient_temperature_c, h3k.ambient_temperature_c)
        self.assertGreater(h3k.ambient_temperature_c, h6k.ambient_temperature_c)

    def test_engine_state_observability(self):
        """Verify EngineState has observable quantities and NO hidden health variables."""
        twin = AeroTwin()
        state = twin.simulate(altitude_m=3000.0, throttle=0.75)

        state_dict = state.to_dict()

        # Observable features must exist
        observable = [
            "altitude_m", "ambient_temp_c", "ambient_pressure_kpa", "air_density_kg_m3",
            "throttle", "rpm", "map_kpa", "engine_load", "power_kw", "fuel_flow_lph",
            "cht_c", "egt_c", "oil_pressure_kpa", "oil_temperature_c", "vibration_g",
            "mission_time_min"
        ]
        for key in observable:
            self.assertIn(key, state_dict)

        # Hidden variables MUST NOT be present
        hidden_vars = ["cooling_health", "fuel_health", "oil_health", "bearing_health", "sensor_health"]
        for var in hidden_vars:
            self.assertNotIn(var, state_dict)

    def test_turbocharged_map_and_bounds(self):
        """Verify MAP is physically bounded between 25 and 200 kPa."""
        twin = AeroTwin()
        # High altitude full throttle
        high_state = twin.simulate(altitude_m=6000.0, throttle=1.0)
        self.assertGreaterEqual(high_state.map_kpa, 25.0)
        self.assertLessEqual(high_state.map_kpa, 200.0)

        # Sea level idle
        idle_state = twin.simulate(altitude_m=0.0, throttle=0.20)
        self.assertGreaterEqual(idle_state.map_kpa, 25.0)
        self.assertLessEqual(idle_state.map_kpa, 200.0)

    def test_degradation_physical_consequences(self):
        """Verify degradation produces correct directional physical responses."""
        twin = AeroTwin()
        healthy = twin.simulate(altitude_m=3000.0, throttle=0.75)

        # 1. Cooling degradation -> higher CHT
        cooling_deg = twin.simulate(altitude_m=3000.0, throttle=0.75, cooling_health=0.50)
        self.assertGreater(cooling_deg.cht_c, healthy.cht_c)

        # 2. Oil degradation -> lower oil pressure
        oil_deg = twin.simulate(altitude_m=3000.0, throttle=0.75, oil_health=0.50)
        self.assertLess(oil_deg.oil_pressure_kpa, healthy.oil_pressure_kpa)
        self.assertGreater(oil_deg.oil_pressure_kpa, 0.0)  # Never negative

        # 3. Bearing degradation -> higher vibration
        bearing_deg = twin.simulate(altitude_m=3000.0, throttle=0.75, bearing_health=0.50)
        self.assertGreater(bearing_deg.vibration_g, healthy.vibration_g)

        # 4. Fuel restriction -> lower fuel flow and power
        fuel_deg = twin.simulate(altitude_m=3000.0, throttle=0.75, fuel_health=0.50)
        self.assertLess(fuel_deg.fuel_flow_lph, healthy.fuel_flow_lph)
        self.assertLess(fuel_deg.power_kw, healthy.power_kw)


if __name__ == "__main__":
    unittest.main()
