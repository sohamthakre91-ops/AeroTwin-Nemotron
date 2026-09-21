"""
Tests for local engineering tool interface designed for future Nemotron investigation.
"""

import unittest
from src.tools import (
    get_engine_state,
    get_sensor_history,
    run_fault_detection,
    compare_with_baseline,
    assess_mission_risk,
    run_what_if_scenario,
    compare_scenarios,
)


class TestLocalTools(unittest.TestCase):

    def test_get_engine_state(self):
        """Verify get_engine_state returns clean observable telemetry dictionary."""
        state = get_engine_state(altitude_m=3000.0, throttle=0.75)
        self.assertIsInstance(state, dict)
        self.assertIn("cht_c", state)
        self.assertIn("rpm", state)
        self.assertIn("power_kw", state)
        self.assertNotIn("cooling_health", state)

    def test_compare_with_baseline(self):
        """Verify baseline comparison computes deltas and percentage deviations."""
        state = get_engine_state(altitude_m=3000.0, throttle=0.75, cooling_health=0.6)
        comp = compare_with_baseline(state)

        self.assertIn("baseline", comp)
        self.assertIn("deviations", comp)
        self.assertIn("significant_deviations", comp)

        # CHT deviation must be positive and significant
        cht_dev = comp["deviations"]["cht_c"]
        self.assertGreater(cht_dev["delta"], 0.0)
        self.assertGreater(cht_dev["percent_deviation"], 0.0)

    def test_run_what_if_scenario(self):
        """Verify counterfactual simulation generates comparative prediction."""
        res = run_what_if_scenario(
            altitude_m=3000.0,
            throttle=0.75,
            cooling_health=0.70,
            fuel_health=1.0,
            oil_health=1.0,
            bearing_health=1.0,
            sensor_health=1.0,
            target_telemetry={"cht_c": 155.0, "egt_c": 750.0},
        )
        self.assertIn("hypothetical_telemetry", res)
        self.assertIn("comparison", res)
        self.assertIn("similarity_score_percent", res["comparison"])

    def test_assess_mission_risk(self):
        """Verify mission assurance risk scoring and categorisation."""
        healthy_state = get_engine_state(altitude_m=3000.0, throttle=0.75)
        risk = assess_mission_risk(healthy_state, mission_duration_min=60.0)

        self.assertIn("risk_score", risk)
        self.assertIn("risk_level", risk)
        self.assertIn("mission_impact", risk)
        self.assertIn(risk["risk_level"], ["LOW", "MODERATE", "HIGH", "CRITICAL"])
        self.assertGreaterEqual(risk["risk_score"], 0.0)
        self.assertLessEqual(risk["risk_score"], 100.0)

    def test_compare_scenarios(self):
        """Verify scenario comparison calculates diffs accurately."""
        s1 = {"rpm": 5400.0, "power_kw": 85.0}
        s2 = {"rpm": 5600.0, "power_kw": 92.0}
        diff = compare_scenarios(s1, s2)

        self.assertIn("differences", diff)
        self.assertEqual(diff["differences"]["rpm"]["delta"], 200.0)


if __name__ == "__main__":
    unittest.main()
