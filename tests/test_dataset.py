"""
Tests for Synthetic Dataset Generation: 7 fault classes, 14 observable features,
scenario grouping, and zero leakage of hidden variables.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.dataset import (
    FAULT_CLASSES,
    FEATURE_COLUMNS,
    generate_scenario_data,
    TELEMETRY_CSV_PATH,
)


class TestSyntheticDataset(unittest.TestCase):

    def test_single_scenario_generation(self):
        """Verify generation of a single cohesive 120-step scenario."""
        df = generate_scenario_data(
            scenario_id=1,
            fault_name="cooling_degradation",
            steps=120,
            seed=42,
        )
        self.assertEqual(len(df), 120)
        self.assertEqual(df["scenario_id"].iloc[0], 1)
        self.assertEqual(df["fault"].iloc[0], "cooling_degradation")

        # Verify 14 observable features are present
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, df.columns)

        # Verify hidden health variables are NOT present
        for hidden in ["cooling_health", "fuel_health", "oil_health", "bearing_health", "sensor_health"]:
            self.assertNotIn(hidden, df.columns)

    def test_dataset_file_integrity(self):
        """Verify that data/telemetry.csv exists, contains >= 42,000 rows, and 7 classes."""
        if not TELEMETRY_CSV_PATH.exists():
            self.skipTest("telemetry.csv not yet generated in data/")

        df = pd.read_csv(TELEMETRY_CSV_PATH)
        self.assertGreaterEqual(len(df), 42000)

        # 7 distinct classes
        classes = df["fault"].unique()
        self.assertEqual(len(classes), 7)
        for c in FAULT_CLASSES:
            self.assertIn(c, classes)

        # 14 observable features present
        for feat in FEATURE_COLUMNS:
            self.assertIn(feat, df.columns)

        # Scenario IDs present and distinct
        self.assertIn("scenario_id", df.columns)
        self.assertGreaterEqual(df["scenario_id"].nunique(), 350)


if __name__ == "__main__":
    unittest.main()
