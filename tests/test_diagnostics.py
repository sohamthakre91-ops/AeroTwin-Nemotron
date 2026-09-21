"""
Tests for ML Diagnostics: GroupShuffleSplit scenario grouping, held-out evaluation,
confusion matrix, feature importances, versioning, and inference.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.diagnostics import (
    train_fault_classifier,
    predict_fault,
    load_metadata,
    FEATURE_COLUMNS,
    METADATA_PATH,
)
from src.dataset import generate_scenario_data, FAULT_CLASSES


class TestDiagnostics(unittest.TestCase):

    def test_model_metadata_registry(self):
        """Verify model_metadata.json contains valid trained model versions and metrics."""
        metadata = load_metadata()
        self.assertIn("latest_version", metadata)
        self.assertIn("versions", metadata)

        latest_ver = metadata["latest_version"]
        self.assertIsNotNone(latest_ver)
        self.assertIn(latest_ver, metadata["versions"])

        latest_metrics = metadata["versions"][latest_ver]
        self.assertIn("accuracy", latest_metrics)
        self.assertIn("macro_f1", latest_metrics)
        self.assertIn("weighted_f1", latest_metrics)
        self.assertIn("confusion_matrix", latest_metrics)
        self.assertIn("feature_importance", latest_metrics)

        # Confirm 14 features and 7 classes
        self.assertEqual(latest_metrics["feature_count"], 14)
        self.assertEqual(latest_metrics["class_count"], 7)
        self.assertGreater(latest_metrics["accuracy"], 0.70)

    def test_inference_prediction(self):
        """Verify predict_fault returns valid hypothesis distribution and probabilities."""
        sample_telemetry = {
            "altitude_m": 3000.0,
            "ambient_temp_c": -4.5,
            "ambient_pressure_kpa": 70.1,
            "throttle": 0.75,
            "rpm": 5420.0,
            "map_kpa": 142.0,
            "engine_load": 0.78,
            "power_kw": 84.5,
            "fuel_flow_lph": 26.2,
            "cht_c": 172.0,  # High CHT indicates cooling degradation
            "egt_c": 745.0,
            "oil_pressure_kpa": 360.0,
            "oil_temperature_c": 98.0,
            "vibration_g": 0.028,
        }

        diag = predict_fault(sample_telemetry)

        self.assertIn("predicted_fault", diag)
        self.assertIn("confidence", diag)
        self.assertIn("ranked_hypotheses", diag)
        self.assertIn("fault_probabilities", diag)

        self.assertIsInstance(diag["confidence"], float)
        self.assertGreaterEqual(diag["confidence"], 0.0)
        self.assertLessEqual(diag["confidence"], 1.0)
        self.assertEqual(len(diag["ranked_hypotheses"]), 7)

    def test_train_pipeline_execution(self):
        """Verify train_fault_classifier runs with progress callback and produces all metrics."""
        # Create a small balanced 14-scenario dataset (2 per class, 30 steps each)
        dfs = []
        scen_id = 9000
        for fc in FAULT_CLASSES:
            for _ in range(2):
                df_s = generate_scenario_data(scen_id, fc, steps=30, seed=scen_id)
                dfs.append(df_s)
                scen_id += 1
        df_small = pd.concat(dfs, ignore_index=True)

        # Backup original metadata
        original_meta = METADATA_PATH.read_text(encoding="utf-8") if METADATA_PATH.exists() else None

        stages_visited = []
        def on_progress(stage, msg):
            stages_visited.append(stage)

        metrics = None
        try:
            model, metrics = train_fault_classifier(
                dataframe=df_small,
                progress_callback=on_progress,
            )

            self.assertIsNotNone(model)
            self.assertEqual(len(stages_visited), 7)
            self.assertIn("accuracy", metrics)
            self.assertIn("macro_f1", metrics)
            self.assertIn("weighted_f1", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertIn("feature_importance", metrics)
        finally:
            # Restore pristine metadata
            if original_meta is not None:
                METADATA_PATH.write_text(original_meta, encoding="utf-8")
            # Clean up ephemeral test model joblib
            if metrics and "version" in metrics:
                ephemeral_joblib = Path(__file__).resolve().parent.parent / "models" / f"{metrics['version']}.joblib"
                if ephemeral_joblib.exists():
                    ephemeral_joblib.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
