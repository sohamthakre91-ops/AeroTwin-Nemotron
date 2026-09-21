"""
AeroTwin-Nemotron: Machine Learning Fault Diagnostics Engine
Trains, evaluates, versions, and performs inference with a Random Forest Classifier
using scenario-grouped validation to prevent data leakage.
"""

from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "telemetry.csv"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

FEATURE_COLUMNS = [
    "altitude_m",
    "ambient_temp_c",
    "ambient_pressure_kpa",
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

TARGET_COLUMN = "fault"


def load_metadata() -> Dict[str, Any]:
    """Load model version registry from models/model_metadata.json."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"latest_version": None, "versions": {}}
    return {"latest_version": None, "versions": {}}


def save_metadata(metadata: Dict[str, Any]):
    """Save model version registry to models/model_metadata.json."""
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def get_latest_model_path() -> Optional[Path]:
    """Return path to latest active model artifact."""
    meta = load_metadata()
    latest_ver = meta.get("latest_version")
    if latest_ver:
        path = MODELS_DIR / f"{latest_ver}.joblib"
        if path.exists():
            return path

    # Fallback checks
    default_joblib = MODELS_DIR / "fault_classifier.joblib"
    if default_joblib.exists():
        return default_joblib

    # Check v1
    v1_joblib = MODELS_DIR / "fault_classifier_v1.joblib"
    if v1_joblib.exists():
        return v1_joblib

    return None


def train_fault_classifier(
    dataset_path: Optional[Path] = None,
    dataframe: Optional[pd.DataFrame] = None,
    progress_callback: Optional[callable] = None,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Train a Random Forest fault classifier with 7-stage progress reporting:
    1. Loading dataset
    2. Preparing features
    3. Grouping scenarios
    4. Creating train/test split
    5. Training classifier
    6. Evaluating model
    7. Saving model & version metadata
    """
    def notify(stage: int, message: str):
        if progress_callback:
            progress_callback(stage, message)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Loading dataset
    notify(1, "Loading dataset...")
    if dataframe is not None:
        df = dataframe.copy()
    else:
        path = dataset_path or DEFAULT_DATASET_PATH
        if not path.exists():
            raise FileNotFoundError(f"Training dataset not found: {path}")
        df = pd.read_csv(path)

    # Validate feature columns
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    # Stage 2: Preparing features
    notify(2, "Preparing features...")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Stage 3: Grouping scenarios
    notify(3, "Grouping scenarios to prevent train/test data leakage...")
    if "scenario_id" in df.columns:
        groups = df["scenario_id"]
    else:
        # Fallback: create scenario groups from contiguous 120-step blocks
        groups = pd.Series(range(len(df)), index=df.index) // 120

    # Stage 4: Creating train/test split (80% train / 20% test)
    notify(4, "Creating scenario-wise train/test split (80/20)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Verify zero scenario leakage
    train_scenarios = set(groups.iloc[train_idx].unique())
    test_scenarios = set(groups.iloc[test_idx].unique())
    assert len(train_scenarios.intersection(test_scenarios)) == 0, "Data leakage detected: scenarios overlap!"

    # Stage 5: Training classifier
    notify(5, "Training RandomForestClassifier (250 estimators, max depth 14)...")
    clf_config = {
        "n_estimators": 250,
        "max_depth": 14,
        "min_samples_leaf": 3,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**clf_config)
    model.fit(X_train, y_train)

    # Stage 6: Evaluating model
    notify(6, "Evaluating model on held-out test scenarios...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    classes = list(model.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
    report = classification_report(y_test, y_pred, labels=classes, target_names=classes, output_dict=True, zero_division=0)

    # Feature importances
    feature_importances = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: x[1], reverse=True)
    ]

    # Stage 7: Saving model & updating version history
    notify(7, "Saving model artifact and updating version registry...")
    metadata = load_metadata()
    existing_versions = metadata.get("versions", {})
    version_num = len(existing_versions) + 1
    version_tag = f"fault_classifier_v{version_num}"

    version_model_path = MODELS_DIR / f"{version_tag}.joblib"
    default_model_path = MODELS_DIR / "fault_classifier.joblib"

    artifact = {
        "model": model,
        "features": FEATURE_COLUMNS,
        "classes": classes,
        "version": version_tag,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    joblib.dump(artifact, version_model_path)
    joblib.dump(artifact, default_model_path)

    # Version metrics record
    eval_metrics = {
        "version": version_tag,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_size": int(len(df)),
        "training_size": int(len(X_train)),
        "testing_size": int(len(X_test)),
        "feature_count": len(FEATURE_COLUMNS),
        "class_count": len(classes),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "feature_list": FEATURE_COLUMNS,
        "class_list": classes,
        "confusion_matrix": cm,
        "feature_importance": feature_importances,
        "classification_report": report,
        "training_configuration": clf_config,
    }

    metadata["latest_version"] = version_tag
    if "versions" not in metadata:
        metadata["versions"] = {}
    metadata["versions"][version_tag] = eval_metrics
    save_metadata(metadata)

    return model, eval_metrics


def predict_fault(
    telemetry: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run real-time inference on observable telemetry dictionary.
    Returns:
        predicted_fault: class label
        confidence: probability [0.0 - 1.0]
        fault_probabilities: dict mapping fault -> probability
        ranked_hypotheses: sorted list of {fault, probability}
    """
    path = model_path or get_latest_model_path()
    if not path or not path.exists():
        return {
            "error": "No trained diagnostic model artifact available.",
            "predicted_fault": "unknown",
            "confidence": 0.0,
            "ranked_hypotheses": [],
        }

    try:
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            features = loaded.get("features", FEATURE_COLUMNS)
            classes = loaded.get("classes", list(model.classes_))
        else:
            model = loaded
            features = FEATURE_COLUMNS
            classes = list(model.classes_)

        # Build feature vector
        row = []
        for feat in features:
            val = telemetry.get(feat, 0.0)
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)

        X_in = pd.DataFrame([row], columns=features)
        pred_class = model.predict(X_in)[0]
        probs = model.predict_proba(X_in)[0]

        prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
        ranked = sorted(
            [{"fault": c, "probability": float(p)} for c, p in prob_map.items()],
            key=lambda x: x["probability"],
            reverse=True,
        )
        conf = prob_map.get(str(pred_class), 0.0)

        return {
            "predicted_fault": str(pred_class),
            "confidence": conf,
            "fault_probabilities": prob_map,
            "ranked_hypotheses": ranked,
            "model_version": getattr(model, "version", "RandomForestClassifier"),
        }

    except Exception as exc:
        return {
            "error": f"Diagnostic inference failed: {str(exc)}",
            "predicted_fault": "unknown",
            "confidence": 0.0,
            "ranked_hypotheses": [],
        }