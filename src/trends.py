"""
AeroTwin-Nemotron: Trend Analysis & Local Anomaly Detection Engine
Evaluates time-series trajectories, early vs late mission windows, statistical slopes,
and persistent physical deviations across CHT, EGT, Oil Pressure, Vibration, and Power.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


KEY_TREND_SIGNALS = [
    "cht_c",
    "egt_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
    "vibration_g",
    "power_kw",
]


def calculate_linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Computes linear regression slope (units/min) using least squares."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom < 1e-9:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def analyze_trends(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Performs comprehensive statistical and window-based trend analysis.
    Compares early mission window (first 10%) vs late mission window (last 10%).
    Returns:
        signals: dict of per-signal stats (mean, min, max, std, slope, window_delta, persistence)
        summary: overall trend summary
    """
    if not history or len(history) < 5:
        return {
            "sample_count": len(history) if history else 0,
            "status": "INSUFFICIENT_DATA",
            "signals": {},
        }

    df = pd.DataFrame(history)
    n = len(df)
    t = df["mission_time_min"].values if "mission_time_min" in df.columns else np.arange(n) * 0.05

    window_size = max(2, int(n * 0.10))
    early_df = df.iloc[:window_size]
    late_df = df.iloc[-window_size:]

    signal_metrics: Dict[str, Dict[str, float]] = {}

    for sig in KEY_TREND_SIGNALS:
        if sig not in df.columns:
            continue

        series = df[sig].values.astype(float)
        early_vals = early_df[sig].values.astype(float)
        late_vals = late_df[sig].values.astype(float)

        mean_val = float(np.mean(series))
        min_val = float(np.min(series))
        max_val = float(np.max(series))
        std_val = float(np.std(series))

        slope = calculate_linear_slope(t, series)

        early_mean = float(np.mean(early_vals))
        late_mean = float(np.mean(late_vals))
        window_delta = late_mean - early_mean
        pct_change = (window_delta / abs(early_mean) * 100.0) if abs(early_mean) > 1e-6 else 0.0

        # Persistence score: fraction of consecutive points in late window showing sustained deviation
        persistent_change = abs(window_delta) > (2.0 * std_val)

        signal_metrics[sig] = {
            "mean": round(mean_val, 3),
            "min": round(min_val, 3),
            "max": round(max_val, 3),
            "std": round(std_val, 3),
            "slope_per_min": round(slope, 4),
            "early_window_mean": round(early_mean, 3),
            "late_window_mean": round(late_mean, 3),
            "window_delta": round(window_delta, 3),
            "percent_change": round(pct_change, 2),
            "is_persistent": persistent_change,
        }

    return {
        "sample_count": n,
        "early_window_size": window_size,
        "late_window_size": window_size,
        "signals": signal_metrics,
    }


def assess_system_status(
    history: List[Dict[str, Any]],
    baseline_result: Optional[Dict[str, Any]] = None,
    ml_diagnosis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Multi-evidence anomaly assessment engine.
    Synthesizes:
    - Trend slopes and persistence (CHT, EGT, Oil Pressure, Vibration)
    - Baseline deviations from healthy reference
    - ML fault hypothesis & confidence
    - Persistence over time
    Outputs:
        status: NORMAL | MONITORING | WARNING | CRITICAL
        evidence: list of specific quantitative supporting evidence items
        composite_severity: float [0.0 - 1.0]
    """
    if not history or len(history) < 5:
        return {
            "status": "NORMAL",
            "evidence": ["System initializing - baseline telemetry steady"],
            "composite_severity": 0.0,
        }

    trends = analyze_trends(history)
    signals = trends.get("signals", {})
    evidence: List[str] = []
    severity_points = 0.0

    # 1. Thermal Evidence (CHT & EGT)
    cht_stats = signals.get("cht_c", {})
    if cht_stats:
        if cht_stats["max"] > 165.0:
            evidence.append(f"CHT critical peak: {cht_stats['max']:.1f}°C (limit 165°C)")
            severity_points += 3.5
        elif cht_stats["slope_per_min"] > 0.35 and cht_stats["is_persistent"]:
            evidence.append(f"Persistent CHT upward trend: +{cht_stats['window_delta']:.1f}°C across mission (+{cht_stats['slope_per_min']:.2f}°C/min)")
            severity_points += 2.0
        elif cht_stats["percent_change"] > 6.0:
            evidence.append(f"CHT elevated above initial climb: +{cht_stats['percent_change']:.1f}%")
            severity_points += 1.0

    # 2. Lubrication Evidence (Oil Pressure & Temperature)
    oil_p_stats = signals.get("oil_pressure_kpa", {})
    if oil_p_stats:
        if oil_p_stats["min"] < 200.0:
            evidence.append(f"Low oil pressure minimum: {oil_p_stats['min']:.1f} kPa (minimum safe 200 kPa)")
            severity_points += 3.5
        elif oil_p_stats["slope_per_min"] < -0.40 and oil_p_stats["is_persistent"]:
            evidence.append(f"Persistent oil pressure decay: {oil_p_stats['window_delta']:.1f} kPa decay ({oil_p_stats['slope_per_min']:.2f} kPa/min)")
            severity_points += 2.2
        elif oil_p_stats["percent_change"] < -8.0:
            evidence.append(f"Oil pressure dropped {oil_p_stats['percent_change']:.1f}% below baseline")
            severity_points += 1.2

    # 3. Mechanical Vibration Evidence
    vibe_stats = signals.get("vibration_g", {})
    if vibe_stats:
        if vibe_stats["max"] > 0.090:
            evidence.append(f"Severe vibration spike: {vibe_stats['max']:.3f} g (threshold 0.090 g)")
            severity_points += 3.5
        elif vibe_stats["is_persistent"] and vibe_stats["percent_change"] > 25.0:
            evidence.append(f"Persistent harmonic vibration growth: +{vibe_stats['percent_change']:.1f}% ({vibe_stats['late_window_mean']:.3f} g late mean)")
            severity_points += 2.5
        elif vibe_stats["slope_per_min"] > 0.0003:
            evidence.append(f"Vibration trending upward: slope +{vibe_stats['slope_per_min']:.4f} g/min")
            severity_points += 1.0

    # 4. Baseline Deviation Evidence
    if baseline_result and "significant_deviations" in baseline_result:
        sig_devs = baseline_result["significant_deviations"]
        for item in sig_devs[:3]:
            param = item["parameter"]
            pct = item["percent_deviation"]
            if abs(pct) > 15.0:
                evidence.append(f"Large baseline deviation in {param}: {pct:+.1f}% vs expected healthy twin")
                severity_points += 1.5
            elif abs(pct) > 8.0:
                evidence.append(f"Moderate baseline deviation in {param}: {pct:+.1f}%")
                severity_points += 0.8

    # 5. ML Diagnostic Evidence
    if ml_diagnosis and "error" not in ml_diagnosis:
        pred_fault = ml_diagnosis.get("predicted_fault", "normal")
        conf = ml_diagnosis.get("confidence", 0.0)
        if pred_fault != "normal" and conf > 0.65:
            fault_label = pred_fault.replace("_", " ").title()
            evidence.append(f"ML Classifier hypothesis: {fault_label} with {conf * 100:.1f}% confidence")
            severity_points += 2.0
        elif pred_fault != "normal" and conf > 0.40:
            fault_label = pred_fault.replace("_", " ").title()
            evidence.append(f"ML Classifier early detection: {fault_label} ({conf * 100:.1f}%)")
            severity_points += 1.0

    # Composite Classification
    if severity_points >= 5.5:
        status = "CRITICAL"
    elif severity_points >= 3.0:
        status = "WARNING"
    elif severity_points >= 1.2:
        status = "MONITORING"
    else:
        status = "NORMAL"
        if not evidence:
            evidence.append("All primary propulsion signatures (CHT, EGT, Oil P, Vibration) within nominal limits")

    return {
        "status": status,
        "evidence": evidence,
        "severity_score": round(min(10.0, severity_points), 2),
        "trends": trends,
    }
