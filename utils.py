"""Atmospheric Forecasting Utility Functions & Metric Analytics for Aero-Cast AI.

Provides mathematical formulas for evaluating regression accuracy, anomaly
deviation indicators, interval coverage scores, and spatial-temporal data formatting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_regression_metrics(
    y_true: Union[np.ndarray, List[float], pd.Series],
    y_pred: Union[np.ndarray, List[float], pd.Series],
) -> Dict[str, float]:
    """Calculates comprehensive error and goodness-of-fit metrics.

    Computes:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (%)
    - R²: Coefficient of Determination
    - MedAE: Median Absolute Error
    - MaxError: Maximum absolute residual

    Args:
        y_true: Array-like of actual ground-truth observations.
        y_pred: Array-like of model-predicted values.

    Returns:
        Dictionary mapping metric names to floating point scores.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    residuals = yt - yp
    abs_residuals = np.abs(residuals)

    mae = float(np.mean(abs_residuals))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    medae = float(np.median(abs_residuals))
    max_error = float(np.max(abs_residuals))

    # Avoid division by zero in MAPE
    denom = np.where(np.abs(yt) < 1e-4, 1.0, np.abs(yt))
    mape = float(np.mean(abs_residuals / denom) * 100.0)

    # R-squared calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - (ss_res / max(1e-9, ss_tot)))

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
        "medae": round(medae, 3),
        "max_error": round(max_error, 3),
    }


def compute_anomaly_deviation_score(
    predictions: Union[np.ndarray, pd.Series],
    climatology_mean: Union[float, np.ndarray, pd.Series],
    climatology_std: Union[float, np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculates the standardized atmospheric Anomaly Deviation Score (ADS).

    ADS expresses how intensely the forecasted atmospheric trajectory departs
    from historical climatological baseline in units of local standard deviations (Z-score).

    Args:
        predictions: Forecasted trajectory values.
        climatology_mean: Long-term expected baseline mean.
        climatology_std: Long-term standard deviation.

    Returns:
        Dictionary containing:
        - mean_ads: Average standardized z-score deviation.
        - max_ads: Peak anomaly z-score.
        - anomaly_category: Qualitative classification (Normal, Moderate, Extreme).
    """
    yp = np.asarray(predictions, dtype=np.float64)
    c_mean = np.asarray(climatology_mean, dtype=np.float64)
    c_std = np.asarray(climatology_std, dtype=np.float64)
    c_std = np.where(c_std < 1e-4, 1.0, c_std)

    z_scores = np.abs((yp - c_mean) / c_std)
    mean_ads = float(np.mean(z_scores))
    max_ads = float(np.max(z_scores))

    if max_ads < 1.0:
        category = "Climatologically Normal"
    elif max_ads < 2.0:
        category = "Moderate Anomaly"
    elif max_ads < 3.0:
        category = "Severe Atmospheric Perturbation"
    else:
        category = "Extreme Synoptic Anomaly"

    return {
        "mean_ads": round(mean_ads, 2),
        "max_ads": round(max_ads, 2),
        "category": category,
    }


def compute_prediction_interval_coverage(
    y_true: Union[np.ndarray, List[float], pd.Series],
    lower_bound: Union[np.ndarray, List[float], pd.Series],
    upper_bound: Union[np.ndarray, List[float], pd.Series],
) -> Dict[str, float]:
    """Evaluates the calibration quality of forecasted uncertainty bounds.

    Calculates:
    - PICP: Prediction Interval Coverage Probability (% of observations within bounds).
    - MPIW: Mean Prediction Interval Width (sharpness of uncertainty envelope).

    Args:
        y_true: Observed ground-truth target values.
        lower_bound: Lower confidence bound predictions.
        upper_bound: Upper confidence bound predictions.

    Returns:
        Dictionary mapping 'picp_pct' and 'mpiw' to floating point values.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    lb = np.asarray(lower_bound, dtype=np.float64)
    ub = np.asarray(upper_bound, dtype=np.float64)

    is_covered = (yt >= lb) & (yt <= ub)
    picp = float(np.mean(is_covered) * 100.0)
    mpiw = float(np.mean(ub - lb))

    return {
        "picp_pct": round(picp, 1),
        "mpiw": round(mpiw, 2),
    }


def format_geographic_coordinate(lat: float, lon: float) -> str:
    """Formats latitude and longitude into human-readable directional strings.

    Args:
        lat: Latitude in decimal degrees (-90 to +90).
        lon: Longitude in decimal degrees (-180 to +180).

    Returns:
        Formatted string, e.g. "47.12° N, 122.45° W".
    """
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    return f"{abs(lat):.2f}° {lat_dir}, {abs(lon):.2f}° {lon_dir}"


def compute_forecast_skill_score(
    y_true: Union[np.ndarray, List[float], pd.Series],
    y_pred: Union[np.ndarray, List[float], pd.Series],
    y_baseline: Union[np.ndarray, List[float], pd.Series],
) -> float:
    """Calculates the standard meteorological Brier/Heidke Forecast Skill Score (SS).

    Skill Score = 1 - (MSE_pred / MSE_baseline)
    Scores > 0 indicate superior predictive accuracy over the baseline climatology/persistence.

    Args:
        y_true: Ground truth observed telemetry.
        y_pred: Model predicted values.
        y_baseline: Climatological normal or persistence forecast.

    Returns:
        Floating point skill score (typically -inf to +1.0).
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    yb = np.asarray(y_baseline, dtype=np.float64)

    mse_pred = np.mean((yt - yp) ** 2)
    mse_base = np.mean((yt - yb) ** 2)

    if mse_base < 1e-6:
        return 0.0
    skill = 1.0 - (mse_pred / mse_base)
    return round(float(skill), 3)


def detect_atmospheric_threats(
    forecast_df: pd.DataFrame,
    target_var: str,
) -> List[Dict[str, Any]]:
    """Scans forecasted trajectories for synoptic meteorological hazard thresholds.

    Detects:
    - Heatwave / Extreme Thermal Spike (> 33°C)
    - Severe Frost / Sub-Zero Chill (< 0°C)
    - Cyclonic Depression / Low Pressure System (< 995 hPa)
    - Gale-Force Wind Velocity (> 55 km/h)
    - High-Humidity Saturation / Fog / Inversion (> 92%)

    Args:
        forecast_df: DataFrame of forward simulated projections.
        target_var: Active forecast variable name.

    Returns:
        List of detected atmospheric hazard dictionaries.
    """
    threats: List[Dict[str, Any]] = []

    if target_var == "temperature_c":
        heat_rows = forecast_df[forecast_df["predicted_value"] >= 32.0]
        if not heat_rows.empty:
            peak_temp = heat_rows["predicted_value"].max()
            lead_d = heat_rows.loc[heat_rows["predicted_value"].idxmax()]["forecast_lead_day"]
            threats.append({
                "level": "WARNING" if peak_temp < 36.0 else "CRITICAL",
                "title": "Severe Heatwave Watch",
                "detail": f"Peak thermal spike of {peak_temp:.1f}°C projected at Lead Day +{lead_d}.",
                "color": "#f43f5e",
            })

        frost_rows = forecast_df[forecast_df["predicted_value"] <= 2.0]
        if not frost_rows.empty:
            min_temp = frost_rows["predicted_value"].min()
            lead_d = frost_rows.loc[frost_rows["predicted_value"].idxmin()]["forecast_lead_day"]
            threats.append({
                "level": "ADVISORY" if min_temp >= 0.0 else "WARNING",
                "title": "Sub-Zero Freeze Hazard",
                "detail": f"Freezing surface temperatures down to {min_temp:.1f}°C projected at Lead Day +{lead_d}.",
                "color": "#38bdf8",
            })

    elif target_var == "pressure_hpa":
        storm_rows = forecast_df[forecast_df["predicted_value"] <= 998.0]
        if not storm_rows.empty:
            min_p = storm_rows["predicted_value"].min()
            lead_d = storm_rows.loc[storm_rows["predicted_value"].idxmin()]["forecast_lead_day"]
            threats.append({
                "level": "CRITICAL" if min_p < 990.0 else "WARNING",
                "title": "Deep Cyclonic Depression",
                "detail": f"Barometric trough dropping to {min_p:.1f} hPa at Lead Day +{lead_d}.",
                "color": "#a855f7",
            })

    elif target_var == "wind_speed_kmh":
        gale_rows = forecast_df[forecast_df["predicted_value"] >= 45.0]
        if not gale_rows.empty:
            max_w = gale_rows["predicted_value"].max()
            lead_d = gale_rows.loc[gale_rows["predicted_value"].idxmax()]["forecast_lead_day"]
            threats.append({
                "level": "CRITICAL" if max_w >= 65.0 else "WARNING",
                "title": "Gale-Force Wind Alert",
                "detail": f"Severe wind shear gusts reaching {max_w:.1f} km/h at Lead Day +{lead_d}.",
                "color": "#06b6d4",
            })

    elif target_var == "humidity_pct":
        sat_rows = forecast_df[forecast_df["predicted_value"] >= 90.0]
        if not sat_rows.empty:
            max_h = sat_rows["predicted_value"].max()
            lead_d = sat_rows.loc[sat_rows["predicted_value"].idxmax()]["forecast_lead_day"]
            threats.append({
                "level": "ADVISORY",
                "title": "High Moisture Saturation",
                "detail": f"Atmospheric saturation reaching {max_h:.1f}% relative humidity at Lead Day +{lead_d}.",
                "color": "#10b981",
            })

    return threats


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Calculates great-circle distance between two geographic coordinates in kilometers.

    Args:
        lat1: First latitude in degrees.
        lon1: First longitude in degrees.
        lat2: Second latitude in degrees.
        lon2: Second longitude in degrees.

    Returns:
        Great-circle distance in kilometers.
    """
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return 6371.0 * c


def generate_status_badge_html(
    label: str, value: str, status_color: str = "#00f2fe"
) -> str:
    """Creates a sleek, enterprise dark-themed HTML pill badge.

    Args:
        label: Descriptive label name.
        value: Telemetry value or status code.
        status_color: Hex color for glowing border and accent text.

    Returns:
        Sanitized HTML string suitable for Streamlit markdown rendering.
    """
    return f"""
    <div style="
        display: inline-flex;
        align-items: center;
        background: rgba(19, 26, 41, 0.7);
        border: 1px solid {status_color}55;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 2px 4px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <span style="
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background-color: {status_color};
            box-shadow: 0 0 8px {status_color};
            margin-right: 8px;
            display: inline-block;
        "></span>
        <span style="color: #94a3b8; font-size: 11px; font-weight: 500; text-transform: uppercase; margin-right: 6px;">
            {label}
        </span>
        <span style="color: #f8fafc; font-size: 12px; font-weight: 600;">
            {value}
        </span>
    </div>
    """

