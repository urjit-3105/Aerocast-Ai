"""Synthetic Meteorological Telemetry Engine for Aero-Cast AI.

Generates realistic spatial-temporal atmospheric time-series data across
geographic station grids incorporating environmental lapse rates, synoptic
pressure systems, spatial autocorrelation, and physical multi-variable coupling.
"""

from typing import Dict, List, Optional, Tuple
import datetime
import numpy as np
import pandas as pd

from config import (
    SIMULATION_ZONES,
    GeographicZone,
    METEOROLOGICAL_VARIABLES,
    DEFAULT_ZONE_KEY,
)


def generate_station_grid(
    zone: GeographicZone,
    num_stations: int = 24,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates a synthetic network of weather telemetry observation stations.

    Distributes stations pseudo-randomly within the zone bounding box and
    assigns realistic synthetic topographic elevations based on spatial coordinates.

    Args:
        zone: GeographicZone defining boundary coordinates and elevation limits.
        num_stations: Number of observation telemetry stations to generate.
        seed: Random seed for reproducible spatial layouts.

    Returns:
        DataFrame containing station metadata: station_id, station_name,
        latitude, longitude, elevation_m, and zone_id.
    """
    rng = np.random.RandomState(seed)

    # Generate pseudo-random coordinates within zone bounds
    lats = rng.uniform(zone.lat_min + 0.2, zone.lat_max - 0.2, size=num_stations)
    lons = rng.uniform(zone.lon_min + 0.2, zone.lon_max - 0.2, size=num_stations)

    # Synthetic Topography: Multi-frequency ridge & valley simulation
    # Normalized coordinates from 0 to 1
    norm_lat = (lats - zone.lat_min) / max(1e-5, (zone.lat_max - zone.lat_min))
    norm_lon = (lons - zone.lon_min) / max(1e-5, (zone.lon_max - zone.lon_min))

    # Orographic elevation surface
    elev_min, elev_max = zone.elevation_range_m
    elev_span = elev_max - elev_min

    topo_raw = (
        0.5 * np.sin(norm_lat * np.pi * 2.2)
        + 0.3 * np.cos(norm_lon * np.pi * 1.8)
        + 0.2 * np.sin((norm_lat + norm_lon) * np.pi * 3.0)
    )
    # Normalize between 0 and 1 with a slight exponential bias for mountainous peaks
    topo_norm = (topo_raw - topo_raw.min()) / max(1e-5, (topo_raw.max() - topo_raw.min()))
    elevations = elev_min + (topo_norm**1.3) * elev_span

    # Generate realistic callsigns/station names
    station_names = [
        f"AERO-{zone.zone_id.upper()[:3]}-{i+1:02d}"
        for i in range(num_stations)
    ]
    station_ids = [f"STN_{i+1:03d}" for i in range(num_stations)]

    df_stations = pd.DataFrame(
        {
            "station_id": station_ids,
            "station_name": station_names,
            "latitude": np.round(lats, 4),
            "longitude": np.round(lons, 4),
            "elevation_m": np.round(elevations, 1),
            "zone_id": zone.zone_id,
        }
    )
    return df_stations


def _compute_spatial_distance_matrix(
    lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Computes an approximate pairwise Euclidean-spherical distance matrix (in km).

    Args:
        lats: 1D array of latitude coordinates.
        lons: 1D array of longitude coordinates.

    Returns:
        2D symmetric matrix of pairwise distances in kilometers.
    """
    n = len(lats)
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    # Vectorized Haversine-like calculation
    dlat = lat_rad[:, np.newaxis] - lat_rad[np.newaxis, :]
    dlon = lon_rad[:, np.newaxis] - lon_rad[np.newaxis, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, np.newaxis])
        * np.cos(lat_rad[np.newaxis, :])
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    dist_km = 6371.0 * c
    return dist_km


def generate_atmospheric_telemetry(
    zone: GeographicZone,
    start_date: str = "2026-05-01",
    num_days: int = 90,
    num_stations: int = 24,
    noise_scale: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulates a multi-variable spatial-temporal atmospheric telemetry dataset.

    Implements realistic meteorological dynamics:
    - Standard Tropospheric Environmental Lapse Rate (approx. -6.5°C per 1,000m elevation).
    - Hydrostatic Barometric Pressure Equation (exponential altitude decay).
    - Annual sinusoidal solar irradiance cycle.
    - Traveling synoptic barometric Rossby wave disturbances.
    - Spatial covariance (correlated perturbations among neighboring stations).
    - Autoregressive AR(1) temporal memory for persistent atmospheric patterns.
    - Physical cross-coupling between pressure gradient, wind speed, and humidity.

    Args:
        zone: Target GeographicZone configuration dataclass.
        start_date: ISO date string for start of historical telemetry.
        num_days: Total consecutive days to simulate.
        num_stations: Total telemetry stations in the sensor array.
        noise_scale: Multiplier for stochastic atmospheric fluctuations.
        seed: Random seed for deterministic generation.

    Returns:
        DataFrame with columns: timestamp, station_id, station_name,
        latitude, longitude, elevation_m, temperature_c, pressure_hpa,
        humidity_pct, wind_speed_kmh, zone_id.
    """
    rng = np.random.RandomState(seed)

    # 1. Establish Station Network
    stations_df = generate_station_grid(zone, num_stations=num_stations, seed=seed)
    n_stn = len(stations_df)
    lats = stations_df["latitude"].values
    lons = stations_df["longitude"].values
    elevs = stations_df["elevation_m"].values

    # Pairwise spatial distance matrix for spatial covariance
    dist_matrix = _compute_spatial_distance_matrix(lats, lons)
    # Spatial correlation kernel: exp(-dist / correlation_length)
    spatial_corr_length = 250.0  # km
    spatial_cov = np.exp(-dist_matrix / spatial_corr_length)
    # Cholesky decomposition for generating spatially correlated noise
    # Add small diagonal jitter for positive definiteness
    jitter = 1e-4 * np.eye(n_stn)
    try:
        chol_spatial = np.linalg.cholesky(spatial_cov + jitter)
    except np.linalg.LinAlgError:
        chol_spatial = np.eye(n_stn)

    # 2. Time-series Calendar Grid
    base_datetime = datetime.datetime.fromisoformat(start_date)
    dates = [base_datetime + datetime.timedelta(days=d) for d in range(num_days)]
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])

    # 3. Base Climatological Profiles
    # Environmental Lapse Rate: 6.5°C / 1000m
    lapse_rate = 0.0065  # °C / m
    temp_elevation_offset = -elevs * lapse_rate

    # Barometric pressure formula: P = P0 * exp(-z / 8400)
    pressure_elevation_ratio = np.exp(-elevs / 8400.0)

    # Seasonal temperature cycle (peaks around day 200 in northern hemisphere, ~day 20 in southern)
    hemisphere_phase = 200.0 if zone.center_lat >= 0 else 20.0
    seasonal_temp_amplitude = 9.0 * (1.0 + abs(zone.center_lat) / 50.0)
    seasonal_temp_curve = seasonal_temp_amplitude * np.cos(
        2 * np.pi * (day_of_year - hemisphere_phase) / 365.25
    )

    # 4. Synoptic Wave Simulator (Traveling Cyclonic / Anticyclonic cells)
    # Simulates Rossby atmospheric wave fronts moving across longitude
    wave_period_days = 9.5
    wave_speed_deg_lon_per_day = 1.8  # eastward progression
    synoptic_pressure_amp = 14.0  # hPa amplitude

    # Allocate telemetry matrices: shape (num_days, num_stations)
    temp_matrix = np.zeros((num_days, n_stn))
    pres_matrix = np.zeros((num_days, n_stn))
    humi_matrix = np.zeros((num_days, n_stn))
    wind_matrix = np.zeros((num_days, n_stn))

    # Initialize AR(1) state variables for noise persistence
    ar_phi = 0.76  # persistence parameter
    noise_temp_prev = np.zeros(n_stn)
    noise_pres_prev = np.zeros(n_stn)
    noise_humi_prev = np.zeros(n_stn)

    for t_idx, d in enumerate(dates):
        # Synoptic wave phase across longitude
        lon_travel = (d - base_datetime).days * wave_speed_deg_lon_per_day
        synoptic_front = np.sin(
            2 * np.pi * (lons + lon_travel) / (zone.lon_max - zone.lon_min)
            + 2 * np.pi * t_idx / wave_period_days
        )

        # Spatially correlated random innovations
        raw_innovations = rng.randn(3, n_stn)
        corr_innov_temp = chol_spatial @ raw_innovations[0]
        corr_innov_pres = chol_spatial @ raw_innovations[1]
        corr_innov_humi = chol_spatial @ raw_innovations[2]

        # AR(1) colored noise updates
        noise_temp = (
            ar_phi * noise_temp_prev
            + np.sqrt(1 - ar_phi**2) * corr_innov_temp * 2.2 * noise_scale
        )
        noise_pres = (
            ar_phi * noise_pres_prev
            + np.sqrt(1 - ar_phi**2) * corr_innov_pres * 3.5 * noise_scale
        )
        noise_humi = (
            ar_phi * noise_humi_prev
            + np.sqrt(1 - ar_phi**2) * corr_innov_humi * 6.0 * noise_scale
        )

        noise_temp_prev = noise_temp
        noise_pres_prev = noise_pres
        noise_humi_prev = noise_humi

        # --- ATMOSPHERIC PRESSURE ---
        # Sea-level base + synoptic front wave + spatial noise
        sea_level_pressure = (
            zone.default_base_pressure
            + synoptic_front * synoptic_pressure_amp
            + noise_pres
        )
        # Apply elevation barometric scaling
        stn_pressure = sea_level_pressure * pressure_elevation_ratio
        pres_matrix[t_idx, :] = stn_pressure

        # --- TEMPERATURE ---
        # Base + Seasonal + Elevation lapse + Synoptic thermal advection + noise
        # Low pressure cells bring cloudiness/frontal boundary modulations
        synoptic_temp_mod = -synoptic_front * 1.8  # colder in trough
        stn_temp = (
            zone.default_base_temp
            + seasonal_temp_curve[t_idx]
            + temp_elevation_offset
            + synoptic_temp_mod
            + noise_temp
        )
        temp_matrix[t_idx, :] = stn_temp

        # --- RELATIVE HUMIDITY ---
        # Inversely tied to pressure trough (low pressure = clouds/rain = high humidity)
        pressure_anomaly = sea_level_pressure - zone.default_base_pressure
        humidity_base = 65.0 - (pressure_anomaly * 1.4) + noise_humi
        # Mountainous orographic lift adds humidity
        orographic_lift = (elevs / max(1.0, zone.elevation_range_m[1])) * 12.0
        stn_humidity = np.clip(humidity_base + orographic_lift, 8.0, 98.0)
        humi_matrix[t_idx, :] = stn_humidity

        # --- WIND VELOCITY ---
        # Geostrophic balance: wind velocity is proportional to local pressure gradient
        # Compute local gradient magnitude from synoptic perturbation
        pressure_gradient_mag = np.abs(
            synoptic_pressure_amp
            * np.cos(
                2 * np.pi * (lons + lon_travel) / (zone.lon_max - zone.lon_min)
                + 2 * np.pi * t_idx / wave_period_days
            )
        )
        # Mountain venturi effects: wind increases with elevation
        elev_wind_boost = (elevs / 1000.0) * 3.5
        wind_gust_noise = rng.exponential(scale=3.0 * noise_scale, size=n_stn)
        stn_wind = (
            8.0
            + pressure_gradient_mag * 1.6
            + elev_wind_boost
            + wind_gust_noise
        )
        wind_matrix[t_idx, :] = np.clip(stn_wind, 1.0, 160.0)

    # 5. Assemble Structured Long-Format Telemetry DataFrame
    records: List[Dict] = []
    for t_idx, d_time in enumerate(dates):
        iso_timestamp = d_time.strftime("%Y-%m-%d")
        for s_idx in range(n_stn):
            records.append(
                {
                    "timestamp": iso_timestamp,
                    "date": d_time.date(),
                    "station_id": stations_df.iloc[s_idx]["station_id"],
                    "station_name": stations_df.iloc[s_idx]["station_name"],
                    "latitude": stations_df.iloc[s_idx]["latitude"],
                    "longitude": stations_df.iloc[s_idx]["longitude"],
                    "elevation_m": stations_df.iloc[s_idx]["elevation_m"],
                    "temperature_c": np.round(temp_matrix[t_idx, s_idx], 2),
                    "pressure_hpa": np.round(pres_matrix[t_idx, s_idx], 2),
                    "humidity_pct": np.round(humi_matrix[t_idx, s_idx], 1),
                    "wind_speed_kmh": np.round(wind_matrix[t_idx, s_idx], 1),
                    "zone_id": zone.zone_id,
                }
            )

    full_telemetry_df = pd.DataFrame(records)
    return full_telemetry_df


def compute_climatological_baselines(
    telemetry_df: pd.DataFrame,
    target_var: str = "temperature_c",
) -> pd.DataFrame:
    """Computes station-level baseline climatological statistics.

    Args:
        telemetry_df: Historical atmospheric telemetry DataFrame.
        target_var: Meteorological metric column name.

    Returns:
        DataFrame with station_id, mean baseline, standard deviation,
        minimum, and maximum observed historical values.
    """
    stats_df = (
        telemetry_df.groupby("station_id")[target_var]
        .agg(
            climatology_mean="mean",
            climatology_std="std",
            climatology_min="min",
            climatology_max="max",
        )
        .reset_index()
    )
    return stats_df
