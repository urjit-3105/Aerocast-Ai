"""Configuration module for Aero-Cast AI.

Defines geographic zones, bounding boxes, meteorological variable schemas,
simulation horizons, color palettes, and ML pipeline hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GeographicZone:
    """Represents a spatial simulation geographic bounding zone."""

    zone_id: str
    display_name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    center_lat: float
    center_lon: float
    elevation_range_m: Tuple[float, float]
    default_base_temp: float
    default_base_pressure: float
    description: str


@dataclass(frozen=True)
class VariableSchema:
    """Defines physical schemas, units, and ranges for meteorological metrics."""

    key: str
    name: str
    unit: str
    baseline_mean: float
    baseline_std: float
    min_physical: float
    max_physical: float
    color_scale: str
    primary_color: str
    secondary_color: str


# ==============================================================================
# 1. GEOGRAPHIC SIMULATION ZONES
# ==============================================================================
SIMULATION_ZONES: Dict[str, GeographicZone] = {
    "pac_nw": GeographicZone(
        zone_id="pac_nw",
        display_name="Pacific Northwest (US / Canada)",
        lat_min=44.0,
        lat_max=50.0,
        lon_min=-126.0,
        lon_max=-119.0,
        center_lat=47.0,
        center_lon=-122.5,
        elevation_range_m=(0.0, 3100.0),
        default_base_temp=12.5,
        default_base_pressure=1014.2,
        description="Maritime temperate zone featuring coastal ranges, Cascade orography, and frontal storm systems.",
    ),
    "alpine_med": GeographicZone(
        zone_id="alpine_med",
        display_name="Alpine Mediterranean (Central Europe)",
        lat_min=43.5,
        lat_max=48.5,
        lon_min=5.5,
        lon_max=15.0,
        center_lat=46.0,
        center_lon=10.2,
        elevation_range_m=(150.0, 4200.0),
        default_base_temp=14.0,
        default_base_pressure=1013.8,
        description="Complex alpine orographic barrier transitioning into Mediterranean and continental weather fronts.",
    ),
    "east_asia": GeographicZone(
        zone_id="east_asia",
        display_name="East Asia Monsoon Basin (Yangtze / Sea of Japan)",
        lat_min=26.0,
        lat_max=36.0,
        lon_min=116.0,
        lon_max=127.0,
        center_lat=31.0,
        center_lon=121.5,
        elevation_range_m=(10.0, 1800.0),
        default_base_temp=17.8,
        default_base_pressure=1012.5,
        description="Subtropical coastal plain characterized by high seasonal moisture flux, monsoon fronts, and cyclonic tracks.",
    ),
    "indo_gangetic": GeographicZone(
        zone_id="indo_gangetic",
        display_name="Indo-Gangetic Basin & Himalayan Arc",
        lat_min=24.5,
        lat_max=32.5,
        lon_min=74.5,
        lon_max=87.5,
        center_lat=28.5,
        center_lon=81.0,
        elevation_range_m=(120.0, 4800.0),
        default_base_temp=24.5,
        default_base_pressure=1009.5,
        description="Extreme elevation contrast with sharp thermal lapse rates and seasonal synoptic monsoon circulation.",
    ),
    "amazon_basin": GeographicZone(
        zone_id="amazon_basin",
        display_name="Amazon Tropical Equatorial Basin",
        lat_min=-7.5,
        lat_max=2.5,
        lon_min=-68.5,
        lon_max=-57.5,
        center_lat=-2.5,
        center_lon=-63.0,
        elevation_range_m=(25.0, 550.0),
        default_base_temp=27.2,
        default_base_pressure=1011.0,
        description="Equatorial high-convective rainforest system governed by diurnal heating cycles and high humidity.",
    ),
}

DEFAULT_ZONE_KEY: str = "pac_nw"


# ==============================================================================
# 2. METEOROLOGICAL FEATURE SCHEMAS
# ==============================================================================
METEOROLOGICAL_VARIABLES: Dict[str, VariableSchema] = {
    "temperature_c": VariableSchema(
        key="temperature_c",
        name="Surface Temperature",
        unit="°C",
        baseline_mean=16.5,
        baseline_std=8.2,
        min_physical=-45.0,
        max_physical=55.0,
        color_scale="Thermal",
        primary_color="#f43f5e",
        secondary_color="#fda4af",
    ),
    "pressure_hpa": VariableSchema(
        key="pressure_hpa",
        name="Atmospheric Pressure",
        unit="hPa",
        baseline_mean=1013.25,
        baseline_std=11.5,
        min_physical=910.0,
        max_physical=1065.0,
        color_scale="Viridis",
        primary_color="#8b5cf6",
        secondary_color="#c4b5fd",
    ),
    "humidity_pct": VariableSchema(
        key="humidity_pct",
        name="Relative Humidity",
        unit="%",
        baseline_mean=68.0,
        baseline_std=19.5,
        min_physical=5.0,
        max_physical=100.0,
        color_scale="Teal",
        primary_color="#10b981",
        secondary_color="#6ee7b7",
    ),
    "wind_speed_kmh": VariableSchema(
        key="wind_speed_kmh",
        name="Wind Velocity",
        unit="km/h",
        baseline_mean=19.0,
        baseline_std=11.2,
        min_physical=0.0,
        max_physical=180.0,
        color_scale="Turbo",
        primary_color="#06b6d4",
        secondary_color="#67e8f9",
    ),
}

DEFAULT_TARGET_VARIABLE: str = "temperature_c"


# ==============================================================================
# 3. SIMULATION HORIZONS & RESOLUTION
# ==============================================================================
MIN_LEAD_DAYS: int = 1
MAX_LEAD_DAYS: int = 42  # 6 full weeks
DEFAULT_LEAD_DAYS: int = 14

MIN_STATIONS: int = 12
MAX_STATIONS: int = 45
DEFAULT_STATIONS: int = 24

MIN_HISTORICAL_DAYS: int = 60
MAX_HISTORICAL_DAYS: int = 180
DEFAULT_HISTORICAL_DAYS: int = 90


# ==============================================================================
# 4. COLOR PALETTES & UI STYLING CONSTANTS
# ==============================================================================
THEME_COLORS = {
    "background": "#0b0f19",
    "surface_card": "#131a29",
    "surface_card_border": "#1e293b",
    "accent_cyan": "#00f2fe",
    "accent_blue": "#38bdf8",
    "accent_indigo": "#6366f1",
    "accent_purple": "#a855f7",
    "accent_rose": "#f43f5e",
    "accent_amber": "#f59e0b",
    "accent_emerald": "#10b981",
    "text_primary": "#f8fafc",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "gridline": "#1e293b",
    "confidence_fill": "rgba(56, 189, 248, 0.15)",
    "confidence_border": "rgba(56, 189, 248, 0.4)",
    "anomaly_positive": "#f43f5e",
    "anomaly_negative": "#38bdf8",
}


# ==============================================================================
# 5. MODEL ENGINE CONFIGURATION
# ==============================================================================
SUPPORTED_MODELS: List[str] = [
    "HistGradientBoostingRegressor",
    "GradientBoostingRegressor",
    "XGBoostRegressor",
]

DEFAULT_MODEL: str = "HistGradientBoostingRegressor"

FEATURE_LAGS: List[int] = [1, 2, 3, 7, 14]
ROLLING_WINDOWS: List[int] = [7, 14]
CONFIDENCE_LEVELS: List[float] = [0.80, 0.90, 0.95]
DEFAULT_CONFIDENCE_LEVEL: float = 0.90
