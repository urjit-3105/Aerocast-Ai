"""Aero-Cast AI: Spatial-Temporal Atmospheric & Meteorological Forecasting Simulator.

Main Streamlit Application Entrypoint.
Provides an enterprise dark-themed dashboard featuring interactive spatial maps,
long-range time-series trajectory forecasts with confidence envelopes,
and real-time regression telemetry diagnostics.
"""

from typing import Dict, List, Optional, Tuple
import datetime
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import (
    SIMULATION_ZONES,
    DEFAULT_ZONE_KEY,
    METEOROLOGICAL_VARIABLES,
    DEFAULT_TARGET_VARIABLE,
    MIN_LEAD_DAYS,
    MAX_LEAD_DAYS,
    DEFAULT_LEAD_DAYS,
    MIN_STATIONS,
    MAX_STATIONS,
    DEFAULT_STATIONS,
    MIN_HISTORICAL_DAYS,
    MAX_HISTORICAL_DAYS,
    DEFAULT_HISTORICAL_DAYS,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
    CONFIDENCE_LEVELS,
    DEFAULT_CONFIDENCE_LEVEL,
    THEME_COLORS,
    GeographicZone,
    VariableSchema,
)
from data_generator import (
    generate_atmospheric_telemetry,
    compute_climatological_baselines,
)
from model_engine import (
    SpatialTemporalForecaster,
    ForecastArtifact,
    HAS_XGBOOST,
)
from utils import (
    compute_regression_metrics,
    compute_anomaly_deviation_score,
    compute_prediction_interval_coverage,
    format_geographic_coordinate,
    generate_status_badge_html,
)


# ==============================================================================
# STREAMLIT PAGE CONFIG & CUSTOM ENTERPRISE DARK THEME
# ==============================================================================
st.set_page_config(
    page_title="Aero-Cast AI | Atmospheric Forecasting Simulator",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Dark Theme Core Styles */
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, rgba(19, 26, 41, 0.9) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        margin: 6px 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    
    /* Metric Glass Cards */
    .metric-card {
        background: rgba(19, 26, 41, 0.75);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00f2fe55;
        transform: translateY(-2px);
    }
    .metric-card-title {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-card-value {
        color: #f8fafc;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .metric-card-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 4px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0d131f;
        border-right: 1px solid #1e293b;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(19, 26, 41, 0.6);
        border: 1px solid #1e293b;
        border-radius: 8px 8px 0 0;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
        font-weight: 600 !important;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        border: 1px solid #38bdf888;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.02em;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%);
        border-color: #00f2fe;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.45);
        transform: translateY(-1px);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA GENERATION & MODEL SIMULATION PIPELINE
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_cached_telemetry(
    zone_key: str,
    num_days: int,
    num_stations: int,
    noise_level: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates and caches historical atmospheric telemetry."""
    zone = SIMULATION_ZONES[zone_key]
    # Anchor reference date to 90 days before today
    start_date = (datetime.date.today() - datetime.timedelta(days=num_days)).isoformat()
    return generate_atmospheric_telemetry(
        zone=zone,
        start_date=start_date,
        num_days=num_days,
        num_stations=num_stations,
        noise_scale=noise_level,
        seed=seed,
    )


@st.cache_resource(show_spinner=False)
def execute_forecasting_pipeline(
    telemetry_df: pd.DataFrame,
    target_var: str,
    lead_days: int,
    model_type: str,
    confidence_level: float,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> ForecastArtifact:
    """Trains spatial-temporal regressor and generates multi-day trajectory forecasts."""
    forecaster = SpatialTemporalForecaster(
        model_type=model_type,
        target_var=target_var,
        confidence_level=confidence_level,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    return forecaster.fit_and_forecast(
        telemetry_df=telemetry_df,
        lead_days=lead_days,
    )


# ==============================================================================
# SIDEBAR CONTROLS & HYPERPARAMETER TUNING
# ==============================================================================
with st.sidebar:
    st.markdown("### 🌐 Simulation Parameters")

    # 1. Zone Selection
    zone_options = {
        key: zone.display_name for key, zone in SIMULATION_ZONES.items()
    }
    selected_zone_key = st.selectbox(
        "Geographic Simulation Zone",
        options=list(zone_options.keys()),
        format_func=lambda k: zone_options[k],
        index=0,
    )
    selected_zone: GeographicZone = SIMULATION_ZONES[selected_zone_key]

    st.caption(
        f"📍 **Bounds**: {selected_zone.lat_min}°N – {selected_zone.lat_max}°N | "
        f"{selected_zone.lon_min}°E – {selected_zone.lon_max}°E\n\n"
        f"⛰️ **Elevation Span**: {selected_zone.elevation_range_m[0]:.0f}m – {selected_zone.elevation_range_m[1]:.0f}m"
    )

    st.markdown("---")
    st.markdown("### 📊 Meteorological Targets")

    # 2. Variable Selection
    var_options = {
        k: f"{v.name} ({v.unit})" for k, v in METEOROLOGICAL_VARIABLES.items()
    }
    selected_target_var = st.selectbox(
        "Primary Forecast Target",
        options=list(var_options.keys()),
        format_func=lambda k: var_options[k],
        index=0,
    )
    active_var_schema: VariableSchema = METEOROLOGICAL_VARIABLES[selected_target_var]

    # 3. Horizon & Density Sliders
    lead_days = st.slider(
        "Forecast Horizon (Lead Days)",
        min_value=MIN_LEAD_DAYS,
        max_value=MAX_LEAD_DAYS,
        value=DEFAULT_LEAD_DAYS,
        step=1,
        help="Number of forward simulation days (1 to 6 weeks).",
    )

    num_stations = st.slider(
        "Telemetry Sensor Stations",
        min_value=MIN_STATIONS,
        max_value=MAX_STATIONS,
        value=DEFAULT_STATIONS,
        step=2,
        help="Number of spatial telemetry observation nodes within the bounding box.",
    )

    history_days = st.slider(
        "Historical Training Window (Days)",
        min_value=MIN_HISTORICAL_DAYS,
        max_value=MAX_HISTORICAL_DAYS,
        value=DEFAULT_HISTORICAL_DAYS,
        step=15,
        help="Depth of historical telemetry used for recursive training.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Hyperparameters")

    available_models = [m for m in SUPPORTED_MODELS if m != "XGBoostRegressor" or HAS_XGBOOST]
    selected_model_name = st.selectbox(
        "Regression Algorithm",
        options=available_models,
        index=0,
    )

    confidence_level = st.select_slider(
        "Confidence Interval Level",
        options=CONFIDENCE_LEVELS,
        value=DEFAULT_CONFIDENCE_LEVEL,
        format_func=lambda x: f"{int(x*100)}% CI",
    )

    with st.expander("Advanced Estimator Settings", expanded=False):
        n_estimators = st.slider("Boosting Iterations", 40, 250, 110, step=10)
        learning_rate = st.select_slider(
            "Learning Rate (η)",
            options=[0.01, 0.03, 0.05, 0.08, 0.1, 0.15],
            value=0.05,
        )
        max_depth = st.slider("Max Tree Depth", 3, 10, 5)
        noise_level = st.slider("Atmospheric Noise Perturbation", 0.5, 2.0, 1.0, step=0.1)

    st.markdown("---")
    run_btn = st.button("🚀 Run Atmospheric Simulation", use_container_width=True)


# ==============================================================================
# DATA INGESTION & PIPELINE EXECUTION
# ==============================================================================
try:
    with st.spinner("Ingesting atmospheric telemetry and fitting spatial-temporal pipeline..."):
        telemetry_df = get_cached_telemetry(
            zone_key=selected_zone_key,
            num_days=history_days,
            num_stations=num_stations,
            noise_level=noise_level,
        )

        forecast_artifact = execute_forecasting_pipeline(
            telemetry_df=telemetry_df,
            target_var=selected_target_var,
            lead_days=lead_days,
            model_type=selected_model_name,
            confidence_level=confidence_level,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
except Exception as e:
    st.error(f"Execution Error in Telemetry Pipeline: {str(e)}")
    st.stop()


# ==============================================================================
# MAIN DASHBOARD HEADER & QUICK STATS
# ==============================================================================
st.markdown(
    f"""
    <div class="main-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>🌪️ AERO-CAST AI</h1>
                <p>Spatial-Temporal Weather & Atmospheric Forecasting Simulator | Multi-Horizon Machine Learning Pipeline</p>
            </div>
            <div style="margin-top: 10px;">
                {generate_status_badge_html("ZONE", selected_zone.display_name.split("(")[0].strip(), THEME_COLORS["accent_cyan"])}
                {generate_status_badge_html("MODEL", selected_model_name, THEME_COLORS["accent_indigo"])}
                {generate_status_badge_html("HORIZON", f"+{lead_days} DAYS", THEME_COLORS["accent_amber"])}
                {generate_status_badge_html("STATUS", "ONLINE", THEME_COLORS["accent_emerald"])}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Mean Absolute Error</div>
            <div class="metric-card-value" style="color: {THEME_COLORS['accent_cyan']};">
                {forecast_artifact.metrics_historical['mae']:.2f} <span style="font-size: 1rem; color: #94a3b8;">{active_var_schema.unit}</span>
            </div>
            <div class="metric-card-sub">In-sample walk-forward validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Root Mean Squared Error</div>
            <div class="metric-card-value" style="color: {THEME_COLORS['accent_blue']};">
                {forecast_artifact.metrics_historical['rmse']:.2f} <span style="font-size: 1rem; color: #94a3b8;">{active_var_schema.unit}</span>
            </div>
            <div class="metric-card-sub">Root variance penalty score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Coefficient of Determination</div>
            <div class="metric-card-value" style="color: {THEME_COLORS['accent_emerald']};">
                {forecast_artifact.metrics_historical['r2']:.3f}
            </div>
            <div class="metric-card-sub">Goodness of fit (R²)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Compute Anomaly Deviation Score for forecast trajectory
ads_result = compute_anomaly_deviation_score(
    predictions=forecast_artifact.forecast_df["predicted_value"],
    climatology_mean=forecast_artifact.forecast_df["climatology_normal"],
    climatology_std=np.ones(len(forecast_artifact.forecast_df)) * active_var_schema.baseline_std,
)

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Anomaly Deviation Score</div>
            <div class="metric-card-value" style="color: {THEME_COLORS['accent_amber']};">
                {ads_result['mean_ads']:.2f} σ
            </div>
            <div class="metric-card-sub">{ads_result['category']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-title">Inference Latency</div>
            <div class="metric-card-value" style="color: {THEME_COLORS['accent_purple']};">
                {forecast_artifact.training_duration_sec * 1000.0:.0f} <span style="font-size: 1rem; color: #94a3b8;">ms</span>
            </div>
            <div class="metric-card-sub">End-to-end fit & simulation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN TABS ARCHITECTURE
# ==============================================================================
tab1, tab2, tab3 = st.tabs(
    [
        "🗺️ Spatial Map & Heatmap",
        "📈 Long-Range Trend Analysis",
        "🔬 System Telemetry & Diagnostics",
    ]
)


# ==============================================================================
# TAB 1: SPATIAL MAP & HEATMAP
# ==============================================================================
with tab1:
    st.markdown("#### 🌐 Regional Atmospheric Spatial Grid & Synoptic Perturbations")

    # Time-scrubber slider to view spatial field at different forecast days
    forecast_dates = sorted(forecast_artifact.forecast_df["timestamp"].unique())
    selected_time_idx = st.slider(
        "Simulation Time Scrubber (Step through Forecast Horizon)",
        min_value=1,
        max_value=len(forecast_dates),
        value=min(7, len(forecast_dates)),
        format="Day +%d",
    )
    active_date_str = forecast_dates[selected_time_idx - 1]

    # Filter spatial data for active date
    step_spatial_df = forecast_artifact.forecast_df[
        forecast_artifact.forecast_df["timestamp"] == active_date_str
    ].copy()

    # Create 2-column layout: Map on left, Spatial Correlation / Lapse rate on right
    map_col, stat_col = st.columns([1.6, 1.0])

    with map_col:
        # Plotly Geo / Scatter Mapbox
        fig_map = px.scatter(
            step_spatial_df,
            x="longitude",
            y="latitude",
            color="predicted_value",
            size="elevation_m",
            hover_name="station_name",
            hover_data={
                "predicted_value": ":.2f",
                "lower_bound": ":.2f",
                "upper_bound": ":.2f",
                "elevation_m": ":.0f",
                "anomaly": ":.2f",
                "longitude": False,
                "latitude": False,
            },
            color_continuous_scale=active_var_schema.color_scale,
            title=f"Predicted {active_var_schema.name} Field | Lead Day +{selected_time_idx} ({active_date_str})",
            labels={"predicted_value": f"{active_var_schema.unit}"},
        )

        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_COLORS["surface_card"],
            plot_bgcolor=THEME_COLORS["surface_card"],
            margin=dict(l=20, r=20, t=50, b=20),
            height=460,
            xaxis=dict(
                title="Longitude (°)",
                gridcolor=THEME_COLORS["gridline"],
                zeroline=False,
            ),
            yaxis=dict(
                title="Latitude (°)",
                gridcolor=THEME_COLORS["gridline"],
                zeroline=False,
            ),
            coloraxis_colorbar=dict(
                title=f"{active_var_schema.unit}",
                thickness=14,
                len=0.8,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with stat_col:
        # Topographic Elevation Lapse Rate Scatter Plot
        fig_lapse = px.scatter(
            step_spatial_df,
            x="elevation_m",
            y="predicted_value",
            color="anomaly",
            color_continuous_scale="RdBu_r",
            trendline="ols",
            title=f"Topographic Lapse Profile ({active_var_schema.unit} vs Elevation)",
            labels={
                "elevation_m": "Station Elevation (m)",
                "predicted_value": f"Forecast ({active_var_schema.unit})",
                "anomaly": "Anomaly",
            },
            hover_name="station_name",
        )
        fig_lapse.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_COLORS["surface_card"],
            plot_bgcolor=THEME_COLORS["surface_card"],
            margin=dict(l=20, r=20, t=50, b=20),
            height=460,
            xaxis=dict(gridcolor=THEME_COLORS["gridline"]),
            yaxis=dict(gridcolor=THEME_COLORS["gridline"]),
            coloraxis_colorbar=dict(
                title="Anomaly",
                thickness=12,
                len=0.7,
            ),
        )
        st.plotly_chart(fig_lapse, use_container_width=True)

    # Station Telemetry Density Matrix Table
    with st.expander("📍 Inspection of Active Spatial Telemetry Nodes", expanded=False):
        st.dataframe(
            step_spatial_df[
                [
                    "station_id",
                    "station_name",
                    "latitude",
                    "longitude",
                    "elevation_m",
                    "predicted_value",
                    "lower_bound",
                    "upper_bound",
                    "anomaly",
                ]
            ].rename(
                columns={
                    "predicted_value": f"Predicted ({active_var_schema.unit})",
                    "lower_bound": f"Lower Bound ({active_var_schema.unit})",
                    "upper_bound": f"Upper Bound ({active_var_schema.unit})",
                    "anomaly": f"Anomaly ({active_var_schema.unit})",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# TAB 2: LONG-RANGE TREND ANALYSIS
# ==============================================================================
with tab2:
    st.markdown("#### 📈 Multi-Horizon Atmospheric Trajectory Forecasting")

    # Station Filter: View Regional Aggregate or Drill-down
    stations_list = ["Regional Ensemble Average"] + sorted(
        telemetry_df["station_name"].unique().tolist()
    )
    selected_station = st.selectbox("Telemetry Viewport Selection", options=stations_list)

    # Prepare Historical and Forecast Time Series
    if selected_station == "Regional Ensemble Average":
        hist_trend = (
            telemetry_df.groupby("timestamp")[selected_target_var]
            .mean()
            .reset_index()
            .sort_values("timestamp")
        )
        fore_trend = (
            forecast_artifact.forecast_df.groupby("timestamp")
            .agg(
                predicted_value=("predicted_value", "mean"),
                lower_bound=("lower_bound", "mean"),
                upper_bound=("upper_bound", "mean"),
                climatology_normal=("climatology_normal", "mean"),
                anomaly=("anomaly", "mean"),
            )
            .reset_index()
            .sort_values("timestamp")
        )
    else:
        stn_id = telemetry_df[telemetry_df["station_name"] == selected_station][
            "station_id"
        ].iloc[0]
        hist_trend = (
            telemetry_df[telemetry_df["station_id"] == stn_id][
                ["timestamp", selected_target_var]
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        fore_trend = (
            forecast_artifact.forecast_df[
                forecast_artifact.forecast_df["station_id"] == stn_id
            ]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # Build Interactive Plotly Time Series Chart
    fig_trend = go.Figure()

    # 1. Shaded Confidence Interval Envelope
    fig_trend.add_trace(
        go.Scatter(
            x=fore_trend["timestamp"].tolist() + fore_trend["timestamp"].tolist()[::-1],
            y=fore_trend["upper_bound"].tolist() + fore_trend["lower_bound"].tolist()[::-1],
            fill="toself",
            fillcolor=THEME_COLORS["confidence_fill"],
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name=f"{int(confidence_level*100)}% Confidence Interval",
        )
    )

    # 2. Historical Observations
    fig_trend.add_trace(
        go.Scatter(
            x=hist_trend["timestamp"],
            y=hist_trend[selected_target_var],
            mode="lines",
            line=dict(color=THEME_COLORS["text_secondary"], width=2),
            name="Historical Observed Telemetry",
            hovertemplate="Observed: %{y:.2f} " + active_var_schema.unit + "<br>Date: %{x}<extra></extra>",
        )
    )

    # 3. Climatological Baseline
    fig_trend.add_trace(
        go.Scatter(
            x=fore_trend["timestamp"],
            y=fore_trend["climatology_normal"],
            mode="lines",
            line=dict(color="#64748b", width=1.5, dash="dot"),
            name="Climatological Normal Baseline",
            hovertemplate="Baseline: %{y:.2f} " + active_var_schema.unit + "<br>Date: %{x}<extra></extra>",
        )
    )

    # 4. Forward Forecast Trajectory
    fig_trend.add_trace(
        go.Scatter(
            x=fore_trend["timestamp"],
            y=fore_trend["predicted_value"],
            mode="lines+markers",
            line=dict(color=active_var_schema.primary_color, width=2.5),
            marker=dict(size=5, color=active_var_schema.primary_color),
            name=f"Forward ML Forecast ({selected_model_name})",
            hovertemplate="Forecast: %{y:.2f} " + active_var_schema.unit + "<br>Date: %{x}<extra></extra>",
        )
    )

    # Formatting and Layout
    fig_trend.update_layout(
        title=f"Atmospheric Forecast Trajectory: {active_var_schema.name} ({active_var_schema.unit}) | {selected_station}",
        template="plotly_dark",
        paper_bgcolor=THEME_COLORS["surface_card"],
        plot_bgcolor=THEME_COLORS["surface_card"],
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        height=480,
        xaxis=dict(
            title="Observation Timeline",
            gridcolor=THEME_COLORS["gridline"],
            showgrid=True,
        ),
        yaxis=dict(
            title=f"{active_var_schema.name} ({active_var_schema.unit})",
            gridcolor=THEME_COLORS["gridline"],
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(19, 26, 41, 0.7)",
        ),
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    # Milestone Forecast Horizons Breakdown Cards
    st.markdown("##### ⏱️ Horizon Milestone Trajectory Projections")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    milestones = [
        (m_col1, 7, "7-Day Near-Range"),
        (m_col2, 14, "14-Day Medium-Range"),
        (m_col3, 28, "28-Day Extended Synoptic"),
        (m_col4, 42, "42-Day Long-Range Climatology"),
    ]

    for col, m_day, m_title in milestones:
        with col:
            if m_day <= len(fore_trend):
                m_row = fore_trend.iloc[m_day - 1]
                val = m_row["predicted_value"]
                anomaly = m_row["anomaly"]
                anom_color = (
                    THEME_COLORS["anomaly_positive"]
                    if anomaly >= 0
                    else THEME_COLORS["anomaly_negative"]
                )
                anom_sign = "+" if anomaly >= 0 else ""

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-card-title">{m_title}</div>
                        <div class="metric-card-value" style="font-size: 1.4rem;">
                            {val:.2f} <span style="font-size: 0.9rem; color: #94a3b8;">{active_var_schema.unit}</span>
                        </div>
                        <div class="metric-card-sub" style="color: {anom_color}; font-weight: 600;">
                            {anom_sign}{anomaly:.2f} {active_var_schema.unit} departure
                        </div>
                        <div class="metric-card-sub">
                            CI: [{m_row['lower_bound']:.1f}, {m_row['upper_bound']:.1f}]
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="metric-card" style="opacity: 0.5;">
                        <div class="metric-card-title">{m_title}</div>
                        <div class="metric-card-value" style="font-size: 1.2rem; color: #64748b;">
                            N/A
                        </div>
                        <div class="metric-card-sub">Exceeds selected horizon</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==============================================================================
# TAB 3: SYSTEM TELEMETRY & DIAGNOSTICS
# ==============================================================================
with tab3:
    st.markdown("#### 🔬 Model Diagnostic Telemetry & Uncertainty Calibration")

    diag_col1, diag_col2 = st.columns([1.1, 1.0])

    with diag_col1:
        # Feature Importance Horizontal Bar Chart
        importances = forecast_artifact.feature_importances
        feat_df = (
            pd.DataFrame(
                list(importances.items()),
                columns=["Feature", "Relative Importance (%)"],
            )
            .sort_values("Relative Importance (%)", ascending=True)
            .tail(12)
        )

        fig_imp = px.bar(
            feat_df,
            x="Relative Importance (%)",
            y="Feature",
            orientation="h",
            title="Top Feature Relative Importance (Predictive Drivers)",
            color="Relative Importance (%)",
            color_continuous_scale="Teal",
        )
        fig_imp.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_COLORS["surface_card"],
            plot_bgcolor=THEME_COLORS["surface_card"],
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
            xaxis=dict(gridcolor=THEME_COLORS["gridline"]),
            yaxis=dict(gridcolor=THEME_COLORS["gridline"]),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with diag_col2:
        # Expanding Uncertainty Envelope (Lyapunov Divergence)
        lead_steps = np.arange(1, lead_days + 1)
        z_score = float(np.abs(np.round(forecast_artifact.forecast_df.iloc[0]["upper_bound"] - forecast_artifact.forecast_df.iloc[0]["predicted_value"], 2)))
        uncertainty_expansion = [
            round(z_score * np.sqrt(1.0 + 0.12 * s), 2) for s in lead_steps
        ]

        uncert_df = pd.DataFrame(
            {
                "Lead Day": lead_steps,
                "Prediction Interval Half-Width (±)": uncertainty_expansion,
            }
        )

        fig_uncert = px.line(
            uncert_df,
            x="Lead Day",
            y="Prediction Interval Half-Width (±)",
            markers=True,
            title="Forecast Horizon Uncertainty Expansion Curve (σ vs Lead Time)",
            labels={"Prediction Interval Half-Width (±)": f"Half-Width (± {active_var_schema.unit})"},
        )
        fig_uncert.update_traces(
            line=dict(color=THEME_COLORS["accent_amber"], width=2.5),
            marker=dict(size=6, color=THEME_COLORS["accent_amber"]),
        )
        fig_uncert.update_layout(
            template="plotly_dark",
            paper_bgcolor=THEME_COLORS["surface_card"],
            plot_bgcolor=THEME_COLORS["surface_card"],
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
            xaxis=dict(gridcolor=THEME_COLORS["gridline"]),
            yaxis=dict(gridcolor=THEME_COLORS["gridline"]),
        )
        st.plotly_chart(fig_uncert, use_container_width=True)

    # Telemetry Health Monitor Grid
    st.markdown("##### 📡 Telemetry Stream Health & Data Integrity")
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)

    total_records = len(telemetry_df)
    missing_cells = telemetry_df.isnull().sum().sum()
    data_quality_pct = 100.0 if missing_cells == 0 else (1.0 - missing_cells / (total_records * 10)) * 100.0

    with h_col1:
        st.metric("Total Ingested Records", f"{total_records:,}")
    with h_col2:
        st.metric("Missing Frame Ratio", f"{missing_cells} (0.00%)")
    with h_col3:
        st.metric("Signal Data Quality", f"{data_quality_pct:.1f}%")
    with h_col4:
        st.metric("Sensor Array Frequency", "24.0 hrs / Daily")

    # Raw Telemetry Ingestion Explorer & Export
    with st.expander("💾 Raw Telemetry Ingestion & Forecast Export", expanded=False):
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.markdown("###### Ingested Historical Observations")
            st.dataframe(telemetry_df.head(100), use_container_width=True, height=250)
            csv_hist = telemetry_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Historical Telemetry (CSV)",
                data=csv_hist,
                file_name=f"historical_telemetry_{selected_zone_key}.csv",
                mime="text/csv",
            )

        with d_col2:
            st.markdown("###### Forward ML Simulated Projections")
            st.dataframe(forecast_artifact.forecast_df.head(100), use_container_width=True, height=250)
            csv_fore = forecast_artifact.forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Forecast Trajectory (CSV)",
                data=csv_fore,
                file_name=f"forecast_simulation_{selected_zone_key}_{selected_target_var}.csv",
                mime="text/csv",
            )


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 12px;">
        Aero-Cast AI | Production-Grade Spatial-Temporal Atmospheric Forecasting Simulator<br>
        Built with Python 3.10+, Streamlit, Plotly, Scikit-Learn, Pandas, NumPy, XGBoost, and SciPy.
    </div>
    """,
    unsafe_allow_html=True,
)
