# 🌪️ Aero-Cast AI
### Spatial-Temporal Weather & Atmospheric Forecasting Simulator

[![CI](https://github.com/urjit-3105/Aerocast-Ai/actions/workflows/ci.yml/badge.svg)](https://github.com/urjit-3105/Aerocast-Ai/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-7.1+-3F4F75.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8+-F7931E.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Aero-Cast AI is a multi-file, production-grade spatial-temporal atmospheric forecasting application designed to simulate and analyze long-range meteorological trajectories across dynamic geographic grids. It fuses synthetic atmospheric telemetry engines with gradient boosting machine learning pipelines to predict multi-week weather anomalies with calibrated confidence intervals.

---

## 🌟 Key Capabilities

- **Synthetic Telemetry Simulation Engine**:
  - Incorporates environmental tropospheric lapse rates (-6.5°C / 1,000m) and hydrostatic barometric altitude decay.
  - Simulates traveling synoptic Rossby wave disturbances and localized pressure gradients.
  - Models spatial autocorrelation and autoregressive AR(1) colored noise for realistic persistence.
  - Couples atmospheric variables: geostrophic wind acceleration, barometric humidity saturation, and cloud albedo cooling.

- **Spatial-Temporal Machine Learning Pipeline**:
  - Cyclical temporal feature engineering (sine/cosine transformations for Julian day, month, and day-of-week).
  - Station-level multi-step rolling lag statistics (lags 1–14, rolling means, std, min/max).
  - Spatial distance weighting and network-wide lag interaction terms.
  - Multi-horizon recursive forecasting with expanding uncertainty envelopes ($\sigma \propto \sqrt{1 + \kappa h}$) modeling dynamical atmospheric chaos.

- **Automated Synoptic Threat Matrix**:
  - Scans forward multi-week forecasts for severe atmospheric hazards: Heatwave Watches (> 32°C), Sub-Zero Freezes (< 0°C), Deep Cyclonic Depressions (< 998 hPa), and Gale-Force Wind shears (> 45 km/h).

- **Interactive Enterprise Dark Dashboard**:
  - **Tab 1 ("Spatial Map & Heatmap")**: High-resolution MapLibre/Plotly dark-matter geospatial maps, synoptic continuous density fields, topographic lapse profiles, and interactive time-scrubber controls.
  - **Tab 2 ("Long-Range Trend Analysis")**: Time-series trajectory comparisons of observed historical data against ML forecasts, shaded confidence bands, multi-station comparative overlays, and synoptic wave anomaly dispersion matrices.
  - **Tab 3 ("System Telemetry & Diagnostics")**: Real-time evaluation scorecards (MAE, RMSE, $R^2$, Heidke/Brier Skill Score, Anomaly Deviation Score), feature importance rankings, uncertainty growth curves, and CSV data export.

---

## 📁 Repository Architecture

```text
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions continuous integration workflow
├── .streamlit/
│   └── config.toml         # Dark enterprise UI theme and server configuration
├── config.py               # Geographic bounding boxes, variable schemas, and palettes
├── data_generator.py       # Synthetic spatial-temporal telemetry engine with atmospheric physics
├── model_engine.py         # Spatial-temporal gradient boosting regression and uncertainty pipeline
├── utils.py                # Accuracy metrics (MAE, RMSE, ADS, Skill Score) and threat detectors
├── app.py                  # Main Streamlit enterprise dashboard application
├── test_pipeline.py        # Integration test verification suite
├── Dockerfile              # Production containerization buildfile
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Production dependencies
├── .gitignore              # Git ignore rules
└── README.md               # Comprehensive project documentation
```

---

## 🚀 Quickstart & Installation

### Option 1: Local Python Environment
```bash
# 1. Clone the repository
git clone https://github.com/urjit-3105/Aerocast-Ai.git
cd Aerocast-Ai

# 2. Set up virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit
streamlit run app.py
```

### Option 2: Run via Docker
```bash
docker compose up --build
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Supported Meteorological Metrics

| Variable | Unit | Baseline Range | Physics Model |
| :--- | :--- | :--- | :--- |
| **Surface Temperature** | °C | -45.0 to +55.0 | Tropospheric lapse rate (-6.5°C/km) + seasonal radiation wave |
| **Atmospheric Pressure**| hPa | 910.0 to 1065.0 | Hydrostatic barometric equation + traveling Rossby waves |
| **Relative Humidity**   | % | 5.0% to 100.0% | Cyclonic trough saturation + orographic condensation |
| **Wind Velocity**       | km/h | 0.0 to 180.0 | Geostrophic horizontal pressure gradient ($|\nabla P|$) + venturi elevation boost |

---

## 🛠️ Machine Learning Algorithms

- **`HistGradientBoostingRegressor`** (Default): High-performance histogram-based gradient boosting optimized for numerical time-series.
- **`GradientBoostingRegressor`**: Classic ensemble boosting regressor with tree-based split criteria.
- **`XGBoostRegressor`**: Extreme Gradient Boosting with shrinkage and subsampling for regularized anomaly tracking.

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
