# 🌪️ Aero-Cast AI
### Spatial-Temporal Weather & Atmospheric Forecasting Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-7.1+-3F4F75.svg)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8+-F7931E.svg)](https://scikit-learn.org/)
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

- **Interactive Enterprise Dark Dashboard**:
  - **Tab 1 ("Spatial Map & Heatmap")**: High-resolution Plotly scatter maps displaying regional sensor arrays, topographic lapse profiles, and interactive time-scrubber controls.
  - **Tab 2 ("Long-Range Trend Analysis")**: Time-series trajectory comparisons of observed historical data against ML forecasts, shaded confidence bands, climatological baselines, and milestone horizon cards.
  - **Tab 3 ("System Telemetry & Diagnostics")**: Real-time evaluation scorecards (MAE, RMSE, $R^2$, Anomaly Deviation Score), feature importance rankings, uncertainty growth curves, and CSV data export.

---

## 📁 Repository Architecture

```text
├── config.py            # Geographic bounding boxes, meteorological variable schemas, and color palettes
├── data_generator.py    # Synthetic spatial-temporal telemetry engine with atmospheric physics
├── model_engine.py      # Spatial-temporal gradient boosting regression and uncertainty pipeline
├── utils.py             # Accuracy metrics (MAE, RMSE, ADS), interval coverage, and formatting helpers
├── app.py               # Main Streamlit enterprise dashboard application
├── requirements.txt     # Production dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Comprehensive project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/urjit-3105/Aerocast-Ai.git
cd Aerocast-Ai
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Production Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will launch automatically in your default browser at `http://localhost:8501`.

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
