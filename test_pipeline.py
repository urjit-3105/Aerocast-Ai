"""Unit and integration test script for Aero-Cast AI."""

import sys
import time

def run_tests():
    print("Testing imports...")
    import config
    import data_generator
    import model_engine
    import utils
    print("All modules imported successfully.")

    print("Testing data generator...")
    zone = config.SIMULATION_ZONES["pac_nw"]
    df = data_generator.generate_atmospheric_telemetry(
        zone=zone,
        start_date="2026-05-01",
        num_days=30,
        num_stations=8,
        noise_scale=1.0,
        seed=42,
    )
    print(f"Telemetry generated: {df.shape[0]} rows, columns: {list(df.columns)}")
    assert len(df) == 30 * 8, f"Expected {30*8} rows, got {len(df)}"

    print("Testing forecaster...")
    forecaster = model_engine.SpatialTemporalForecaster(
        model_type="HistGradientBoostingRegressor",
        target_var="temperature_c",
        confidence_level=0.90,
        n_estimators=30,
        max_depth=4,
    )
    artifact = forecaster.fit_and_forecast(df, lead_days=5)
    print(f"Forecast generated: {artifact.forecast_df.shape[0]} rows.")
    print(f"Historical MAE: {artifact.metrics_historical['mae']:.2f} °C")
    assert len(artifact.forecast_df) == 5 * 8, f"Expected {5*8} rows, got {len(artifact.forecast_df)}"

    print("All pipeline tests PASSED perfectly!")

if __name__ == "__main__":
    run_tests()

