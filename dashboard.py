from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ncf.data_loader import load_data
from src.ncf.data_processing import run_data_quality_pipeline
from src.ncf.feature_engineering import (
    engineer_features,
    create_forecasting_target,
)
from src.ncf.data_split import (
    chronological_train_val_test_split,
    split_features_target,
)
from src.ncf.models import load_saved_model


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "institutions"
    / "agg_10_minutes"
    / "0.csv"
)

TARGET_COLUMN = "future_n_bytes"

st.set_page_config(
    page_title="Network Capacity Forecasting",
    page_icon="📡",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD SAVED MODEL AND SCALER
# --------------------------------------------------

@st.cache_resource
def get_model():
    model, scaler = load_saved_model()
    return model, scaler


# --------------------------------------------------
# PREPARE TEST DATA
# --------------------------------------------------

@st.cache_data(show_spinner="Preparing network traffic data...")
def prepare_test_data(data_path):

    df = load_data(data_path)

    df = run_data_quality_pipeline(df)

    df = engineer_features(df)

    df = create_forecasting_target(
        df,
        target_column="n_bytes",
        horizon=1,
    )

    train_df, validation_df, test_df = (
        chronological_train_val_test_split(df)
    )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = split_features_target(
        train_df,
        validation_df,
        test_df,
        target_column=TARGET_COLUMN,
    )

    # The scaler was fitted on training features only.
    # Exclude id_time, exactly as in the training pipeline.
    feature_names = X_test.drop(
        columns=["id_time"]
    ).columns.tolist()

    X_test_features = X_test[feature_names]

    return X_test, X_test_features, y_test, feature_names


# --------------------------------------------------
# MAKE PREDICTIONS
# --------------------------------------------------

@st.cache_data(show_spinner="Generating predictions...")
def make_predictions(_model, _scaler, data_path):

    X_test, X_test_features, y_test, feature_names = (
        prepare_test_data(data_path)
    )

    X_test_scaled = _scaler.transform(X_test_features)

    y_pred_log = _model.predict(
        X_test_scaled,
        verbose=0,
    ).reshape(-1)

    # The model predicts log1p(bytes).
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)

    # y_test is still in the original bytes scale.
    y_actual = np.asarray(y_test).reshape(-1)

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    results = pd.DataFrame(
        {
            "Observation": np.arange(1, len(y_actual) + 1),
            "Actual Traffic (bytes)": y_actual,
            "Predicted Traffic (bytes)": y_pred,
            "Absolute Error (bytes)": np.abs(y_actual - y_pred),
        }
    )

    return results, mae, rmse, r2, X_test_scaled, y_actual, feature_names


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📡 Network Forecasting")

st.sidebar.caption(
    "AI-powered network traffic analytics"
)

st.sidebar.markdown("---")

st.sidebar.subheader("Data source")

st.sidebar.code(
    "CESNET-TimeSeries24\n"
    "10-minute aggregation\n"
    "Institution 0"
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This dashboard evaluates the saved model "
    "using the chronological test dataset."
)


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("📡 Network Capacity Forecasting System")

st.write(
    "Monitor model performance, compare actual and predicted "
    "network traffic, and explore traffic capacity thresholds."
)

st.caption(
    "Forecast target: network bytes in the next observation "
    "(10-minute aggregation)."
)

st.markdown("---")


# --------------------------------------------------
# CHECK FILES AND LOAD RESULTS
# --------------------------------------------------

if not DATA_PATH.exists():

    st.error(
        f"Dataset not found:\n\n{DATA_PATH}\n\n"
        "Place the CESNET CSV file at the expected path."
    )

    st.stop()


model_path = (
    PROJECT_DIR
    / "artifacts"
    / "network_forecasting_model.keras"
)

scaler_path = (
    PROJECT_DIR
    / "artifacts"
    / "network_forecasting_model_scaler.joblib"
)

if not model_path.exists() or not scaler_path.exists():

    st.error(
        "The saved model or scaler is missing. "
        "Run `python main.py` to train and save them first."
    )

    st.stop()


try:

    model, scaler = get_model()

    (
        results,
        mae,
        rmse,
        r2,
        X_test_scaled,
        y_actual,
        feature_names,
    ) = make_predictions(
        model,
        scaler,
        str(DATA_PATH),
    )

except Exception as exc:

    st.error(
        f"Could not load the model or prepare predictions: {exc}"
    )

    st.stop()


# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "MAE",
        f"{mae:,.0f} bytes",
    )

with col2:
    st.metric(
        "RMSE",
        f"{rmse:,.0f} bytes",
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2:.3f}",
    )

with col4:
    st.metric(
        "Test Observations",
        f"{len(results):,}",
    )


st.caption(
    "Metrics are recalculated on the test dataset using the saved model. "
    "They may differ from previously recorded results if the data or "
    "pipeline has changed."
)

st.markdown("---")


# --------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

st.subheader("Actual vs. Predicted Network Traffic")

number_of_points = st.slider(
    "Number of test observations to display",
    min_value=100,
    max_value=len(results),
    value=min(500, len(results)),
    step=100,
)

chart_data = results.head(number_of_points)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=chart_data["Observation"],
        y=chart_data["Actual Traffic (bytes)"],
        mode="lines",
        name="Actual Traffic",
        line=dict(width=2),
    )
)

fig.add_trace(
    go.Scatter(
        x=chart_data["Observation"],
        y=chart_data["Predicted Traffic (bytes)"],
        mode="lines",
        name="Predicted Traffic",
        line=dict(width=2, dash="dash"),
    )
)

fig.update_layout(
    xaxis_title="Test Observation (chronological order)",
    yaxis_title="Network Traffic (bytes)",
    hovermode="x unified",
    legend_title="Traffic",
    height=480,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)


# --------------------------------------------------
# CAPACITY THRESHOLD EXPLORER
# --------------------------------------------------

st.markdown("---")

st.subheader("Capacity Threshold Explorer")

st.write(
    "Set an illustrative traffic threshold to identify test observations "
    "whose actual or predicted byte counts exceed it. This is a planning "
    "aid, not a measurement of link utilization."
)

max_traffic = float(
    max(
        results["Actual Traffic (bytes)"].max(),
        results["Predicted Traffic (bytes)"].max(),
        1,
    )
)

threshold = st.number_input(
    "Traffic threshold (bytes per 10-minute interval)",
    min_value=0.0,
    max_value=max_traffic * 10,
    value=float(results["Actual Traffic (bytes)"].quantile(0.90)),
    step=max(1.0, max_traffic / 100),
)

actual_exceedances = int(
    (results["Actual Traffic (bytes)"] > threshold).sum()
)

predicted_exceedances = int(
    (results["Predicted Traffic (bytes)"] > threshold).sum()
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Actual observations above threshold",
        f"{actual_exceedances:,}",
    )

with col2:
    st.metric(
        "Predicted observations above threshold",
        f"{predicted_exceedances:,}",
    )

st.caption(
    "Use a threshold derived from your network's actual capacity and "
    "measurement interval before making operational decisions."
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.markdown("---")

st.subheader("Feature Importance")

st.write(
    "Permutation importance estimates how much prediction error changes "
    "when a feature's values are shuffled. Calculating it may take time."
)

if st.button("Calculate Feature Importance"):

    with st.spinner("Calculating feature importance..."):

        try:

            from src.ncf.explainability import (
                calculate_feature_importance,
            )

            importance_df = calculate_feature_importance(
                model,
                X_test_scaled,
                y_actual,
                feature_names,
            )

            importance_df = importance_df.sort_values(
                "importance_mean",
                ascending=True,
            ).tail(15)

            fig_importance = px.bar(
                importance_df,
                x="importance_mean",
                y="feature",
                orientation="h",
                error_x="importance_std",
                title="Top 15 Features by Permutation Importance",
                labels={
                    "importance_mean": "Increase in prediction error",
                    "feature": "Network Feature",
                },
            )

            fig_importance.update_layout(height=600)

            st.plotly_chart(
                fig_importance,
                use_container_width=True,
            )

            st.caption(
                "Negative importance means shuffling a feature did not "
                "increase error in this evaluation; it does not prove "
                "that the feature is inherently unimportant."
            )

        except Exception as exc:

            st.error(
                f"Could not calculate feature importance: {exc}"
            )


# --------------------------------------------------
# PREDICTION TABLE AND DOWNLOAD
# --------------------------------------------------

st.markdown("---")

st.subheader("Prediction Details")

st.dataframe(
    results.head(number_of_points),
    use_container_width=True,
    hide_index=True,
)

csv_data = results.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download predictions as CSV",
    data=csv_data,
    file_name="network_traffic_predictions.csv",
    mime="text/csv",
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Network Capacity Forecasting System | "
    "TensorFlow | CESNET-TimeSeries24 | Streamlit"
)