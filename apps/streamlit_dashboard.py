import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json

# Setup page layout
st.set_page_config(
    page_title="GridSight UK — Solar Forecasting Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for dark-themed, modern aesthetics
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0b1220;
        color: #e6edf7;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0e192e !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Metrics panel custom design */
    div[data-testid="metric-container"] {
        background-color: #0e192e;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: bold !important;
        color: #fdb813 !important; /* Solar Gold */
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #9fb0c7 !important;
    }
    
    /* General Titles */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    .subtitle {
        color: #9fb0c7;
        font-size: 16px;
        margin-top: -15px;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


def calculate_metrics(df):
    """Fallback function to calculate metrics on the fly from prediction data."""
    y = df["y_true"].to_numpy()
    q10 = df["q10"].to_numpy()
    q50 = df["q50"].to_numpy()
    q90 = df["q90"].to_numpy()
    
    def pinball(y_true, y_pred, q):
        d = y_true - y_pred
        return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))
        
    p10 = pinball(y, q10, 0.1)
    p50 = pinball(y, q50, 0.5)
    p90 = pinball(y, q90, 0.9)
    mean_p = (p10 + p50 + p90) / 3.0
    
    cov = float(np.mean((y >= q10) & (y <= q90)))
    
    mae_model = float(np.mean(np.abs(y - q50)))
    
    skill = None
    if "neso" in df.columns:
        neso = df["neso"].to_numpy()
        mae_neso = float(np.mean(np.abs(y - neso)))
        if mae_neso > 0:
            skill = 1.0 - mae_model / mae_neso
            
    return {
        "pinball_q10": p10,
        "pinball_q50": p50,
        "pinball_q90": p90,
        "mean_pinball": mean_p,
        "coverage_10-90": cov,
        "skill_vs_neso_q50": skill
    }


def load_data(model_type, horizon_hours, split):
    steps = {6: 12, 12: 24, 24: 48}[horizon_hours]
    
    if model_type == "Model A (Stacking)":
        base_dir = Path("artifacts/model") if steps == 48 else Path(f"artifacts/model_h{steps}")
    elif model_type == "Model B (Standalone LSTM-Q)":
        base_dir = Path("artifacts/lstm") if steps == 48 else Path(f"artifacts/lstm_h{steps}")
    else:  # Model C (Pretrained Chronos-Q)
        base_dir = Path("artifacts/chronos") if steps == 48 else Path(f"artifacts/chronos_h{steps}")
            
    parquet_file = base_dir / f"pred_{split}.parquet"
    metrics_file = base_dir / "metrics.json"
    
    if not parquet_file.exists():
        return None, None, f"No prediction data found at `{parquet_file}`. Please run training/inference for this configuration first."
        
    df = pd.read_parquet(parquet_file)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    
    # Try reading pre-saved metrics, fallback to calculating on-the-fly
    metrics_data = None
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f).get(split, {})
        except Exception:
            pass
            
    if not metrics_data:
        metrics_data = calculate_metrics(df)
        
    return df, metrics_data, None


# --- SIDEBAR CONTROL PANEL ---
st.sidebar.image("https://img.icons8.com/color/96/000000/sun--v1.png", width=60)
st.sidebar.title("GridSight UK")
st.sidebar.subheader("Forecasting Control Panel")

model_option = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["Model A (Stacking)", "Model B (Standalone LSTM-Q)", "Model C (Pretrained Chronos-Q)"]
)

horizon_option = st.sidebar.selectbox(
    "Select Forecast Horizon",
    [6, 12, 24],
    format_func=lambda x: f"{x} Hours Ahead ({x*2} steps)"
)

split_option = st.sidebar.selectbox(
    "Select Data Split",
    ["val", "test"],
    format_func=lambda x: "Validation Split" if x == "val" else "Test Split"
)

# --- LOAD DATA ---
df, metrics, error_msg = load_data(model_option, horizon_option, split_option)

# --- MAIN APP LAYOUT ---
st.title("☀️ GridSight UK — Probabilistic Solar Forecasting")
st.markdown("<div class='subtitle'>MSc Data Science Team Project · Solar Generation Forecast Diagnostics Dashboard</div>", unsafe_allow_html=True)

if error_msg:
    st.error(error_msg)
    
    st.info("""
    ### How to train/run this model:
    Run the following command in your terminal to generate prediction outputs:
    
    * **Model A Stacking**:
      ```bash
      ./venv/bin/python -m gridsight.models.stacking --horizon-steps %d
      ```
    * **Model B Standalone LSTM**:
      ```bash
      ./venv/bin/python -m gridsight.models.lstm.train --horizon-steps %d
      ```
    * **Model C Pretrained Chronos-Q**:
      ```bash
      ./venv/bin/python -m gridsight.models.chronos --horizon-steps %d --gold-dir data/gold/gold_features_h%d
      ```
    """ % ({6: 12, 12: 24, 24: 48}[horizon_option], {6: 12, 12: 24, 24: 48}[horizon_option], {6: 12, 12: 24, 24: 48}[horizon_option], {6: 12, 12: 24, 24: 48}[horizon_option]))

else:
    # Scale variables if predictions are in capacity factor (CF)
    is_cf = ("target" in df and df["target"].iloc[0] == "target_cf")
    scale_factor = df["capacity_mwp"].to_numpy() if (is_cf and "capacity_mwp" in df) else 1.0
    
    # Scale all columns to MW
    df["y_true_mw"] = df["y_true"].to_numpy() * scale_factor
    df["q10_mw"] = df["q10"].to_numpy() * scale_factor
    df["q50_mw"] = df["q50"].to_numpy() * scale_factor
    df["q90_mw"] = df["q90"].to_numpy() * scale_factor
    if "neso" in df.columns:
        df["neso_mw"] = df["neso"].to_numpy() * scale_factor

    # Date Slider filter
    min_date = df["timestamp_utc"].min().to_pydatetime()
    max_date = df["timestamp_utc"].max().to_pydatetime()
    
    # Select default first 5 days for a clean visualization
    default_end = min(max_date, df["timestamp_utc"].iloc[min(len(df)-1, 240)].to_pydatetime())
    
    start_time, end_time = st.sidebar.slider(
        "Select Time Window",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, default_end),
        format="MMM DD, YYYY"
    )
    
    df_filtered = df[(df["timestamp_utc"] >= start_time) & (df["timestamp_utc"] <= end_time)].copy()

    # --- KPI CARDS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Mean Pinball Loss",
            value=f"{metrics.get('mean_pinball', 0.0):.4f}"
        )
    with col2:
        cov = metrics.get('coverage_10-90', 0.0)
        st.metric(
            label="Empirical Coverage (Target: 80%)",
            value=f"{cov:.2%}",
            delta=f"{cov - 0.80:.2%}"
        )
    with col3:
        skill = metrics.get('skill_vs_neso_q50', None)
        st.metric(
            label="q50 Skill vs NESO Baseline",
            value=f"{skill:.2%}" if skill is not None else "N/A",
            delta="Better than NESO" if (skill is not None and skill > 0) else None
        )
    with col4:
        crossing = metrics.get('crossing_rate', 0.0)
        st.metric(
            label="Quantile Crossing Rate",
            value=f"{crossing:.2%}",
            delta="OK" if crossing == 0 else "Anomalous",
            delta_color="normal" if crossing == 0 else "inverse"
        )

    # --- TIME SERIES FORECAST PLOT ---
    st.subheader("📊 Probabilistic Solar Generation Time Series (MW)")
    
    fig = go.Figure()
    
    # Shaded 80% Prediction Interval (q10 to q90)
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp_utc"],
        y=df_filtered["q90_mw"],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name="q90 Upper Bound"
    ))
    
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp_utc"],
        y=df_filtered["q10_mw"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(253, 184, 19, 0.18)', # Amber/Gold transparent fill
        name="80% Prediction Interval (q10-q90)"
    ))
    
    # Median Forecast (q50)
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp_utc"],
        y=df_filtered["q50_mw"],
        mode='lines',
        line=dict(color='#fdb813', width=2.5),
        name="q50 Median Forecast"
    ))
    
    # Ground Truth Actual
    fig.add_trace(go.Scatter(
        x=df_filtered["timestamp_utc"],
        y=df_filtered["y_true_mw"],
        mode='lines',
        line=dict(color='#3ddc97', width=2),
        name="Actual Solar Generation"
    ))
    
    # NESO Operator Baseline (if present)
    if "neso_mw" in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered["timestamp_utc"],
            y=df_filtered["neso_mw"],
            mode='lines',
            line=dict(color='#ec4899', width=1.5, dash='dash'),
            name="NESO Operator Baseline"
        ))
        
    fig.update_layout(
        plot_bgcolor='#0b1220',
        paper_bgcolor='#0b1220',
        font=dict(color='#e6edf7'),
        xaxis=dict(
            gridcolor='#1b2740',
            zerolinecolor='#1b2740',
            title="Time (UTC)"
        ),
        yaxis=dict(
            gridcolor='#1b2740',
            zerolinecolor='#1b2740',
            title="Solar Power (MW)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=20, b=40),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- SECONDARY DIAGNOSTICS ROW ---
    diag_col1, diag_col2 = st.columns(2)
    
    with diag_col1:
        st.subheader("🎯 Quantile Calibration (Reliability)")
        
        # Calculate calibration points
        y_np = df["y_true"].to_numpy()
        q10_np = df["q10"].to_numpy()
        q50_np = df["q50"].to_numpy()
        q90_np = df["q90"].to_numpy()
        
        emp = [
            float(np.mean(y_np <= q10_np)),
            float(np.mean(y_np <= q50_np)),
            float(np.mean(y_np <= q90_np))
        ]
        
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(color='#5a6b86', width=1.5, dash='dash'),
            name="Ideal Calibration"
        ))
        fig_calib.add_trace(go.Scatter(
            x=[0.1, 0.5, 0.9], y=emp,
            mode='lines+markers',
            marker=dict(size=10, color='#fdb813'),
            line=dict(color='#fdb813', width=2),
            name="Empirical Coverage"
        ))
        
        fig_calib.update_layout(
            plot_bgcolor='#0b1220',
            paper_bgcolor='#0b1220',
            font=dict(color='#e6edf7'),
            xaxis=dict(
                gridcolor='#1b2740',
                zerolinecolor='#1b2740',
                title="Nominal Quantile (Target)",
                range=[0, 1]
            ),
            yaxis=dict(
                gridcolor='#1b2740',
                zerolinecolor='#1b2740',
                title="Empirical Cumulative Probability",
                range=[0, 1]
            ),
            height=320,
            margin=dict(l=40, r=40, t=20, b=40),
            showlegend=False
        )
        st.plotly_chart(fig_calib, use_container_width=True)
        
    with diag_col2:
        st.subheader("📊 Forecast Error by Hour of Day (MW)")
        
        df_hourly = df.copy()
        df_hourly["hour"] = df_hourly["timestamp_utc"].dt.hour
        h_mae = df_hourly.groupby("hour").apply(
            lambda g: np.mean(np.abs(g["y_true_mw"] - g["q50_mw"])),
            include_groups=False
        )
        
        fig_error = go.Figure(go.Bar(
            x=h_mae.index,
            y=h_mae.values,
            marker_color='rgba(253, 184, 19, 0.7)',
            marker_line=dict(color='#fdb813', width=1)
        ))
        
        fig_error.update_layout(
            plot_bgcolor='#0b1220',
            paper_bgcolor='#0b1220',
            font=dict(color='#e6edf7'),
            xaxis=dict(
                gridcolor='#1b2740',
                zerolinecolor='#1b2740',
                title="Hour of Day (UTC)",
                tickmode='linear',
                tick0=0,
                dtick=2
            ),
            yaxis=dict(
                gridcolor='#1b2740',
                zerolinecolor='#1b2740',
                title="MAE (MW)"
            ),
            height=320,
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig_error, use_container_width=True)

    # --- RAW PREDICTIONS TABLE DATA EXPLORER ---
    with st.expander("🔍 Explore Raw Prediction Data Table"):
        st.dataframe(
            df_filtered[["timestamp_utc", "y_true_mw", "q10_mw", "q50_mw", "q90_mw"]].rename(
                columns={
                    "timestamp_utc": "Timestamp (UTC)",
                    "y_true_mw": "Actual Solar (MW)",
                    "q10_mw": "q10 Prediction (MW)",
                    "q50_mw": "q50 Prediction (MW)",
                    "q90_mw": "q90 Prediction (MW)"
                }
            ).reset_index(drop=True),
            use_container_width=True
        )
