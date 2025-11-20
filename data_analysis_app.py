"""
Data Analysis Dashboard for cleaned_data.csv
-------------------------------------------
This Streamlit app reads `cleaned_data.csv`, provides exploratory data analysis
views, and surfaces the key model metrics obtained previously from `model1.py`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
DATA_PATH = "/Users/jeetashwar/Desktop/gdgc/cleaned_data.csv"
TARGET_COLUMN = "relationship_probability"
CATEGORICAL_HINT = ["F5", "F7", "F8", "F9", "F10", "F13"]
MODEL_METRICS = {
    "RMSE": 4.6647,
    "MAE": 3.7859,
    "R²": 0.7323,
}

st.set_page_config(
    page_title="Relationship Probability Insights",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Relationship Probability Insights")
st.caption(
    "Explore `cleaned_data.csv`, slice segments with interactive filters, "
    "and view the XGBoost model metrics captured in `model1.py`."
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Load the dataset once and reuse it across reruns."""
    return pd.read_csv(path)


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical column lists."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    # Ensure hinted categorical columns are treated as categorical even if numeric
    for col in CATEGORICAL_HINT:
        if col in df.columns and col not in categorical_cols:
            categorical_cols.append(col)
            if col in numeric_cols:
                numeric_cols.remove(col)
    return numeric_cols, categorical_cols


def filter_dataframe(
    df: pd.DataFrame,
    target_range: Tuple[float, float],
    categorical_filters: Dict[str, List[str]],
) -> pd.DataFrame:
    """Filter by target range and categorical selections."""
    filtered = df[
        (df[TARGET_COLUMN] >= target_range[0]) & (df[TARGET_COLUMN] <= target_range[1])
    ]
    for col, allowed_values in categorical_filters.items():
        if allowed_values:
            filtered = filtered[filtered[col].isin(allowed_values)]
    return filtered


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Please check the path.")
    st.stop()

numeric_cols, categorical_cols = detect_column_types(df)


# -----------------------------------------------------------------------------
# Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    st.success("Data loaded successfully ✅")

    st.subheader("Target filter")
    target_min, target_max = float(df[TARGET_COLUMN].min()), float(df[TARGET_COLUMN].max())
    selected_target_range = st.slider(
        "Relationship probability range",
        min_value=target_min,
        max_value=target_max,
        value=(target_min, target_max),
    )

    st.subheader("Categorical filters")
    cat_filters: Dict[str, List[str]] = {}
    for col in categorical_cols:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        if len(unique_vals) <= 30:  # keep UI manageable
            cat_filters[col] = st.multiselect(
                f"{col} values", unique_vals, default=unique_vals, key=f"cat_{col}"
            )
        else:
            cat_filters[col] = unique_vals  # skip UI when too many levels

filtered_df = filter_dataframe(df, selected_target_range, cat_filters)

st.info(f"Current selection: {filtered_df.shape[0]} rows filtered from {df.shape[0]}")


# -----------------------------------------------------------------------------
# Tabs for content sections
# -----------------------------------------------------------------------------
overview_tab, visuals_tab, model_tab = st.tabs(["Data Overview", "Visuals", "Model Metrics"])


# -----------------------------------------------------------------------------
# Overview Tab
# -----------------------------------------------------------------------------
with overview_tab:
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{filtered_df.shape[0]:,}")
    c2.metric("Columns", f"{filtered_df.shape[1]:,}")
    c3.metric("Numeric columns", len(numeric_cols))

    st.markdown("### Descriptive statistics")
    st.dataframe(filtered_df.describe(include="all").T.fillna(""), use_container_width=True)

    st.markdown("### Preview filtered rows")
    st.dataframe(filtered_df.head(100), use_container_width=True, height=350)

    st.markdown("### Download selection")
    st.download_button(
        label="Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_cleaned_data.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# Visuals Tab
# -----------------------------------------------------------------------------
with visuals_tab:
    st.subheader("Interactive charts")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Histogram**")
        if numeric_cols:
            hist_col = st.selectbox("Numeric column", numeric_cols, key="hist_col")
            hist_fig = px.histogram(
                filtered_df,
                x=hist_col,
                nbins=35,
                color_discrete_sequence=["#00B5AD"],
            )
            st.plotly_chart(hist_fig, use_container_width=True)
        else:
            st.warning("No numeric columns detected.")

    with col2:
        st.markdown("**Box plot**")
        if numeric_cols:
            box_col = st.selectbox("Value column", numeric_cols, key="box_col")
            hue_col = st.selectbox(
                "Color by (optional)", ["None"] + categorical_cols, key="box_hue"
            )
            box_fig = px.box(
                filtered_df,
                y=box_col,
                color=None if hue_col == "None" else hue_col,
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.warning("No numeric columns detected.")

    st.markdown("**Scatter plot**")
    if len(numeric_cols) >= 2:
        scatter_cols = st.multiselect(
            "Select X and Y", numeric_cols, default=numeric_cols[:2], max_selections=2
        )
        color_col = st.selectbox(
            "Color by", ["None"] + categorical_cols + [TARGET_COLUMN], key="scatter_color"
        )
        if len(scatter_cols) == 2:
            scatter_fig = px.scatter(
                filtered_df,
                x=scatter_cols[0],
                y=scatter_cols[1],
                color=None if color_col == "None" else color_col,
                color_discrete_sequence=px.colors.qualitative.Vivid,
                hover_data=[TARGET_COLUMN],
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Choose two columns to render the scatter plot.")
    else:
        st.info("Need at least two numeric columns for scatter visualization.")

    st.markdown("**Correlation heatmap**")
    if len(numeric_cols) >= 2:
        corr_matrix = filtered_df[numeric_cols].corr()
        heatmap = px.imshow(
            corr_matrix,
            color_continuous_scale="Viridis",
            aspect="auto",
        )
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("Need numeric columns to compute correlations.")


# -----------------------------------------------------------------------------
# Model Metrics Tab
# -----------------------------------------------------------------------------
with model_tab:
    st.subheader("XGBoost model snapshot (from model1.py)")
    st.write(
        "Metrics captured from the XGBRegressor pipeline trained in `model1.py` "
        "using a train/test split on `cleaned_data.csv`."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE", f"{MODEL_METRICS['RMSE']:.4f}")
    m2.metric("MAE", f"{MODEL_METRICS['MAE']:.4f}")
    m3.metric("R²", f"{MODEL_METRICS['R²']:.4f}")

    st.markdown("### Model context")
    st.markdown(
        "- Estimator: `XGBRegressor` with 400 trees, depth 5, learning rate 0.05\n"
        "- Preprocessing: One-hot encoding for categorical columns "
        f"{', '.join(CATEGORICAL_HINT)} plus passthrough numeric features\n"
        "- Target: `{TARGET_COLUMN}`\n"
        "- Train/test split: 80/20 with random_state=42"
    )

    st.markdown("### Next steps")
    st.info(
        "Use the filtered dataset from the overview tab to identify segments with "
        "higher or lower relationship probabilities before retraining or tuning the model."
    )

