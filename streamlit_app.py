"""
streamlit_app.py
----------------
Skoda Used Car Price Predictor — interactive Streamlit front-end.

Run with:
    streamlit run streamlit_app.py
"""

import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Škoda Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Skoda brand colours
# ---------------------------------------------------------------------------
SKODA_GREEN       = "#4EA94B"
SKODA_DARK_GREEN  = "#2C5F2D"
SKODA_LIGHT_GREEN = "#A8D5A2"
GREEN_PALETTE     = [SKODA_DARK_GREEN, SKODA_GREEN, SKODA_LIGHT_GREEN,
                     "#6BBF6A", "#3A7D3A", "#8DC98B"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* Main header bar */
    .hero-banner {{
        background: linear-gradient(135deg, {SKODA_DARK_GREEN} 0%, {SKODA_GREEN} 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }}
    .hero-banner h1 {{ color: white; font-size: 2.4rem; margin: 0; }}
    .hero-banner p  {{ color: #e0f0df; font-size: 1.05rem; margin: 0.4rem 0 0 0; }}

    /* Metric cards */
    [data-testid="metric-container"] {{
        background: #f8fdf8;
        border: 1px solid {SKODA_LIGHT_GREEN};
        border-radius: 10px;
        padding: 1rem;
    }}
    [data-testid="stMetricValue"] {{ color: {SKODA_DARK_GREEN}; font-size: 2rem !important; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: #f0f7f0;
    }}
    .sidebar-title {{
        font-weight: 700;
        color: {SKODA_DARK_GREEN};
        font-size: 1.1rem;
        border-bottom: 2px solid {SKODA_GREEN};
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #ddd;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data / model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_feature_ranges():
    with open("feature_ranges.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_data():
    df = pd.read_csv("data/skoda.csv")
    df = df.dropna()
    df = df[df["price"].between(500, 60_000)]
    df = df[df["mileage"] <= 200_000]
    df = df[df["year"] >= 2000]
    df["age"] = 2024 - df["year"]
    return df


model    = load_model()
encoders = load_encoders()
ranges   = load_feature_ranges()
df_full  = load_data()


# ---------------------------------------------------------------------------
# Hero banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🚗 Škoda Used Car Price Predictor</h1>
        <p>ML-powered pricing tool for the UK used car market &nbsp;·&nbsp;
           Gradient Boosting · scikit-learn · ~5 000 real-world listings</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="sidebar-title">🔧 Configure Your Car</p>', unsafe_allow_html=True)

    selected_model = st.selectbox(
        "Model",
        options=ranges["models"],
        index=ranges["models"].index("Octavia") if "Octavia" in ranges["models"] else 0,
    )

    year_min, year_max = ranges["year_range"]
    selected_year = st.slider("Year", min_value=year_min, max_value=year_max, value=2018)

    mile_max = ranges["mileage_range"][1]
    selected_mileage = st.slider(
        "Mileage (miles)", min_value=0, max_value=mile_max, value=30_000, step=1_000,
        format="%d mi"
    )

    selected_transmission = st.selectbox("Transmission", options=ranges["transmissions"])
    selected_fuel         = st.selectbox("Fuel Type",     options=ranges["fuel_types"])
    selected_engine       = st.selectbox(
        "Engine Size (L)",
        options=[str(e) for e in ranges["engine_sizes"]],
        index=min(2, len(ranges["engine_sizes"]) - 1),
    )

    # Auto-fill tax & mpg from medians
    auto_tax = ranges["tax_median_by_model"].get(selected_model, 145)
    auto_mpg = ranges["mpg_median_by_model"].get(selected_model, 48.0)

    st.markdown("---")
    st.markdown("**Auto-filled from model averages:**")
    col_t, col_m = st.columns(2)
    col_t.metric("Tax (£/yr)", f"£{int(auto_tax)}")
    col_m.metric("MPG", f"{auto_mpg:.1f}")

    st.markdown("---")
    st.caption("Adjust inputs above and the prediction updates instantly.")


# ---------------------------------------------------------------------------
# Prepare inputs & predict
# ---------------------------------------------------------------------------
age            = 2024 - selected_year
model_enc      = encoders["model"].transform([selected_model])[0]
trans_enc      = encoders["transmission"].transform([selected_transmission])[0]
fuel_enc       = encoders["fuelType"].transform([selected_fuel])[0]
engine_size    = float(selected_engine)

X_input = np.array([[age, selected_mileage, engine_size,
                      auto_tax, auto_mpg,
                      model_enc, trans_enc, fuel_enc]])

predicted_price = float(model.predict(X_input)[0])

# Market stats for selected model
df_model = df_full[df_full["model"] == selected_model]
market_avg  = df_model["price"].mean()
sample_size = len(df_model)
delta_pct   = ((predicted_price - market_avg) / market_avg) * 100


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="💰 Predicted Price",
        value=f"£{predicted_price:,.0f}",
    )

with c2:
    st.metric(
        label="📊 Market Average",
        value=f"£{market_avg:,.0f}",
        delta=f"{delta_pct:+.1f}% vs avg",
    )

with c3:
    st.metric(
        label="📈 Sample Size",
        value=f"{sample_size:,} cars",
        delta=f"{selected_model} listings",
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Price Analysis", "🔍 Model Comparison", "📊 Feature Importance", "📋 Raw Data"]
)


# ── Tab 1: Price Analysis ────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        fig_scatter = px.scatter(
            df_model,
            x="mileage",
            y="price",
            color="year",
            title=f"Mileage vs Price — {selected_model}",
            labels={"mileage": "Mileage (miles)", "price": "Price (£)", "year": "Year"},
            color_continuous_scale="Greens",
            template="plotly_white",
            opacity=0.7,
        )
        fig_scatter.add_hline(
            y=predicted_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prediction: £{predicted_price:,.0f}",
            annotation_position="top left",
        )
        fig_scatter.update_traces(marker=dict(size=6))
        fig_scatter.update_layout(height=400, coloraxis_colorbar=dict(title="Year"))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            df_model,
            x="price",
            nbins=40,
            title=f"Price Distribution — {selected_model}",
            labels={"price": "Price (£)"},
            color_discrete_sequence=[SKODA_GREEN],
            template="plotly_white",
        )
        fig_hist.add_vline(
            x=predicted_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"£{predicted_price:,.0f}",
            annotation_position="top right",
        )
        fig_hist.update_layout(height=400, showlegend=False,
                                yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)


# ── Tab 2: Model Comparison ───────────────────────────────────────────────────
with tab2:
    avg_by_model = (
        df_full.groupby("model")["price"]
        .mean()
        .reset_index()
        .sort_values("price", ascending=True)
    )
    avg_by_model["colour"] = avg_by_model["model"].apply(
        lambda m: SKODA_DARK_GREEN if m == selected_model else SKODA_GREEN
    )

    fig_bar = go.Figure(
        go.Bar(
            x=avg_by_model["price"],
            y=avg_by_model["model"],
            orientation="h",
            marker_color=avg_by_model["colour"],
            text=avg_by_model["price"].apply(lambda p: f"£{p:,.0f}"),
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="Average Price by Škoda Model",
        xaxis_title="Average Price (£)",
        yaxis_title="",
        template="plotly_white",
        height=500,
        xaxis=dict(tickprefix="£"),
        margin=dict(l=20, r=80, t=50, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(f"Highlighted in dark green: **{selected_model}**")


# ── Tab 3: Feature Importance ─────────────────────────────────────────────────
with tab3:
    FEATURE_LABELS = {
        "age":              "Car Age (years)",
        "mileage":          "Mileage",
        "engineSize":       "Engine Size",
        "tax":              "Annual Tax",
        "mpg":              "Fuel Efficiency (MPG)",
        "model_enc":        "Model",
        "transmission_enc": "Transmission",
        "fuelType_enc":     "Fuel Type",
    }
    feature_names = ["age", "mileage", "engineSize", "tax", "mpg",
                     "model_enc", "transmission_enc", "fuelType_enc"]
    importances = model.feature_importances_

    fi_df = pd.DataFrame(
        {"feature": [FEATURE_LABELS[f] for f in feature_names],
         "importance": importances}
    ).sort_values("importance", ascending=True)

    fig_fi = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (GradientBoostingRegressor)",
        labels={"importance": "Importance", "feature": ""},
        color="importance",
        color_continuous_scale="Greens",
        template="plotly_white",
    )
    fig_fi.update_layout(
        height=420,
        coloraxis_showscale=False,
        xaxis_tickformat=".2f",
        margin=dict(l=20, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.info(
        "Feature importance measures how much each feature contributes to "
        "reducing prediction error across all 300 trees."
    )


# ── Tab 4: Raw Data ──────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Listings for **{selected_model}** ({sample_size:,} rows)")

    display_df = df_model.reset_index(drop=True).sort_values("price", ascending=False)
    st.dataframe(
        display_df.style.format({
            "price":      "£{:,.0f}",
            "mileage":    "{:,} mi",
            "mpg":        "{:.1f}",
            "engineSize": "{:.1f} L",
        }),
        height=420,
        use_container_width=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"skoda_{selected_model.lower()}_listings.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        Built with <strong>Streamlit</strong> &amp; <strong>scikit-learn</strong>
        &nbsp;·&nbsp; Data: Kaggle UK Used Cars Dataset
        &nbsp;·&nbsp;
        <a href="https://github.com/" target="_blank">GitHub</a>
        &nbsp;·&nbsp;
        <a href="https://www.kaggle.com/" target="_blank">Kaggle Notebook</a>
    </div>
    """,
    unsafe_allow_html=True,
)
