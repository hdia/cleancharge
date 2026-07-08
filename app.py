import os
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="CleanCharge Explorer",
    page_icon="⚡",
    layout="wide",
)

APP_TITLE = "CleanCharge"
APP_SUBTITLE = "Explorer: emissions-aware EV charging using open data"
APP_SCOPE = "Melbourne, Victoria, Australia"


st.markdown(
    """
    <style>
    .cc-card {
        padding: 1.1rem 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid #e6e6e6;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        min-height: 128px;
    }
    .cc-card-green {
        border-left: 6px solid #2E8B57;
        background: #F1FAF4;
    }
    .cc-card-red {
        border-left: 6px solid #C0392B;
        background: #FFF3F1;
    }
    .cc-card-blue {
        border-left: 6px solid #2F80ED;
        background: #F2F7FF;
    }
    .cc-card-grey {
        border-left: 6px solid #777777;
        background: #F7F7F7;
    }
    .cc-label {
        font-size: 0.82rem;
        color: #555;
        margin-bottom: 0.25rem;
    }
    .cc-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #1F2937;
        margin-bottom: 0.2rem;
    }
    .cc-small {
        font-size: 0.82rem;
        color: #555;
        line-height: 1.35;
    }
    .cc-recommendation {
        padding: 1.1rem 1.3rem;
        border-radius: 0.9rem;
        background: linear-gradient(90deg, #E6F6EC, #F5FBF7);
        border: 1px solid #C8EAD3;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

SEARCH_DIRS = [
    ROOT / "data" / "dashboard",
    ROOT / "data" / "processed",
    ROOT / "data" / "processed" / "ev_outputs",
]


def find_file(filename: str) -> Path | None:
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


@st.cache_data
def load_csv(filename: str) -> pd.DataFrame:
    p = find_file(filename)
    if p is None:
        return pd.DataFrame()
    return pd.read_csv(p)


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        name = c.lower()
        if "time" in name or "date" in name or "start" in name or "end" in name:
            converted = pd.to_datetime(out[c], errors="coerce")
            if converted.notna().sum() > 0:
                out[c] = converted

    return out

def fmt_kg(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.1f} kg CO₂"


def fmt_intensity(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.0f} gCO₂/kWh"


def fmt_aud(x):
    if pd.isna(x):
        return "n/a"
    return f"${x:.2f}"


def fmt_pct(x):
    if pd.isna(x):
        return "n/a"
    return f"{x:.0f}%"


def fmt_duration(hours_float):
    if pd.isna(hours_float):
        return "n/a"
    mins = int(round(hours_float * 60))
    h = mins // 60
    m = mins % 60
    if h == 0:
        return f"{m} min"
    if m == 0:
        return f"{h} h"
    return f"{h} h {m} min"


def metric_card(label, value, note, style="grey"):
    st.markdown(
        f"""
        <div class="cc-card cc-card-{style}">
            <div class="cc-label">{label}</div>
            <div class="cc-value">{value}</div>
            <div class="cc-small">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_forecast_col(df: pd.DataFrame) -> str | None:
    for c in ["intensity_hat", "y_hat", "predicted", "forecast"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "hat" in c.lower() or "forecast" in c.lower() or "pred" in c.lower():
            return c
    return None


def detect_time_col(df: pd.DataFrame) -> str | None:
    for c in ["local_time", "timestamp", "time", "date"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "time" in c.lower() or "date" in c.lower():
            return c
    return None


def contiguous_window(df: pd.DataFrame, time_col: str, value_col: str, window_hours: int, best=True):
    d = df[[time_col, value_col]].dropna().copy()
    d = d.sort_values(time_col).reset_index(drop=True)
    if len(d) < window_hours:
        return pd.DataFrame(), np.nan

    rolling = d[value_col].rolling(window_hours).mean()
    idx = rolling.idxmin() if best else rolling.idxmax()
    if pd.isna(idx):
        return pd.DataFrame(), np.nan

    start_idx = int(idx) - window_hours + 1
    end_idx = int(idx)
    block = d.iloc[start_idx:end_idx + 1].copy()
    return block, float(block[value_col].mean())


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
forecast = parse_datetime_columns(load_csv("intensity_forecast_next24.csv"))
backtest = parse_datetime_columns(load_csv("intensity_backtest_last7d.csv"))
system_summary = parse_datetime_columns(load_csv("system_summary.csv"))
savings_summary = parse_datetime_columns(load_csv("savings_sensitivity_summary.csv"))
savings_daily = parse_datetime_columns(load_csv("savings_sensitivity_daily.csv"))
chargers = parse_datetime_columns(load_csv("ocm_fast_chargers_melbourne.csv"))
plans = parse_datetime_columns(load_csv("clean_window_plans.csv"))
per_origin = parse_datetime_columns(load_csv("per_origin_summary_v2.csv"))


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title(APP_TITLE)
st.caption(
    f"{APP_SUBTITLE} | {APP_SCOPE}"
)

st.info(
    "CleanCharge Explorer is a research prototype for emissions-aware EV charging in Melbourne, Victoria, Australia. "
    "It uses processed open electricity market data and public charging infrastructure data to reproduce results from the CleanCharge study. "
    "It is not a live consumer recommendation system."
)

st.markdown(
    """
    **Reference:**  
    Dia, H. (2026). *CleanCharge: Emissions-aware electric vehicle charging and infrastructure equity with open data in Melbourne.*  
    *International Journal of Sustainable Transportation* (2026).  
    DOI: https://doi.org/10.1080/15568318.2026.2693676
    """
)

with st.expander("Data sources and scope"):
    st.markdown(
        """
        **Study location:** Melbourne, Victoria, Australia  

        **Electricity data:** OpenElectricity / OpenNEM (Victoria region, NEM system data)  

        **Charging infrastructure:** Open Charge Map  

        **Purpose:**  
        This dashboard reproduces and explores the CleanCharge research workflow using archived datasets.  
        It is intended for research and demonstration purposes only and does not provide live operational advice.
        """
    )    

available_files = {
    "Forecast": not forecast.empty,
    "Backtest": not backtest.empty,
    "System summary": not system_summary.empty,
    "Savings summary": not savings_summary.empty,
    "Savings daily": not savings_daily.empty,
    "Chargers": not chargers.empty,
    "Clean-window plans": not plans.empty,
    "Per-origin summary v2": not per_origin.empty,
}

with st.expander("Loaded data files"):
    st.write(pd.DataFrame(
        [{"File": k, "Loaded": "Yes" if v else "No"} for k, v in available_files.items()]
    ))


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Charging session")

need_kwh = st.sidebar.selectbox(
    "Energy required",
    [10.0, 20.0, 40.0],
    index=1,
    format_func=lambda x: f"{int(x)} kWh",
)

charger_kw = st.sidebar.selectbox(
    "Charger power",
    [7.0, 11.0, 22.0, 50.0],
    index=0,
    format_func=lambda x: f"{int(x)} kW",
)

duration_exact = need_kwh / charger_kw
window_hours = max(1, int(np.ceil(duration_exact)))

st.sidebar.metric(
    "Approximate charging duration",
    fmt_duration(duration_exact),
    help="The forecast window is rounded up to the nearest full hour for the dashboard calculation.",
)


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Carbon intensity forecast",
        "Charging scenarios",
        "Charging infrastructure",
        "Emissions and cost sensitivity",
        "Accessibility and equity",
    ]
)


# ------------------------------------------------------------
# Tab 1: Forecast
# ------------------------------------------------------------
with tab1:
    st.subheader("24-hour carbon intensity forecast for Melbourne electricity supply")

    if forecast.empty:
        st.warning("Missing intensity_forecast_next24.csv")
    else:
        time_col = detect_time_col(forecast)
        value_col = detect_forecast_col(forecast)

        if time_col is None or value_col is None:
            st.error("Could not detect time or forecast intensity column.")
            st.write(forecast.head())
        else:
            forecast[time_col] = pd.to_datetime(forecast[time_col], errors="coerce")
            d = forecast.dropna(subset=[time_col, value_col]).copy()

            best_block, best_intensity = contiguous_window(
                d, time_col, value_col, window_hours=window_hours, best=True
            )
            worst_block, worst_intensity = contiguous_window(
                d, time_col, value_col, window_hours=window_hours, best=False
            )

            best_kg = best_intensity * need_kwh / 1000 if not pd.isna(best_intensity) else np.nan
            worst_kg = worst_intensity * need_kwh / 1000 if not pd.isna(worst_intensity) else np.nan
            saving_kg = worst_kg - best_kg if not pd.isna(best_kg) and not pd.isna(worst_kg) else np.nan

            saving_pct = (saving_kg / worst_kg * 100) if not pd.isna(saving_kg) and worst_kg > 0 else np.nan

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                metric_card(
                    "Lowest forecast carbon intensity",
                    fmt_intensity(best_intensity),
                    "Average intensity during the cleanest charging period.",
                    "green",
                )

            with c2:
                metric_card(
                    "Highest forecast carbon intensity",
                    fmt_intensity(worst_intensity),
                    "Average intensity during the highest-intensity charging period.",
                    "red",
                )

            with c3:
                metric_card(
                    "Lowest-window emissions",
                    fmt_kg(best_kg),
                    f"Estimated emissions for a {int(need_kwh)} kWh charging session.",
                    "blue",
                )

            with c4:
                metric_card(                    
                    "Emissions savings",
                    f"{fmt_kg(saving_kg)}",
                    f"Equivalent to {fmt_pct(saving_pct)} lower emissions than the highest-intensity window.",
                    "green",
                )

            if not best_block.empty:
                start = best_block[time_col].iloc[0]
                end = best_block[time_col].iloc[-1] + pd.Timedelta(hours=1)

                st.markdown(
                    f"""
                    <div class="cc-recommendation">
                        <div class="cc-label">Recommended clean charging window</div>
                        <div class="cc-value">{start.strftime('%a %d %b, %I:%M %p')} to {end.strftime('%I:%M %p')}</div>
                        <div class="cc-small">
                            For the selected {int(need_kwh)} kWh session using a {int(charger_kw)} kW charger in Melbourne.
                            This is the lowest forecast carbon-intensity window in Melbourne for the selected charging scenario.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=d[time_col],
                y=d[value_col],
                mode="lines+markers",
                name="Forecast carbon intensity",
                line=dict(color="#2E8B57", width=3),
                marker=dict(size=6),
            ))

            if not best_block.empty:
                fig.add_vrect(
                    x0=best_block[time_col].iloc[0],
                    x1=best_block[time_col].iloc[-1] + pd.Timedelta(hours=1),
                    fillcolor="green",
                    opacity=0.18,
                    line_width=0,
                    annotation_text="Cleanest window",
                    annotation_position="top left",
                )

            if not worst_block.empty:
                fig.add_vrect(
                    x0=worst_block[time_col].iloc[0],
                    x1=worst_block[time_col].iloc[-1] + pd.Timedelta(hours=1),
                    fillcolor="red",
                    opacity=0.12,
                    line_width=0,
                    annotation_text="Highest-intensity window",
                    annotation_position="top right",
                )

            fig.update_layout(
                title="Forecast carbon intensity for Melbourne's electricity supply",
                xaxis_title="Time",
                yaxis_title="Carbon intensity (gCO₂/kWh)",
                height=480,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Backtest validation")

    if backtest.empty:
        st.warning("Missing intensity_backtest_last7d.csv")
    else:
        tcol = detect_time_col(backtest)
        if tcol and {"y_true", "y_hat"}.issubset(backtest.columns):
            backtest[tcol] = pd.to_datetime(backtest[tcol], errors="coerce")
            bt = backtest.dropna(subset=[tcol, "y_true", "y_hat"]).copy()

            mae = np.mean(np.abs(bt["y_true"] - bt["y_hat"]))
            smape = np.mean(
                np.abs(bt["y_true"] - bt["y_hat"]) /
                ((np.abs(bt["y_true"]) + np.abs(bt["y_hat"])) / 2)
            ) * 100

            c1, c2 = st.columns(2)
            c1.metric("Validation MAE", f"{mae:.1f} gCO₂/kWh")
            c2.metric("Validation sMAPE", f"{smape:.1f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt[tcol], y=bt["y_true"], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(x=bt[tcol], y=bt["y_hat"], mode="lines", name="Predicted"))
            fig.update_layout(
                title="Backtest validation over held-out period",
                xaxis_title="Time",
                yaxis_title="Carbon intensity (gCO₂/kWh)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(backtest.head())


# ------------------------------------------------------------
# Tab 2: Charging calculator
# ------------------------------------------------------------
with tab2:
    st.subheader("Cost-optimal vs emissions-optimal charging comparison")
    st.caption(
    "This section compares charging outcomes under different optimisation rules: electricity price minimisation vs carbon intensity minimisation."
    )

    if system_summary.empty:
        st.warning("Missing system_summary.csv")
    else:
        price_options = sorted(system_summary["price_basis"].dropna().unique().tolist())
        price_basis = st.selectbox("Price basis", price_options, index=0)

        sub = system_summary[
            (system_summary["price_basis"] == price_basis)
            & (system_summary["need_kwh"] == need_kwh)
            & (system_summary["charger_kw"] == charger_kw)
        ].copy()

        if sub.empty:
            st.warning("No matching scenario in system_summary.csv. Try another energy or charger power.")
            st.dataframe(system_summary.head(20), use_container_width=True)
        else:
            row = sub.iloc[0]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cleanest-window emissions", fmt_kg(row.get("best_intensity_emissions_kg")))
            c2.metric("Cheapest-window emissions", fmt_kg(row.get("best_price_emissions_kg")))
            c3.metric("Emissions saving", fmt_kg(row.get("emissions_delta_kg")))
            c4.metric("Cost difference", fmt_aud(abs(row.get("cost_delta_AUD", np.nan))))

            compare = pd.DataFrame({
                "Window": ["Cheapest", "Cleanest"],
                "Emissions (kg CO₂)": [
                    row.get("best_price_emissions_kg"),
                    row.get("best_intensity_emissions_kg"),
                ],
                "Cost (AUD)": [
                    row.get("best_price_cost_AUD"),
                    row.get("best_intensity_cost_AUD"),
                ],
            })

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    compare,
                    x="Window",
                    y="Emissions (kg CO₂)",
                    title="Emissions comparison",
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                fig = px.bar(
                    compare,
                    x="Window",
                    y="Cost (AUD)",
                    title="Cost comparison",
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Scenario details"):
                st.dataframe(sub, use_container_width=True)


# ------------------------------------------------------------
# Tab 3: Charger map
# ------------------------------------------------------------
with tab3:
    st.subheader("Melbourne public fast-charging sites")

    if chargers.empty:
        st.warning("Missing ocm_fast_chargers_melbourne.csv")
    else:
        min_kw = st.slider("Minimum charger power shown", 0, 350, 50, step=10)

        m = chargers.copy()
        if "power_kw" in m.columns:
            m["power_kw"] = pd.to_numeric(m["power_kw"], errors="coerce")
            m = m[m["power_kw"].fillna(0) >= min_kw]

        if {"lat", "lon"}.issubset(m.columns):
            st.metric("Sites shown", len(m))
            st.map(m[["lat", "lon"]].dropna(), zoom=9)

            cols = [c for c in ["name", "operator", "power_kw", "connectors", "lat", "lon"] if c in m.columns]
            st.dataframe(m[cols].sort_values("power_kw", ascending=False), use_container_width=True)
        else:
            st.error("Could not find lat/lon columns.")
            st.dataframe(chargers.head(), use_container_width=True)


# ------------------------------------------------------------
# Tab 4: Savings sensitivity
# ------------------------------------------------------------
with tab4:
    st.subheader("Theoretical upper-bound emissions savings")

    if savings_summary.empty:
        st.warning("Missing savings_sensitivity_summary.csv")
    else:
        s = savings_summary.copy()
        s["scenario"] = s["need_kwh"].astype(int).astype(str) + " kWh @ " + s["charger_kw"].astype(int).astype(str) + " kW"

        fig = px.bar(
            s,
            x="scenario",
            y="saving_kg_mean",
            title="Mean theoretical daily saving by charging scenario",
            labels={
                "scenario": "Charging scenario",
                "saving_kg_mean": "Mean saving (kg CO₂)",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(s, use_container_width=True)

    if not savings_daily.empty:
        with st.expander("Daily savings distribution"):
            d = savings_daily.copy()
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            fig = px.box(
                d,
                x="need_kwh",
                y="saving_kg",
                color="charger_kw",
                title="Distribution of daily theoretical savings",
                labels={
                    "need_kwh": "Energy required (kWh)",
                    "saving_kg": "Saving (kg CO₂)",
                    "charger_kw": "Charger power (kW)",
                },
            )
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------
# Tab 5: Origins and equity
# ------------------------------------------------------------
with tab5:
    st.subheader("Origin-level charging outcomes")

    if per_origin.empty:
        st.warning("Missing per_origin_summary.csv")
    else:
        po = per_origin.copy()

        if "price_basis" in po.columns:
            price_options = sorted(po["price_basis"].dropna().unique().tolist())
            pb = st.selectbox("Origin analysis price basis", price_options, index=0, key="origin_price")
            po = po[po["price_basis"] == pb]

        if "need_kwh" in po.columns:
            po = po[po["need_kwh"] == need_kwh]

        if "charger_kw" in po.columns:
            po = po[po["charger_kw"] == charger_kw]

        if po.empty:
            st.warning("No matching origin-level scenario.")
        else:
            if {"name", "emissions_delta_kg"}.issubset(po.columns):
                fig = px.bar(
                    po.sort_values("emissions_delta_kg", ascending=False),
                    x="name",
                    y="emissions_delta_kg",
                    title="Origin-level emissions savings",
                    labels={
                        "name": "Origin",
                        "emissions_delta_kg": "Saving (kg CO₂)",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
                                
                st.caption(
                    "Origin-level differences reflect travel time to charging sites under the selected scenario. "
                    "No additional travel emissions are included. Variations in results are therefore driven by differences "
                    "in effective charging duration rather than differences in electricity carbon intensity."
                )

            st.dataframe(po, use_container_width=True)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.divider()

st.caption(
    "CleanCharge Explorer v1.0 is a research prototype for Melbourne, Victoria, Australia. "
    "It reproduces results from Dia (2026) using processed OpenElectricity/OpenNEM data and Open Charge Map infrastructure data. "
    "It is intended for research and demonstration purposes only."
)