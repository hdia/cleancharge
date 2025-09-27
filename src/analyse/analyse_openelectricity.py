#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_openelectricity.py
--------------------------

Reads the six CSVs produced by the fetcher and generates:
1) 90-day analyses (emissions, energy, intensity)
2) 15-day market analyses (price, demand, intensity)
3) Hybrid overview with clear 'has_market' coverage
4) Clean KPI tables and charts

Expected input files (default locations):
- data/processed/openelectricity_90d_daily.csv
- data/processed/openelectricity_90d_daily_local.csv
- data/processed/openelectricity_15d_hourly.csv
- data/processed/openelectricity_15d_hourly_local.csv
- data/processed/openelectricity_90d_hybrid.csv
- data/processed/openelectricity_90d_hybrid_local.csv

Outputs (default to data/analysis):
- kpis_90d.csv
- kpis_15d.csv
- negative_price_windows.csv
- plots/*.png

Usage:
    python src/analyze_openelectricity.py
    python src/analyze_openelectricity.py --in-dir data/processed --out-dir data/analysis
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.labelsize": 10,   # axis labels
    "xtick.labelsize": 8,   # x tick labels
    "ytick.labelsize": 8,   # y tick labels
    "legend.fontsize": 9,   # legend
    "figure.titlesize": 12, # titles
})

# ---------------------------
# I/O helpers
# ---------------------------
def read_csv_required(path: str, parse_dates: List[str] = ["timestamp"]) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"!! Missing file: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    # Normalize timestamp column name if user uploaded a variant
    if "timestamp" not in df.columns:
        # Sometimes local files lead with 'local_time' first — we still want UTC 'timestamp' for joins
        # If absent, try to infer from first datetime-like column, otherwise bail.
        dt_cols = [c for c in df.columns if "time" in c or "timestamp" in c]
        if dt_cols:
            # keep original, create a naive UTC timestamp as fallback for sorting
            df["timestamp"] = pd.to_datetime(df[dt_cols[-1]], errors="coerce", utc=True)
        else:
            print(f"!! No 'timestamp' column in {path} and could not infer one.")
            sys.exit(1)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Cleaning / types
# ---------------------------
def as_float(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clip_outliers(series: pd.Series, q_low=0.001, q_high=0.999) -> pd.Series:
    # Gentle clip for plotting robustness
    lo, hi = series.quantile(q_low), series.quantile(q_high)
    return series.clip(lower=lo, upper=hi)


# ---------------------------
# Computations
# ---------------------------
def summarize_90d(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs over the 90-day daily dataset.
    """
    cols = ["emissions_t", "energy_mwh", "intensity_g_per_kwh"]
    df = as_float(df_daily, cols).dropna(subset=["timestamp"]).copy()

    # Simple summaries
    kpis = {
        "days_covered": df["timestamp"].dt.normalize().nunique(),
        "start_utc": df["timestamp"].min(),
        "end_utc": df["timestamp"].max(),
        "emissions_t_total": df["emissions_t"].sum(skipna=True),
        "emissions_t_avg_per_day": df["emissions_t"].mean(skipna=True),
        "energy_mwh_total": df["energy_mwh"].sum(skipna=True),
        "energy_mwh_avg_per_day": df["energy_mwh"].mean(skipna=True),
        "intensity_g_per_kwh_avg": df["intensity_g_per_kwh"].mean(skipna=True),
        "intensity_g_per_kwh_min": df["intensity_g_per_kwh"].min(skipna=True),
        "intensity_g_per_kwh_max": df["intensity_g_per_kwh"].max(skipna=True),
    }
    return pd.DataFrame([kpis])


def summarize_15d_market(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    KPIs for the 15-day hourly dataset (market metrics available).
    """
    cols = ["emissions_t", "energy_mwh", "intensity_g_per_kwh", "spot_price_per_mwh", "demand_mw", "spot_c_per_kwh"]
    df = as_float(df_hourly, cols).dropna(subset=["timestamp"]).copy()

    # Count negative price hours (after coercion)
    neg_hours = (df["spot_price_per_mwh"] < 0).sum(skipna=True) if "spot_price_per_mwh" in df else np.nan
    kpis = {
        "hours_covered": df["timestamp"].dt.floor("h").nunique(),
        "start_utc": df["timestamp"].min(),
        "end_utc": df["timestamp"].max(),
        "emissions_t_total": df["emissions_t"].sum(skipna=True),
        "energy_mwh_total": df["energy_mwh"].sum(skipna=True),
        "intensity_g_per_kwh_avg": df["intensity_g_per_kwh"].mean(skipna=True),
        "demand_mw_avg": df["demand_mw"].mean(skipna=True) if "demand_mw" in df else np.nan,
        "spot_price_per_mwh_avg": df["spot_price_per_mwh"].mean(skipna=True) if "spot_price_per_mwh" in df else np.nan,
        "spot_price_per_mwh_min": df["spot_price_per_mwh"].min(skipna=True) if "spot_price_per_mwh" in df else np.nan,
        "spot_price_per_mwh_max": df["spot_price_per_mwh"].max(skipna=True) if "spot_price_per_mwh" in df else np.nan,
        "negative_price_hours": neg_hours,
    }
    return pd.DataFrame([kpis])


def find_negative_price_windows(df_hourly: pd.DataFrame, min_len: int = 2) -> pd.DataFrame:
    """
    Identify contiguous windows of negative spot price (>= min_len hours).
    """
    if "spot_price_per_mwh" not in df_hourly.columns:
        return pd.DataFrame(columns=["start_utc", "end_utc", "hours", "min_price_per_mwh"])

    df = df_hourly[["timestamp", "spot_price_per_mwh"]].dropna().copy()
    df["neg"] = df["spot_price_per_mwh"] < 0
    if df["neg"].sum() == 0:
        return pd.DataFrame(columns=["start_utc", "end_utc", "hours", "min_price_per_mwh"])

    # group by consecutive negative runs
    df["grp"] = (df["neg"] != df["neg"].shift()).cumsum()
    runs = []
    for g, block in df.groupby("grp"):
        if block["neg"].iloc[0] and len(block) >= min_len:
            runs.append({
                "start_utc": block["timestamp"].iloc[0],
                "end_utc": block["timestamp"].iloc[-1],
                "hours": len(block),
                "min_price_per_mwh": block["spot_price_per_mwh"].min()
            })
    return pd.DataFrame(runs)


# ---------------------------
# Plots
# ---------------------------

def plot_series(df: pd.DataFrame, y: str, title: str, out_path: str, smooth: bool = False) -> None:
    if y not in df.columns or df.empty:
        return
    series = df[["timestamp", y]].dropna().copy()
    if series.empty:
        return

    plt.figure(figsize=(12, 4))
    y_plot = (series[y].rolling(7, min_periods=1).mean()
              if smooth else series[y])
    plt.plot(series["timestamp"], y_plot,
             color="red", linestyle="--", linewidth=1.8, label="historical")

    plt.title(title, fontsize=10, pad=8)
    plt.xlabel("Date–Time")
    # friendlier y-labels for the three daily plots
    if y == "emissions_t":
        plt.ylabel("Emissions (tCO₂/day)")
    elif y == "energy_mwh":
        plt.ylabel("Energy (MWh/day)")
    elif y == "intensity_g_per_kwh":
        plt.ylabel("Carbon intensity (gCO₂/kWh)")
    else:
        plt.ylabel(y)

    plt.grid(True, alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_scatter(df: pd.DataFrame, x: str, y: str, c: str, title: str, out_path: str) -> None:
    needed = [x, y] + ([c] if c else [])
    if any(col not in df.columns for col in needed):
        return
    dat = df[needed].dropna().copy()
    if dat.empty:
        return

    plt.figure(figsize=(7, 6))
    if c:
        sc = plt.scatter(dat[x], dat[y],
                         c=dat[c], s=20, alpha=0.65, cmap="viridis", edgecolors="none")
        cb = plt.colorbar(sc)
        cb.set_label("Carbon intensity (gCO₂/kWh)", fontsize=10)
    else:
        plt.scatter(dat[x], dat[y], s=20, alpha=0.65, edgecolors="none")

    plt.title(title, fontsize=10, pad=8)
    # nicer labels
    if x == "demand_mw": plt.xlabel("Demand (MW)")
    else: plt.xlabel(x)
    if y == "spot_price_per_mwh": plt.ylabel("Wholesale price ($/MWh)")
    else: plt.ylabel(y)

    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_tod_profiles(df_hourly: pd.DataFrame, cols: List[str], title: str, out_path: str) -> None:
    if df_hourly.empty:
        return
    dat = df_hourly.copy()
    dat["hour"] = dat["timestamp"].dt.hour
    means = dat.groupby("hour")[cols].mean(numeric_only=True)
    if means.empty:
        return

    # Min–max normalise each series for shape comparison
    mm = means.copy()
    for c in mm.columns:
        v = mm[c].values.astype("float64")
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        mm[c] = (v - lo) / (hi - lo) if hi > lo else 0.0

    plt.figure(figsize=(8.5, 4))
    for c in mm.columns:
        label = {
            "spot_price_per_mwh": "Price (norm.)",
            "demand_mw": "Demand (norm.)",
            "intensity_g_per_kwh": "Intensity (norm.)"
        }.get(c, f"{c} (norm.)")

        style = {"intensity_g_per_kwh": ("red", "--"),
                 "demand_mw": ("green", "-"),
                 "spot_price_per_mwh": ("black", "-")}.get(c, ("blue", "-"))

        plt.plot(mm.index, mm[c], label=label,
                 color=style[0], linestyle=style[1], linewidth=1.8)

    plt.title("Last 15 days (hourly): average diurnal profiles (normalised to [0–1])", fontsize=10, pad=8)
    plt.xlabel("Hour of day (UTC)")
    plt.ylabel("Normalised index [0–1]")
    plt.grid(True, alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", default="data/processed", help="Directory containing the six CSVs")
    p.add_argument("--out-dir", default="data/analysis", help="Directory to write analysis outputs")
    args = p.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    figs_dir = os.path.join(out_dir, "plots")
    ensure_dir(out_dir)
    ensure_dir(figs_dir)

    # Load all six (UTC variants are the primary inputs for analysis)
    daily_path = os.path.join(in_dir, "openelectricity_90d_daily.csv")
    hourly_path = os.path.join(in_dir, "openelectricity_15d_hourly.csv")
    hybrid_path = os.path.join(in_dir, "openelectricity_90d_hybrid.csv")

    df_daily = read_csv_required(daily_path)
    df_hourly = read_csv_required(hourly_path)
    df_hybrid = read_csv_required(hybrid_path)

    # Ensure expected numeric columns exist (some may be absent in daily)
    for df in (df_daily, df_hourly, df_hybrid):
        for c in ["emissions_t", "energy_mwh", "intensity_g_per_kwh",
                  "spot_price_per_mwh", "spot_c_per_kwh",
                  "retailA_c_per_kwh", "retailB_c_per_kwh", "demand_mw"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------------- 90-day analyses (daily) ----------------
    kpis_90 = summarize_90d(df_daily)
    kpis_90.to_csv(os.path.join(out_dir, "kpis_90d.csv"), index=False)

    # Plots: emissions, energy, intensity (90d, daily)
    plot_series(df_daily, "emissions_t", "Grid emissions — 90-day daily totals (tCO₂/day)", os.path.join(figs_dir, "daily_emissions.png"))
    plot_series(df_daily, "energy_mwh", "Electricity supplied — 90-day daily energy (MWh/day)", os.path.join(figs_dir, "daily_energy.png"))
    plot_series(df_daily, "intensity_g_per_kwh", "Carbon intensity — 90-day daily mean (gCO₂/kWh)", os.path.join(figs_dir, "daily_intensity.png"), smooth=True)

    # ---------------- 15-day market analyses (hourly) ----------------
    # Filter to rows that actually have market coverage if the file is bigger than expected
    has_market = "has_market" in df_hourly.columns
    if has_market:
        df_hourly_mkt = df_hourly[df_hourly["has_market"] == True].copy()
        if df_hourly_mkt.empty:
            df_hourly_mkt = df_hourly.copy()
    else:
        df_hourly_mkt = df_hourly.copy()

    kpis_15 = summarize_15d_market(df_hourly_mkt)
    kpis_15.to_csv(os.path.join(out_dir, "kpis_15d.csv"), index=False)

    # Scatter: demand vs price colored by intensity
    plot_scatter(
        df_hourly_mkt,
        x="demand_mw",
        y="spot_price_per_mwh",
        c="intensity_g_per_kwh",
        title="Last 15 days (hourly): wholesale price vs demand (colour = carbon intensity)",
        out_path=os.path.join(figs_dir, "hourly_price_vs_demand.png"),
    )

    # Time-of-day profiles (hourly means)
    plot_tod_profiles(
        df_hourly_mkt,
        cols=["spot_price_per_mwh", "demand_mw", "intensity_g_per_kwh"],
        title="15d hourly averages by hour-of-day (UTC)",
        out_path=os.path.join(figs_dir, "hourly_tod_profiles.png"),
    )

    # Negative price windows
    neg_runs = find_negative_price_windows(df_hourly_mkt, min_len=2)
    neg_runs.to_csv(os.path.join(out_dir, "negative_price_windows.csv"), index=False)

    # ---------------- Hybrid overview ----------------
    # Coverage chart: show which rows have/ lack market data

    if "has_market" in df_hybrid.columns:
        plot_series(
            df_hybrid.assign(has_market_num=df_hybrid["has_market"].astype(float)),
            "has_market_num",
            "Hybrid coverage: 1=market data present, 0=absent",
            os.path.join(figs_dir, "hybrid_has_market.png")
        )

    # Save a slimmed hybrid preview for quick inspection
    keep = ["timestamp", "emissions_t", "energy_mwh", "intensity_g_per_kwh",
            "spot_price_per_mwh", "demand_mw", "has_market"]
    slim = df_hybrid[[c for c in keep if c in df_hybrid.columns]].copy()
    slim.to_csv(os.path.join(out_dir, "hybrid_preview.csv"), index=False)

    # ---------------- Console summary ----------------
    print("\n=== SUMMARY ===")
    print("90d KPIs:")
    print(kpis_90.to_string(index=False))
    print("\n15d Market KPIs:")
    print(kpis_15.to_string(index=False))
    if not neg_runs.empty:
        print(f"\nNegative price windows found: {len(neg_runs)} (see negative_price_windows.csv)")
    else:
        print("\nNo multi-hour negative price windows detected.")

    print(f"\nOutputs written to: {out_dir}")
    print("Plots saved under:", os.path.join(out_dir, "plots"))

if __name__ == "__main__":
    main()
