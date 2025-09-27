"""
python src/zzev_charging_analyser.py --timeseries_csv data/processed/openelectricity_90d_hybrid_local_with_intensity.csv --landmarks_csv data/processed/landmarks.csv --outdir data/processed/ev_outputs --need_kwh "10,20,40" --charger_kw "7,11,50" --price_cols "spot_c_per_kwh,retailA_c_per_kwh,retailB_c_per_kwh" --intensity_col "intensity_g_per_kwh" --time_col "local_time" --scatter_price_col "spot_c_per_kwh" --color_clean "green" --color_cheap "red" --font_size 11 --dpi 150
"""

import argparse
import os
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import modf

def parse_args():
    p = argparse.ArgumentParser(description="EV charging cost and emissions analyser over rolling windows.")
    p.add_argument("--timeseries_csv", type=str, required=True, help="Path to 90-day CSV with intensity and price columns.")
    p.add_argument("--landmarks_csv", type=str, required=False, default="", help="Optional landmarks CSV with 'name' column for per-origin replication.")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write outputs.")
    p.add_argument("--need_kwh", type=str, default="10,20,40", help="Comma-separated kWh amounts, e.g. '10,20,40'.")
    p.add_argument("--charger_kw", type=str, default="7,11,50", help="Comma-separated charger powers, e.g. '7,11,50'.")
    p.add_argument("--price_cols", type=str, default="spot_c_per_kwh,retailA_c_per_kwh,retailB_c_per_kwh", help="Comma-separated price column names.")
    p.add_argument("--intensity_col", type=str, default="intensity_g_per_kwh", help="Column to use for intensity in gCO2/kWh (fallback to 'intensity' if NaN).")
    p.add_argument("--time_col", type=str, default="local_time", help="Timestamp column, e.g. 'local_time'.")
    p.add_argument("--scatter_price_col", type=str, default="spot_c_per_kwh", help="Which price column to use for the scatter vs intensity.")
    p.add_argument("--color_clean", type=str, default="green", help="Matplotlib color name for 'cleanest' bars/points.")
    p.add_argument("--color_cheap", type=str, default="red", help="Matplotlib color name for 'cheapest' bars/points.")
    p.add_argument("--font_size", type=int, default=11, help="Base font size for plots.")
    p.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    return p.parse_args()

def rolling_weighted_mean(values: np.ndarray, window_hours: float) -> np.ndarray:
    """
    Compute start-aligned rolling mean over a window that may include a fractional final hour.
    Assumes equally spaced hourly samples.
    Returns an array aligned to the window start; trailing entries are NaN where incomplete.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    frac, whole = modf(window_hours)
    whole = int(whole)
    if whole < 1:
        whole = 1
    for start in range(0, n):
        end_full = start + whole - 1
        next_idx = end_full + 1
        if end_full >= n:
            break
        slice_full = values[start:end_full+1]
        if np.isnan(slice_full).any() or len(slice_full) < whole:
            continue
        if frac > 0 and next_idx < n and not np.isnan(values[next_idx]):
            mean_val = (slice_full.mean() * whole + values[next_idx] * frac) / (whole + frac)
        else:
            mean_val = slice_full.mean()
        out[start] = mean_val
    return out

def find_best_windows(df_in, intensity_col, price_col, duration_hours):
    """
    Given a dataframe with ['local_time', intensity_col, price_col], compute the best-by-price and best-by-intensity windows.
    """
    d = df_in[["local_time", intensity_col, price_col]].dropna().copy()
    d = d.sort_values("local_time").reset_index(drop=True)
    price = d[price_col].to_numpy(float)
    inten = d[intensity_col].to_numpy(float)
    price_wmean = rolling_weighted_mean(price, duration_hours)
    inten_wmean  = rolling_weighted_mean(inten, duration_hours)
    best_price_idx = int(np.nanargmin(price_wmean))
    best_price_time = d.loc[best_price_idx, "local_time"]
    best_price = float(price_wmean[best_price_idx])
    best_price_intensity = float(inten_wmean[best_price_idx])
    best_inten_idx = int(np.nanargmin(inten_wmean))
    best_inten_time = d.loc[best_inten_idx, "local_time"]
    best_inten = float(inten_wmean[best_inten_idx])
    best_inten_price = float(price_wmean[best_inten_idx])
    return {
        "price_col": price_col,
        "duration_h": duration_hours,
        "best_price_start": pd.to_datetime(best_price_time),
        "best_price_c_per_kwh": best_price,
        "best_price_intensity_g_per_kwh": best_price_intensity,
        "best_intensity_start": pd.to_datetime(best_inten_time),
        "best_intensity_g_per_kwh": best_inten,
        "best_intensity_c_per_kwh": best_inten_price,
    }

def analyse(timeseries_csv, landmarks_csv, outdir, need_kwh_list, charger_kw_list, price_cols, intensity_col, time_col, scatter_price_col, color_clean, color_cheap, font_size, dpi):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(timeseries_csv)
    # Parse times
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    # Build 'intensity_used' with fallback
    if intensity_col not in df.columns:
        if "intensity" in df.columns:
            df[intensity_col] = df["intensity"]
        else:
            raise ValueError(f"Cannot find intensity column '{intensity_col}' or fallback 'intensity'.")
    else:
        if "intensity" in df.columns:
            df[intensity_col] = df[intensity_col].where(df[intensity_col].notna(), df["intensity"])
    # Filter to rows where at least one price column is present
    for c in price_cols:
        if c not in df.columns:
            raise ValueError(f"Missing price column '{c}'. Available: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    any_price_mask = df[price_cols].notna().any(axis=1)
    dfn = df[[time_col, intensity_col] + price_cols][any_price_mask].copy()
    dfn = dfn.rename(columns={time_col: "local_time"})
    # Sensitivity sweep
    rows = []
    for need in need_kwh_list:
        for pwr in charger_kw_list:
            duration = need / pwr
            for price_col in price_cols:
                d = find_best_windows(dfn, intensity_col, price_col, duration)
                cost_cents_best_price = need * d["best_price_c_per_kwh"]
                cost_aud_best_price = cost_cents_best_price / 100.0
                cost_cents_best_inten = need * d["best_intensity_c_per_kwh"]
                cost_aud_best_inten = cost_cents_best_inten / 100.0
                emis_kg_best_price = need * d["best_price_intensity_g_per_kwh"] / 1000.0
                emis_kg_best_inten = need * d["best_intensity_g_per_kwh"] / 1000.0
                rows.append({
                    "price_basis": price_col,
                    "need_kwh": need,
                    "charger_kw": pwr,
                    "duration_h": round(duration, 2),
                    "best_price_start": d["best_price_start"],
                    "best_price_mean_c_per_kwh": round(d["best_price_c_per_kwh"], 3),
                    "best_price_cost_AUD": round(cost_aud_best_price, 2),
                    "best_price_emissions_kg": round(emis_kg_best_price, 2),
                    "best_intensity_start": d["best_intensity_start"],
                    "best_intensity_mean_g_per_kwh": round(d["best_intensity_g_per_kwh"], 1),
                    "best_intensity_cost_AUD": round(cost_aud_best_inten, 2),
                    "best_intensity_emissions_kg": round(emis_kg_best_inten, 2),
                })
    sweep = pd.DataFrame(rows)
    sweep["cost_delta_AUD"] = (sweep["best_price_cost_AUD"] - sweep["best_intensity_cost_AUD"]).round(2)
    sweep["emissions_delta_kg"] = (sweep["best_price_emissions_kg"] - sweep["best_intensity_emissions_kg"]).round(2)
    sweep_path = os.path.join(outdir, "system_summary.csv")
    sweep.to_csv(sweep_path, index=False)
    # Per-origin replication if landmarks present
    per_origin_path = ""
    if landmarks_csv and os.path.exists(landmarks_csv):
        lm = pd.read_csv(landmarks_csv)
        if "name" in lm.columns:
            per_origin = lm[["name"]].assign(key=1).merge(sweep.assign(key=1), on="key").drop(columns=["key"])
            per_origin_path = os.path.join(outdir, "per_origin_summary.csv")
            per_origin.to_csv(per_origin_path, index=False)
    # Correlation
    corrs = []
    for col in price_cols:
        mask = dfn[["local_time", intensity_col, col]].dropna()
        if not mask.empty:
            r = mask[intensity_col].corr(mask[col])
            corrs.append({"price_basis": col, "pearson_r_intensity_vs_price": r})
    corr_df = pd.DataFrame(corrs)
    corr_path = os.path.join(outdir, "correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    # Plots
    plt.rcParams.update({"font.size": font_size})
    # Scatter: price vs intensity
    if scatter_price_col in dfn.columns:
        m = dfn[[scatter_price_col, intensity_col]].dropna()
        plt.figure(dpi=dpi)
        plt.scatter(m[scatter_price_col], m[intensity_col], s=8, color="green", alpha=0.6)
        if len(m) > 2:
            x = m[scatter_price_col].to_numpy()
            y = m[intensity_col].to_numpy()
            coeffs = np.polyfit(x, y, 1)
            xfit = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            yfit = coeffs[0]*xfit + coeffs[1]
            plt.plot(xfit, yfit, color="red", linewidth=1.1)
#       plt.xlabel(f"{scatter_price_col} (c/kWh)", fontsize=8)
        plt.xlabel("Wholesale spot price (cents per kWh)", fontsize=8)
#       plt.ylabel("Intensity (gCO$_2$/kWh)", fontsize=8)
        plt.ylabel("Carbon intensity (gCO₂ per kWh)", fontsize=8)
        plt.xticks(fontsize=6)   # change to your preferred size
        plt.yticks(fontsize=6)
        plt.title("Relationship between wholesale electricity price and carbon intensity", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "scatter_price_vs_intensity.png"))
        plt.close()
    # Bars: for each scenario, compare cheapest vs cleanest (cost and emissions)
    for col in price_cols:
        sub = sweep[sweep["price_basis"] == col].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(["duration_h", "need_kwh", "charger_kw"]).reset_index(drop=True)
        labels = [f"{int(n)}kWh@{int(p)}kW" for n, p in zip(sub["need_kwh"], sub["charger_kw"])]
        x = np.arange(len(labels))
        width = 0.38
        plt.figure(dpi=dpi)
        plt.bar(x - width/2, sub["best_price_cost_AUD"], width, label="Cheapest", color=color_cheap)
        plt.bar(x + width/2, sub["best_intensity_cost_AUD"], width, label="Cleanest", color=color_clean)
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=6)
        plt.ylabel("Cost (AUD)")
        plt.yticks(fontsize=6)   # or size that works best
        plt.title("Cost of EV charging under cheapest vs cleanest windows", fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bars_cost_{col}.png"))
        plt.close()
        plt.figure(dpi=dpi)
        plt.bar(x - width/2, sub["best_price_emissions_kg"], width, label="Cheapest", color=color_cheap)
        plt.bar(x + width/2, sub["best_intensity_emissions_kg"], width, label="Cleanest", color=color_clean)
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=6)
        plt.ylabel("Emissions (kg CO₂)")
        plt.yticks(fontsize=6)   # or size that works best
        plt.title("Emissions from EV charging under cheapest vs cleanest windows", fontsize=10)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bars_emissions_{col}.png"))
        plt.close()
    return sweep_path, per_origin_path, corr_path

def main():
    args = parse_args()
    need_kwh = [float(x) for x in args.need_kwh.split(",") if x.strip()]
    charger_kw = [float(x) for x in args.charger_kw.split(",") if x.strip()]
    price_cols = [x.strip() for x in args.price_cols.split(",") if x.strip()]
    _ = analyse(
        timeseries_csv=args.timeseries_csv,
        landmarks_csv=args.landmarks_csv,
        outdir=args.outdir,
        need_kwh_list=need_kwh,
        charger_kw_list=charger_kw,
        price_cols=price_cols,
        intensity_col=args.intensity_col,
        time_col=args.time_col,
        scatter_price_col=args.scatter_price_col,
        color_clean=args.color_clean,
        color_cheap=args.color_cheap,
        font_size=args.font_size,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()
