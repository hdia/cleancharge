#!/usr/bin/env python3
"""
Per-origin extension for EV charging analysis (with flexible column mapping).

Adds origin-specific variation by:
1) Increasing the effective session duration by round-trip travel time:
     duration_eff = need_kwh/charger_kw + 2*distance_km/speed_kmph
2) Optionally adding per-km travel cost/emissions.

You can specify the name and distance columns in --equity_csv via flags:
  --equity_name_col "Origin"
  --equity_distance_col "Distance (km)"

If omitted, common names are auto-detected.

python .\src\zzev_charging_analyser_per_origin.py --timeseries_csv ".\data\processed\openelectricity_90d_hybrid_local_with_intensity.csv" --equity_csv ".\data\processed\equity_intensity_savings.csv" --outdir ".\data\processed\ev_outputs" --need_kwh "10,20,40" --charger_kw "7,11,50" --price_cols "spot_c_per_kwh,retailA_c_per_kwh,retailB_c_per_kwh" --speed_kmph 40 --travel_cost_per_km_aud 0.25 --travel_emissions_g_per_km 0 --equity_name_col "Origin" --equity_distance_col "Distance (km)"

Then
python .\src\zzmap_per_origin_plot.py --landmarks ".\data\processed\landmarks.csv" --per_origin ".\data\processed\ev_outputs\per_origin_summary_v2.csv" --basis retailA_c_per_kwh --metric emissions_delta_kg --scale_by_metric --scale_factor 800 --min_marker 25 --cmap YlGn --vmin_q 0.05 --vmax_q 0.95 --out ".\data\processed\ev_outputs\fig_map_emissions_delta_v2.png"

Outputs:
  per_origin_summary_v2.csv  (origin-specific cheapest vs cleanest results)
"""

import argparse
import os
from pathlib import Path
from datetime import time
import numpy as np
import pandas as pd
from math import modf

def parse_args():
    p = argparse.ArgumentParser(description="Per-origin EV charging analysis with travel-time/cost.")
    p.add_argument("--timeseries_csv", required=True)
    p.add_argument("--equity_csv", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--need_kwh", default="10,20,40")
    p.add_argument("--charger_kw", default="7,11,50")
    p.add_argument("--price_cols", default="spot_c_per_kwh,retailA_c_per_kwh,retailB_c_per_kwh")

    p.add_argument("--intensity_col", default="intensity_g_per_kwh")
    p.add_argument("--time_col", default="local_time")

    p.add_argument("--speed_kmph", type=float, default=40.0)
    p.add_argument("--travel_cost_per_km_aud", type=float, default=0.25)
    p.add_argument("--travel_emissions_g_per_km", type=float, default=0.0)

    p.add_argument("--allowed_start", default="", help='Optional "HH:MM-HH:MM" window for allowed start times (local).')

    # NEW: explicit mapping for equity columns
    p.add_argument("--equity_name_col", default="", help="Column in equity CSV to use as origin name (e.g., 'Origin').")
    p.add_argument("--equity_distance_col", default="", help="Column in equity CSV for distance in km (e.g., 'Distance (km)').")

    return p.parse_args()

def parse_hour_window(s: str):
    if not s:
        return None
    left, right = s.split("-")
    h1, m1 = map(int, left.split(":"))
    h2, m2 = map(int, right.split(":"))
    return time(h1, m1), time(h2, m2)

def rolling_weighted_mean(values: np.ndarray, window_hours: float) -> np.ndarray:
    """Start-aligned rolling mean with fractional final hour support."""
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    frac, whole = modf(window_hours); whole = int(whole or 1)
    for start in range(n):
        end_full = start + whole - 1
        next_idx = end_full + 1
        if end_full >= n: break
        sl = values[start:end_full+1]
        if np.isnan(sl).any() or len(sl) < whole: continue
        if frac > 0 and next_idx < n and not np.isnan(values[next_idx]):
            out[start] = (sl.mean()*whole + values[next_idx]*frac) / (whole + frac)
        else:
            out[start] = sl.mean()
    return out

def find_best_windows(df_in, intensity_col, price_col, duration_hours, allowed=None):
    d = df_in[["local_time", intensity_col, price_col]].dropna().copy()
    d = d.sort_values("local_time").reset_index(drop=True)
    if allowed:
        t1, t2 = allowed
        if t1 <= t2:
            d = d[d["local_time"].dt.time.between(t1, t2)]
        else:  # wrap-around
            mask = ~d["local_time"].dt.time.between(t2, t1)
            d = d[mask]
    if d.empty:
        raise ValueError("No rows after applying allowed start window.")
    price = d[price_col].to_numpy(float)
    inten = d[intensity_col].to_numpy(float)
    price_w = rolling_weighted_mean(price, duration_hours)
    inten_w = rolling_weighted_mean(inten, duration_hours)
    i_p = int(np.nanargmin(price_w)); i_c = int(np.nanargmin(inten_w))
    return {
        "best_price_start": pd.to_datetime(d.loc[i_p, "local_time"]),
        "best_price_c_per_kwh": float(price_w[i_p]),
        "best_price_intensity_g_per_kwh": float(inten_w[i_p]),
        "best_intensity_start": pd.to_datetime(d.loc[i_c, "local_time"]),
        "best_intensity_g_per_kwh": float(inten_w[i_c]),
        "best_intensity_c_per_kwh": float(price_w[i_c]),
    }

def main():
    args = parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    need_kwh = [float(x) for x in args.need_kwh.split(",") if x.strip()]
    charger_kw = [float(x) for x in args.charger_kw.split(",") if x.strip()]
    price_cols = [x.strip() for x in args.price_cols.split(",") if x.strip()]
    allowed = parse_hour_window(args.allowed_start) if args.allowed_start else None

    # Load timeseries
    df = pd.read_csv(args.timeseries_csv)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.sort_values(args.time_col).reset_index(drop=True)

    # Intensity column (with fallback)
    if args.intensity_col not in df.columns:
        if "intensity" in df.columns:
            df[args.intensity_col] = df["intensity"]
        else:
            raise ValueError(f"Missing intensity '{args.intensity_col}' and no 'intensity' fallback.")
    for c in price_cols:
        if c not in df.columns:
            raise ValueError(f"Missing price column '{c}'. Available: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    any_price = df[price_cols].notna().any(axis=1)
    dfn = df[[args.time_col, args.intensity_col] + price_cols][any_price].copy()
    dfn = dfn.rename(columns={args.time_col: "local_time"})

    # Load equity / distances
    eq = pd.read_csv(args.equity_csv)

    # Determine name column
    name_col = args.equity_name_col.strip() or next(
        (c for c in ["name", "Name", "origin", "Origin", "location", "Location", "site", "Site"] if c in eq.columns),
        None
    )
    if not name_col:
        raise ValueError(f"Could not find an origin name column. Pass --equity_name_col or include one of "
                         f"[name, Name, origin, Origin, location, Location, site, Site]. Found: {list(eq.columns)}")

    # Determine distance column
    dist_col = args.equity_distance_col.strip() or next(
        (c for c in ["distance_km", "Distance (km)", "Distance_km", "distance", "dist_km", "Dist_km"] if c in eq.columns),
        None
    )
    if not dist_col:
        raise ValueError(f"Could not find a distance column. Pass --equity_distance_col or include one of "
                         f"[distance_km, 'Distance (km)', Distance_km, distance, dist_km, Dist_km]. Found: {list(eq.columns)}")

    rows = []
    for _, r in eq.iterrows():
        origin = str(r[name_col])
        try:
            distance_km = float(r[dist_col])
        except Exception:
            distance_km = np.nan
        if not np.isfinite(distance_km):
            distance_km = 0.0

        travel_hours = (2.0 * distance_km) / max(args.speed_kmph, 1e-6)
        travel_cost_aud = (2.0 * distance_km) * args.travel_cost_per_km_aud
        travel_emis_kg = (2.0 * distance_km) * args.travel_emissions_g_per_km / 1000.0

        for need in need_kwh:
            for pwr in charger_kw:
                base_duration = need / pwr
                duration_eff = base_duration + travel_hours
                for price_col in price_cols:
                    w = find_best_windows(dfn, args.intensity_col, price_col, duration_eff, allowed)
                    cost_price_aud = (need * w["best_price_c_per_kwh"] / 100.0) + travel_cost_aud
                    cost_inten_aud = (need * w["best_intensity_c_per_kwh"] / 100.0) + travel_cost_aud
                    emis_price_kg = (need * w["best_price_intensity_g_per_kwh"] / 1000.0) + travel_emis_kg
                    emis_inten_kg = (need * w["best_intensity_g_per_kwh"] / 1000.0) + travel_emis_kg
                    rows.append({
                        "name": origin,
                        "distance_km": round(distance_km, 2),
                        "speed_kmph": args.speed_kmph,
                        "price_basis": price_col,
                        "need_kwh": need,
                        "charger_kw": pwr,
                        "duration_h_base": round(base_duration, 2),
                        "duration_h_travel": round(travel_hours, 2),
                        "duration_h_effective": round(duration_eff, 2),
                        "best_price_start": w["best_price_start"],
                        "best_price_mean_c_per_kwh": round(w["best_price_c_per_kwh"], 3),
                        "best_price_cost_AUD": round(cost_price_aud, 2),
                        "best_price_emissions_kg": round(emis_price_kg, 2),
                        "best_intensity_start": w["best_intensity_start"],
                        "best_intensity_mean_g_per_kwh": round(w["best_intensity_g_per_kwh"], 1),
                        "best_intensity_cost_AUD": round(cost_inten_aud, 2),
                        "best_intensity_emissions_kg": round(emis_inten_kg, 2),
                    })

    per_origin = pd.DataFrame(rows)
    per_origin["cost_delta_AUD"] = (per_origin["best_price_cost_AUD"] - per_origin["best_intensity_cost_AUD"]).round(2)
    per_origin["emissions_delta_kg"] = (per_origin["best_price_emissions_kg"] - per_origin["best_intensity_emissions_kg"]).round(2)

    outcsv = Path(args.outdir) / "per_origin_summary_v2.csv"
    per_origin.to_csv(outcsv, index=False)
    print(f">> Wrote {len(per_origin)} rows to {outcsv}")

if __name__ == "__main__":
    main()
