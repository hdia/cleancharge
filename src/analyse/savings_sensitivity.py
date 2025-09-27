
#!/usr/bin/env python3
"""
Savings sensitivity (theoretical upper bound) for EV charging.

For each day and each (need_kwh, charger_kw) scenario, select the cleanest N hours
and the dirtiest N hours (N may be fractional for the final hour) from within the day,
compute session emissions for each, and record the saving (worst - best).
No continuity constraint (hours may be non-contiguous). Prices are not used here.

Run with defaults (10/20/40 kWh, 7/50 kW)
python .\src\zzsavings_sensitivity_revised.py --timeseries_csv ".\data\processed\openelectricity_90d_hybrid_local_with_intensity.csv" --intensity_col "intensity_g_per_kwh" --time_col "local_time" --outdir ".\data\processed\ev_outputs" --need_kwh "10,20,40" --charger_kw "7,50"

Optional: restrict to an availability window (e.g., overnight 18:00–07:00):
python .\src\zzsavings_sensitivity_revised.py --timeseries_csv ".\data\processed\openelectricity_90d_hybrid_local_with_intensity.csv" --intensity_col "intensity_g_per_kwh" --time_col "local_time" --outdir ".\data\processed\ev_outputs" --need_kwh "10,20,40" --charger_kw "7,50" --daily_avail_window "18:00-07:00"


"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Savings sensitivity (theoretical upper bound).")
    p.add_argument("--timeseries_csv", type=str, default="data/processed/openelectricity_90d_hybrid_local_with_intensity.csv")
    p.add_argument("--intensity_col", type=str, default="intensity_g_per_kwh")
    p.add_argument("--time_col", type=str, default="local_time")
    p.add_argument("--outdir", type=str, default="data/processed/ev_outputs")
    p.add_argument("--need_kwh", type=str, default="10,20,40")
    p.add_argument("--charger_kw", type=str, default="7,50")
    p.add_argument("--daily_avail_window", type=str, default="", help='Optional "HH:MM-HH:MM" (local) window per day.')
    return p.parse_args()

def ensure_cols(df: pd.DataFrame, time_col: str, intensity_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'. Found: {list(df.columns)}")
    if intensity_col not in df.columns:
        if "intensity" in df.columns:
            df[intensity_col] = df["intensity"]
        else:
            raise ValueError(f"Missing intensity column '{intensity_col}' (and no 'intensity' fallback).")
    df = df[[time_col, intensity_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[intensity_col] = pd.to_numeric(df[intensity_col], errors="coerce")
    df = df.dropna().sort_values(time_col).reset_index(drop=True)
    return df

def within_window_local(s: pd.Series, start: str, end: str) -> pd.Series:
    t = s.dt.time
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    start_min = sh*60 + sm; end_min = eh*60 + em
    cur_min = s.dt.hour*60 + s.dt.minute
    if end_min >= start_min:
        return (cur_min >= start_min) & (cur_min < end_min)
    else:
        return (cur_min >= start_min) | (cur_min < end_min)

def pick_hours_mean_intensity(day_df: pd.DataFrame, hours_needed: float, intensity_col: str, best: bool) -> float:
    if day_df.empty or hours_needed <= 0:
        return np.nan
    ordered = day_df.sort_values(intensity_col, ascending=best).reset_index(drop=True)
    remaining = hours_needed
    tot = 0.0
    taken = 0.0
    for _, row in ordered.iterrows():
        if remaining <= 0:
            break
        take = min(1.0, remaining)
        tot += float(row[intensity_col]) * take
        taken += take
        remaining -= take
    return tot / taken if taken > 0 else np.nan

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    need_kwh = [float(x) for x in args.need_kwh.split(",") if x.strip()]
    charger_kw = [float(x) for x in args.charger_kw.split(",") if x.strip()]
    daily_window = None
    if args.daily_avail_window:
        try:
            start, end = args.daily_avail_window.split("-")
            daily_window = (start, end)
        except Exception:
            raise ValueError("Bad --daily_avail_window format. Use 'HH:MM-HH:MM'.")

    df = pd.read_csv(args.timeseries_csv)
    df = ensure_cols(df, time_col=args.time_col, intensity_col=args.intensity_col)
    # Ensure hourly frequency
    df = df.set_index(args.time_col).resample("1H").mean(numeric_only=True).dropna().reset_index()
    df["date"] = df[args.time_col].dt.date

    rows = []
    for need in need_kwh:
        for kw in charger_kw:
            hours = need / kw
            for day, day_df in df.groupby("date"):
                cur = day_df.copy()
                if daily_window:
                    mask = within_window_local(cur[args.time_col], daily_window[0], daily_window[1])
                    cur = cur[mask]
                if cur.empty:
                    continue
                best_mean = pick_hours_mean_intensity(cur, hours, args.intensity_col, best=True)
                worst_mean = pick_hours_mean_intensity(cur, hours, args.intensity_col, best=False)
                if np.isnan(best_mean) or np.isnan(worst_mean):
                    continue
                best_kg  = best_mean * need / 1000.0
                worst_kg = worst_mean * need / 1000.0
                rows.append({
                    "date": pd.to_datetime(day),
                    "need_kwh": need,
                    "charger_kw": kw,
                    "hours_exact": hours,
                    "best_mean_intensity": best_mean,
                    "worst_mean_intensity": worst_mean,
                    "best_kg": best_kg,
                    "worst_kg": worst_kg,
                    "saving_kg": worst_kg - best_kg,
                    "saving_pct": ((worst_kg - best_kg) / worst_kg * 100.0) if worst_kg > 0 else np.nan,
                })
    daily = pd.DataFrame(rows).sort_values(["need_kwh","charger_kw","date"])
    daily_path = outdir / "savings_sensitivity_daily.csv"
    daily.to_csv(daily_path, index=False)

    summary = (
        daily
        .groupby(["need_kwh","charger_kw"], as_index=False)
        .apply(lambda g: pd.Series({
            "days": g.shape[0],
            "best_kg_mean": g["best_kg"].mean(),
            "worst_kg_mean": g["worst_kg"].mean(),
            "saving_kg_mean": g["saving_kg"].mean(),
            "saving_kg_median": g["saving_kg"].median(),
            "saving_kg_p10": g["saving_kg"].quantile(0.10),
            "saving_kg_p90": g["saving_kg"].quantile(0.90),
            "saving_pct_mean": g["saving_pct"].mean(),
            "saving_pct_median": g["saving_pct"].median(),
            "saving_pct_p10": g["saving_pct"].quantile(0.10),
            "saving_pct_p90": g["saving_pct"].quantile(0.90),
        }))
        .reset_index(drop=True)
        .sort_values(["need_kwh","charger_kw"])
    )
    summary_path = outdir / "savings_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f">> Wrote {daily_path} ({len(daily)} rows)")
    print(f">> Wrote {summary_path} ({len(summary)} scenarios)")

if __name__ == "__main__":
    main()
