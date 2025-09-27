#!/usr/bin/env python3
"""
Boxplots of daily theoretical savings (kg CO2) per scenario.

Usage (PowerShell):
  python .\src\zzplot_theoretical_savings_boxplots.py ^
    --daily_csv ".\data\processed\ev_outputs\savings_sensitivity_daily.csv" ^
    --out_png ".\data\processed\ev_outputs\fig_box_theoretical_savings.png" ^
    --dpi 150 --ylim 15 --jitter --mean_markers

Run with
python .\src\zzplot_theoretical_savings_boxplots.py --daily_csv ".\data\processed\ev_outputs\savings_sensitivity_daily.csv" --out_png ".\data\processed\ev_outputs\fig_box_theoretical_savings_nonzero.png" --dpi 150 --ylim 25 --jitter --mean_markers --drop_zero


"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

def parse_args():
    p = argparse.ArgumentParser(description="Boxplots of daily theoretical savings per scenario.")
    p.add_argument("--daily_csv", required=True)
    p.add_argument("--out_png", required=True)
    p.add_argument("--label_fmt", default="{need}@{kw}")
    p.add_argument("--ylim", type=float, default=None)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--jitter", action="store_true", help="Overlay jittered daily points")
    p.add_argument("--drop_zero", action="store_true", help="Exclude zero-saving days in boxplots")
    p.add_argument("--mean_markers", action="store_true", help="Overlay per-scenario means (diamonds)")
    return p.parse_args()

def fmt_num(x): return int(x) if float(x).is_integer() else x

def main():
    a = parse_args()
    df = pd.read_csv(a.daily_csv)
    for col in ["need_kwh","charger_kw","saving_kg"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    d = df[["need_kwh","charger_kw","saving_kg"]].dropna().copy()
    d["label"] = [a.label_fmt.format(need=fmt_num(n), kw=fmt_num(k))
                  for n,k in zip(d["need_kwh"], d["charger_kw"])]

    order = (d.groupby(["need_kwh","charger_kw"], as_index=False).size()
               .sort_values(["need_kwh","charger_kw"]))
    labels = [a.label_fmt.format(need=fmt_num(n), kw=fmt_num(k))
              for n,k in zip(order["need_kwh"], order["charger_kw"])]

    # Build data arrays + basic counts
    data_full = [d.loc[d["label"]==lab, "saving_kg"].to_numpy() for lab in labels]
    counts = []
    for lab, arr in zip(labels, data_full):
        n = len(arr)
        n_zero = int(np.sum(np.isclose(arr, 0.0)))
        counts.append((lab, n, n_zero, n - n_zero))

    # Optionally drop the zeros for the visual
    data = [arr[~np.isclose(arr, 0.0)] if a.drop_zero else arr for arr in data_full]

    # Figure
    plt.figure(dpi=a.dpi)
    bp = plt.boxplot(
        data,
        tick_labels=labels,
        showfliers=True,
        patch_artist=True,
        boxprops=dict(facecolor="#e7eef7", edgecolor="#1f2937", linewidth=1.4),
        medianprops=dict(color="#b91c1c", linewidth=2.0),
        whiskerprops=dict(color="#1f2937", linewidth=1.4),
        capprops=dict(color="#1f2937", linewidth=1.4),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none", markeredgecolor="#374151", alpha=0.7),
    )

    # Optional jitter (all points, including zeros, lightly)
    if a.jitter:
        for xi, vals in enumerate(data_full, start=1):
            x = xi + (rng.random(len(vals)) - 0.5) * 0.18
            plt.plot(x, vals, "o", ms=2.2, mfc="none", mec="#6b7280", alpha=0.35, zorder=1)

    # Optional mean markers on top of boxes
    if a.mean_markers:
        means = [np.nanmean(v) if len(v) else np.nan for v in data_full]
        plt.plot(range(1, len(labels)+1), means, "D", ms=5, mfc="#111827", mec="#111827", zorder=3)

    plt.ylabel("Daily theoretical saving (kg CO$_2$)", fontsize=8)
    plt.xlabel("Scenario (kWh@kW)", fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    title = "Theoretical upper-bound savings per session (by day - non-zero days)"
#   if a.drop_zero:
#       title += " — non-zero days"
    plt.title(title, fontsize=10)
    if a.ylim is not None:
        plt.ylim(0, a.ylim)
    else:
        allv = np.concatenate([v for v in data if len(v)])
        ymax = np.nanmax(allv) if len(allv) else 10
        plt.ylim(0, ymax * 1.5)
    plt.tight_layout()
    plt.savefig(a.out_png)
    plt.close()

    # Print counts so you can cite them
    print("Scenario\tN_days\tN_zero\tN_nonzero")
    for lab, n, n0, n1 in counts:
        print(f"{lab}\t{n}\t{n0}\t{n1}")

if __name__ == "__main__":
    main()
