#!/usr/bin/env python3
"""
Map per-origin average savings with size + colour encoding.

Inputs:
  --landmarks  CSV with: name, lat, lon
  --per_origin CSV from analyser with: name, price_basis, cost_delta_AUD, emissions_delta_kg, ...

Example:

python src/zzmap_per_origin_plot.py --landmarks data/processed/landmarks.csv --per_origin data/processed/ev_outputs/per_origin_summary_v2.csv --basis retailA_c_per_kwh --metric emissions_delta_kg --scale_by_metric --scale_factor 800 --min_marker 25 --cmap YlGn --vmin_q 0.05 --vmax_q 0.95 --out data/processed/ev_outputs/fig_map_emissions_delta_v2.png

"""

import argparse
from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm, colors
# keep only the lower part of 'hot' (0.00–0.50 ≈ black→red)
base = cm.get_cmap("hot")
hot_black_red = colors.LinearSegmentedColormap.from_list(
    "hot_black_red", base(np.linspace(0.0, 0.3, 256))
)

def parse_args():
    p = argparse.ArgumentParser(description="Map per-origin average savings (size + colour).")
    p.add_argument("--landmarks", required=True, help="Path to landmarks.csv with name,lat,lon.")
    p.add_argument("--per_origin", required=True, help="Path to per_origin_summary.csv.")
    p.add_argument("--basis", default="retailA_c_per_kwh",
                   help="Price basis to filter: retailA_c_per_kwh, retailB_c_per_kwh, spot_c_per_kwh.")
    p.add_argument("--metric", default="emissions_delta_kg",
                   choices=["emissions_delta_kg", "cost_delta_AUD"],
                   help="Metric to encode by colour/size.")
    p.add_argument("--cmap", default="Greens", help="Matplotlib colormap. For savings, 'Greens' works well.")
    # Size encoding
    p.add_argument("--scale_by_metric", action="store_true",
                   help="Scale marker size by metric value.")
    p.add_argument("--scale_factor", type=float, default=60.0,
                   help="Multiplicative factor for marker area. Tune per figure.")
    # Colour scaling
    p.add_argument("--vmin", type=float, default=None, help="Explicit colour scale min. Overrides vmin_q if set.")
    p.add_argument("--vmax", type=float, default=None, help="Explicit colour scale max. Overrides vmax_q if set.")
    p.add_argument("--vmin_q", type=float, default=0.05, help="Lower quantile for robust min if vmin not set.")
    p.add_argument("--vmax_q", type=float, default=0.95, help="Upper quantile for robust max if vmax not set.")
    # Aesthetics
    p.add_argument("--markersize", type=float, default=90.0,
                   help="Baseline marker size when not scaling by metric (or as minimum size if scaling).")
    p.add_argument("--min_marker", type=float, default=30.0,
                   help="Minimum marker size when scaling by metric.")
    p.add_argument("--label_fontsize", type=float, default=8.0, help="Text label font size.")
    p.add_argument("--out", required=True, help="Output image path.")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lm = pd.read_csv(args.landmarks)
    po = pd.read_csv(args.per_origin)

    for col in ["name", "lat", "lon"]:
        if col not in lm.columns:
            raise ValueError(f"Landmarks must include '{col}'. Found: {list(lm.columns)}")

    req_po = {"name", "price_basis", "cost_delta_AUD", "emissions_delta_kg"}
    if not req_po.issubset(po.columns):
        raise ValueError(f"Per-origin file must include {req_po}. Found: {list(po.columns)}")

    sub = po[po["price_basis"] == args.basis].copy()
    if sub.empty:
        raise ValueError(f"No rows for price_basis='{args.basis}'. "
                         f"Available: {sorted(po['price_basis'].unique().tolist())}")

    # Aggregate to average per origin
    agg = (sub.groupby("name")[["cost_delta_AUD", "emissions_delta_kg"]]
              .mean()
              .reset_index())

    df_map = lm.merge(agg, on="name", how="left")
    gdf = gpd.GeoDataFrame(df_map,
                           geometry=gpd.points_from_xy(df_map["lon"], df_map["lat"]),
                           crs="EPSG:4326")

    # Metric values and robust colour scaling
    vals = gdf[args.metric].astype(float)
    if args.vmin is None:
        vmin = np.nanquantile(vals, args.vmin_q)
    else:
        vmin = args.vmin
    if args.vmax is None:
        vmax = np.nanquantile(vals, args.vmax_q)
    else:
        vmax = args.vmax
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)

    # Size scaling
    if args.scale_by_metric:
        # Normalise to [0,1] within [vmin, vmax], clamp, then scale
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else vals*0
        norm = np.clip(norm, 0, 1)
        sizes = args.min_marker + norm * args.scale_factor
    else:
        sizes = np.full(len(gdf), args.markersize, dtype=float)

    # Plot
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = plt.gca()
    gplt = gdf.plot(column=args.metric,
                    cmap=hot_black_red,
                    legend=True,
                    vmin=vmin, vmax=vmax,
                    markersize=sizes,
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax)
    ax.set_xlim(gdf.geometry.x.min() - 0.2, gdf.geometry.x.max() + 0.2)
    ax.set_ylim(gdf.geometry.y.min() - 0.2, gdf.geometry.y.max() + 0.2)

    # Labels
    for x, y, size, label in zip(gdf.geometry.x, gdf.geometry.y, sizes, gdf["name"]):
        if pd.notna(x) and pd.notna(y) and isinstance(label, str):
        # Scale offset to marker size (smaller circles get smaller offset)
          offset = 0.01 + size / max(sizes) * 0.03
          ax.text(x + offset, y + offset, label, fontsize=args.label_fontsize)
#         ax.text(x + 0.05, y + 0.05, label, fontsize=args.label_fontsize)

    # Titles
    metric_label = "Average emissions savings (kg CO₂)" if args.metric == "emissions_delta_kg" \
                   else "Average cost difference (AUD, cheapest − cleanest)"
#   ax.set_title(f"{metric_label} by origin\nBasis: {args.basis}")
    ax.set_title(f"{metric_label} by origin\n", fontsize=11)
    ax.set_xlabel("Longitude", fontsize=9, labelpad=12)
    ax.set_ylabel("Latitude", fontsize=9, labelpad=1)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # Grab the current colorbar and adjust font
    cbar = gplt.get_figure().axes[-1]   # the last axis is usually the colorbar
    cbar.tick_params(labelsize=9)
    cbar.set_ylabel(metric_label, fontsize=9)
    cbar_ax = gplt.get_figure().axes[-1]   # last axis is the colorbar
    cbar_ax.set_ylabel("Average emissions savings (kg CO₂)", fontsize=8, labelpad=10)
    cbar_ax.tick_params(labelsize=8)


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
