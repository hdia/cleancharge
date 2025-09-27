# Assumes: venv active; CSVs already placed under data\processed\ as below

# 0) Ensure dirs exist
New-Item -ItemType Directory -Force -Path "data\processed","data\processed\ev_outputs","results\figures" | Out-Null

# 1) System-wide charging analysis (contiguous, tariff-aware)
Write-Host "System analysis: cheapest vs cleanest (retail & spot)…"
python src\analyse\ev_charging_analyser_system.py `
  --timeseries_csv "data\processed\openelectricity_90d_hybrid_local_with_intensity.csv" `
  --outdir "data\processed\ev_outputs" `
  --need_kwh "10,20,40" `
  --charger_kw "7,11,50" `
  --price_cols "spot_c_per_kwh,retailA_c_per_kwh" `
  --intensity_col "intensity_g_per_kwh" `
  --time_col "local_time" `
  --scatter_price_col "spot_c_per_kwh" `
  --font_size 9 --dpi 150

# 2) Theoretical (non-contiguous) savings sensitivity
Write-Host "Sensitivity: theoretical upper bound (non-contiguous)…"
python src\analyse\savings_sensitivity.py `
  --timeseries_csv "data\processed\openelectricity_90d_hybrid_local_with_intensity.csv" `
  --intensity_col "intensity_g_per_kwh" `
  --time_col "local_time" `
  --outdir "data\processed\ev_outputs" `
  --need_kwh "10,20,40" `
  --charger_kw "7,50"

# 3) Map of per-origin average emissions savings (uses existing per_origin_summary_v2.csv)
Write-Host "Map: per-origin average emissions savings…"
python src\plots\plot_map_per_origin.py `
  --landmarks "data\processed\landmarks.csv" `
  --per_origin "data\processed\ev_outputs\per_origin_summary_v2.csv" `
  --metric "emissions_delta_kg" `
  --cmap "hot" `
  --scale_by_metric `
  --scale_factor 180 `
  --vmin_q 0.05 `
  --vmax_q 0.95 `
  --label_fontsize 8 `
  --out "results\figures\fig_map_emissions_delta.png"

Write-Host "Done. Figures under results\figures, outputs under data\processed\ev_outputs."
