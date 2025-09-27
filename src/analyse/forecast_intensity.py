# zzforecast_intensity.py
# Clean replacement: consistent lag features, raw metrics, smoothed plot-only
# Usage example:
#   python .\src\zzforecast_intensity.py --input .\data\processed\openelectricity_90d_hybrid_local.csv --outdir .\data\processed --smooth_plot 2 --window_hours 3
#   (You can change --smooth_plot to 0/1/2/3 and --window_hours to 2–4 as you like.)

# 30-day (outputs go to data\processed\30d\)
# python .\src\zzforecast_intensity.py --input .\data\processed\openelectricity_emissions_30d_local.csv --outdir .\data\processed\30d --smooth_plot 2 --window_hours 3

# 90-day (outputs go to data\processed\90d\)
# python .\src\zzforecast_intensity.py --input .\data\processed\openelectricity_90d_hybrid_local.csv --outdir .\data\processed\90d --smooth_plot 2 --window_hours 3


from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ----------------------------
# Config / constants
# ----------------------------
TZ_NAME = "Australia/Melbourne"
TIME_COL = "local_time"         # will be created/normalized from input
INTENSITY_CANDIDATES = [
    "intensity", "intensity_g_per_kwh", "intensity_g_per_kWh",
    "co2_intensity_g_per_kwh", "carbon_intensity_g_per_kwh"
]
PRICE_CANDIDATES = ["spot_price_per_mwh", "price_per_mwh", "price"]
DEMAND_CANDIDATES = ["demand_mw", "demand", "load_mw"]

# fixed lag set we promise to model and later recreate for recursive forecasting
INT_LAGS = [1,2,3,6,12,24]
# optional exogenous (only if columns exist end-to-end)
EXO_LAGS = [1,2,3,6,12,24]

# ----------------------------
# Utilities
# ----------------------------
def find_column(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None

def ensure_local_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a TZ-aware 'local_time' (Australia/Melbourne), hourly frequency.
    Accepts any of:
      - 'local_time' (any tz) or naive
      - 'ts' (UTC)
      - any datetime-looking column
    """
    df = df.copy()

    # choose a source time column
    time_col = None
    for cand in [TIME_COL, "ts", "datetime", "time", "date"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError("Could not find a time column. Expected one of: local_time/ts/datetime/time/date.")

    # to datetime
    s = pd.to_datetime(df[time_col], errors="coerce", utc=True if time_col=="ts" else False)
    if s.dt.tz is None:
        # naive → assume it's local Melbourne time (your processed CSVs are local)
        s = s.dt.tz_localize(TZ_NAME)
    else:
        # convert to Melbourne
        s = s.dt.tz_convert(TZ_NAME)

    df[TIME_COL] = s

    # sort & hourly reindex (preserve existing hourly cadence; if sub-hourly, aggregate)
    df = df.sort_values(TIME_COL)
    # if sub-hourly exists, take hourly mean
    # create an hourly index covering the range
    idx = pd.date_range(
        df[TIME_COL].min().floor("H"),
        df[TIME_COL].max().ceil("H"),
        freq="H",
        tz=TZ_NAME
    )
    df = (
        df.set_index(TIME_COL)
          .resample("1H")
          .mean(numeric_only=True)  # keeps only numeric
          .reindex(idx)
          .reset_index()
          .rename(columns={"index": TIME_COL})
    )
    return df

def choose_intensity_column(df: pd.DataFrame) -> str:
    col = find_column(df, INTENSITY_CANDIDATES)
    if not col:
        raise ValueError(f"Could not find intensity column. Tried: {INTENSITY_CANDIDATES}")
    return col

def optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return find_column(df, candidates)

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df[TIME_COL]
    df["hour"] = t.dt.hour
    df["dow"] = t.dt.dayofweek
    # cyclic encoding
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7)
    return df

def add_lags(df: pd.DataFrame, col: str, prefix: str, lags: list[int]) -> pd.DataFrame:
    df = df.copy()
    for L in lags:
        df[f"{prefix}_lag_{L}"] = df[col].shift(L)
    return df

def feature_frame(
    df_raw: pd.DataFrame,
    intensity_col: str,
    price_col: str | None,
    demand_col: str | None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a feature matrix with a **fixed** set of columns:
    intensity lags (INT_LAGS), optional price/demand lags if available, and calendar.
    Returns (df_design, feature_names).
    """
    df = df_raw.copy()
    # calendar first
    df = add_calendar_features(df)

    # intensity lags (always enforced)
    df = add_lags(df, intensity_col, "int", INT_LAGS)

    feature_names = ["hour_sin","hour_cos","dow_sin","dow_cos"] + [f"int_lag_{L}" for L in INT_LAGS]

    # optional exogenous – include only if fully present
    if price_col:
        df = add_lags(df, price_col, "price", EXO_LAGS)
        feature_names += [f"price_lag_{L}" for L in EXO_LAGS]
    if demand_col:
        df = add_lags(df, demand_col, "demand", EXO_LAGS)
        feature_names += [f"demand_lag_{L}" for L in EXO_LAGS]

    # drop rows with NaNs created by lagging at the start
    df = df.dropna(subset=feature_names + [intensity_col]).reset_index(drop=True)
    return df, feature_names

@dataclass
class ModelBundle:
    model: GradientBoostingRegressor
    features: list[str]
    intensity_col: str
    price_col: str | None
    demand_col: str | None

# ----------------------------
# Modeling / evaluation
# ----------------------------
def train_model(df_design: pd.DataFrame, feature_names: list[str], target: str) -> ModelBundle:
    X = df_design[feature_names].to_numpy()
    y = df_design[target].to_numpy()
    m = GradientBoostingRegressor(random_state=42)
    m.fit(X, y)
    return ModelBundle(m, feature_names, target, None, None)

def walk_forward_backtest_last7(df_design: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    """
    Hourly walk-forward over the last 7 days of df_design.
    Train on all data up to t-1, predict t.
    Metrics on **raw** predictions.
    """
    df = df_design.copy()
    last7_cut = df[TIME_COL].max() - pd.Timedelta(days=7)
    df7 = df[df[TIME_COL] >= last7_cut].copy()
    preds = []
    for i in range(len(df7)):
        t_row = df7.iloc[i]
        t_idx = df.index.get_loc(t_row.name)
        # train up to original index of current row
        train_df = df.iloc[:t_idx].dropna(subset=features + [target])
        if len(train_df) < 100:
            preds.append(np.nan)
            continue
        m = GradientBoostingRegressor(random_state=42)
        m.fit(train_df[features].to_numpy(), train_df[target].to_numpy())
        yhat = float(m.predict(t_row[features].to_numpy().reshape(1, -1))[0])
        preds.append(yhat)
    out = df7[[TIME_COL, target]].copy()
    out["y_hat"] = preds
    out = out.dropna()
    return out

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    num = np.abs(y_true - y_pred)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.where(den == 0, 0.0, num / den))) * 100.0

# ----------------------------
# Recursive 24h forecast (raw series)
# ----------------------------
def recursive_forecast_next24(df_design_full: pd.DataFrame, mb: ModelBundle) -> pd.DataFrame:
    """
    Use the last available hour as the starting point, roll forward 24 steps.
    Recreate the exact lag feature set each step from the **growing** series.
    Returns DataFrame with TIME_COL, intensity_hat (raw).
    """
    # working copy of the full design (so we can take last row and extend)
    df = df_design_full.sort_values(TIME_COL).reset_index(drop=True)
    horizon = 24

    # Keep the last row as the starting context
    history = df.tail(max(24, max(INT_LAGS)+1)).copy()

    rows = []
    for step in range(1, horizon + 1):
        # next timestamp
        t_next = history[TIME_COL].iloc[-1] + pd.Timedelta(hours=1)

        # build one-row feature vector from history
        new = pd.DataFrame({TIME_COL: [t_next]})
        # calendar
        new["hour"] = t_next.hour
        new["dow"] = t_next.dayofweek
        new["hour_sin"] = np.sin(2*np.pi*new["hour"]/24)
        new["hour_cos"] = np.cos(2*np.pi*new["hour"]/24)
        new["dow_sin"]  = np.sin(2*np.pi*new["dow"]/7)
        new["dow_cos"]  = np.cos(2*np.pi*new["dow"]/7)

        # intensity lags from last 'history'
        for L in INT_LAGS:
            new[f"int_lag_{L}"] = history[mb.intensity_col].iloc[-L]

        # optional exogenous – only if present in model features
        if any(f.startswith("price_lag_") for f in mb.features):
            for L in EXO_LAGS:
                src = f"price_lag_{L}".replace("_lag_", "_")  # we don't store, so read from raw column name
                # But we don't know the raw price column name here; instead infer from last design frame columns:
                # find a column in history that is the raw price series (not lagged)
                price_raw = None
                for cand in PRICE_CANDIDATES:
                    if cand in history.columns:
                        price_raw = cand
                        break
                if price_raw is None:
                    raise RuntimeError("Model expects price lags but no raw price column found in history.")
                new[f"price_lag_{L}"] = history[price_raw].iloc[-L]

        if any(f.startswith("demand_lag_") for f in mb.features):
            demand_raw = None
            for cand in DEMAND_CANDIDATES:
                if cand in history.columns:
                    demand_raw = cand
                    break
            if demand_raw is None:
                raise RuntimeError("Model expects demand lags but no raw demand column found in history.")
            for L in EXO_LAGS:
                new[f"demand_lag_{L}"] = history[demand_raw].iloc[-L]

        # predict raw intensity
        X = new[mb.features].to_numpy()
        yhat = float(mb.model.predict(X)[0])
        new_out = new[[TIME_COL]].copy()
        new_out["intensity_hat"] = yhat
        rows.append(new_out.iloc[0])

        # append to history as if observed (so future lags work)
        append_row = pd.DataFrame({c: [np.nan] for c in history.columns})
        append_row[TIME_COL] = t_next
        append_row[mb.intensity_col] = yhat
        # Also carry over raw exogenous (we don't have future exogenous; keep last observed value as a hold)
        for cand in PRICE_CANDIDATES + DEMAND_CANDIDATES:
            if cand in history.columns:
                append_row[cand] = history[cand].iloc[-1]
        history = pd.concat([history, append_row], ignore_index=True)

    fc = pd.DataFrame(rows)
    return fc

# ----------------------------
# Plotting helpers
# ----------------------------
def style_axes(ax, title: str):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Date-Time", fontsize=10)
    ax.set_ylabel("Grid carbon intensity (gCO₂/kWh)", fontsize=10)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(True, alpha=0.3)

def plot_validation(df_backtest: pd.DataFrame, outpath: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_backtest[TIME_COL], df_backtest["y_true"], ls="--", c="red", label="Actual", linewidth=2, alpha=0.9)
    ax.plot(df_backtest[TIME_COL], df_backtest["y_hat"],  ls="-",  c="green", label="Predicted", linewidth=2)
    style_axes(ax, "Validation of carbon intensity predictions using 90-day historical data - Melbourne (last 7 days)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def plot_forecast(
    df_hist: pd.DataFrame,
    df_fc_raw: pd.DataFrame,
    best_block: pd.DataFrame,
    worst_block: pd.DataFrame,
    outpath: Path,
    smooth_k: int = 0
):
    import matplotlib.pyplot as plt
    # prepare 48h history
    hist_cut = df_hist[TIME_COL].max() - pd.Timedelta(hours=48)
    h = df_hist[df_hist[TIME_COL] >= hist_cut].copy()

    # smooth only for display
    f = df_fc_raw.copy()
    if smooth_k and smooth_k > 1:
        f["intensity_hat_plot"] = f["intensity_hat"].rolling(smooth_k, min_periods=1, center=True).mean()
    else:
        f["intensity_hat_plot"] = f["intensity_hat"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(h[TIME_COL], h["y_true"], ls="--", c="red", linewidth=2, label="Historical")
    ax.plot(f[TIME_COL], f["intensity_hat_plot"], c="green", linewidth=2, label="Predicted next 24h (smoothed)")

    # shade windows
    def shade_block(block: pd.DataFrame, color: str, alpha: float, label: str):
        if len(block) == 0:
            return
        start = block[TIME_COL].iloc[0]
        end   = block[TIME_COL].iloc[-1] + pd.Timedelta(hours=1)
        ax.axvspan(start, end, color=color, alpha=alpha, label=label)

    shade_block(best_block,  "green", 0.12, "Best charging window")
    shade_block(worst_block, "red",   0.12, "Worst charging window")

    style_axes(ax, "Grid carbon intensity – Melbourne (last 48h + next 24h forecast - using 90-day historical data)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

# ----------------------------
# Savings helpers (raw forecast)
# ----------------------------
def block_emissions_from_forecast(df_fc: pd.DataFrame, need_kwh: float, charger_kw: float, block_h: int) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Compute best/worst contiguous 'block_h' hours using **raw** intensity_hat.
    Return (best_block_df, worst_block_df, kg_best, kg_avg, kg_worst)
    """
    f = df_fc[[TIME_COL, "intensity_hat"]].copy()
    f["block_mean"] = f["intensity_hat"].rolling(block_h, min_periods=block_h).mean()

    # find best and worst starting indices
    valid = f.dropna(subset=["block_mean"]).reset_index(drop=True)
    if valid.empty:
        # fallback – not enough horizon yet
        return f.head(0), f.head(0), np.nan, np.nan, np.nan

    i_best = int(valid["block_mean"].idxmin())
    i_worst = int(valid["block_mean"].idxmax())

    best_block = f.iloc[i_best - (block_h - 1): i_best + 1].copy()
    worst_block = f.iloc[i_worst - (block_h - 1): i_worst + 1].copy()

    avg_int = f["intensity_hat"].mean()
    best_int = best_block["intensity_hat"].mean()
    worst_int = worst_block["intensity_hat"].mean()

    kg_best  = best_int  * need_kwh / 1000.0
    kg_avg   = avg_int   * need_kwh / 1000.0
    kg_worst = worst_int * need_kwh / 1000.0
    return best_block, worst_block, float(kg_best), float(kg_avg), float(kg_worst)

# ----------------------------
# Main
# ----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",    required=True, help="Path to processed hybrid/local CSV (hourly or sub-hourly).")
    p.add_argument("--outdir",   required=True, help="Directory to write plots/CSVs.")
    p.add_argument("--smooth_plot", type=int, default=2, help="Rolling window (hours) to smooth ONLY the plotted forecast (0/1=off).")
    p.add_argument("--window_hours", type=int, default=3, help="Charging session block (e.g., 3 hours for 20 kWh @7 kW).")
    p.add_argument("--need_kwh", type=float, default=20.0)
    p.add_argument("--charger_kw", type=float, default=7.0)
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load & normalize time
    df0 = pd.read_csv(args.input)
    df0 = ensure_local_time(df0)

    # choose columns
    intensity_col = choose_intensity_column(df0)
    price_col = optional_column(df0, PRICE_CANDIDATES)
    demand_col = optional_column(df0, DEMAND_CANDIDATES)

    # Build design frame (fixed features)
    df_design, features = feature_frame(df0, intensity_col, price_col, demand_col)

    # Keep a small convenience frame with 'y_true'
    df_design["y_true"] = df_design[intensity_col]

    # ---------------- Backtest (raw) ----------------
    df_bt = walk_forward_backtest_last7(df_design[[TIME_COL, "y_true"] + features], features, "y_true")
    if len(df_bt) > 10:
        mae = mean_absolute_error(df_bt["y_true"], df_bt["y_hat"])
        r2  = r2_score(df_bt["y_true"], df_bt["y_hat"])
        s   = smape(df_bt["y_true"].to_numpy(), df_bt["y_hat"].to_numpy())
        print("\n=== Validation of short-term carbon intensity forecasts (last 7 days) ===")
        print(f"MAE:  {mae:.1f} gCO₂/kWh")
        print(f"sMAPE:{s:.1f}%")
        print(f"R²:   {r2:.2f}")
        plot_validation(df_bt, outdir / "intensity_backtest_last168h.png")
        df_bt.to_csv(outdir / "intensity_backtest_last7d.csv", index=False)
    else:
        print("Not enough data to compute 7-day backtest.")

    # ---------------- Train final model on all design rows ----------------
    mb = ModelBundle(
        model=GradientBoostingRegressor(random_state=42).fit(df_design[features].to_numpy(), df_design["y_true"].to_numpy()),
        features=features,
        intensity_col="y_true",
        price_col=price_col,
        demand_col=demand_col,
    )

    # ---------------- Forecast next 24h (raw) ----------------
    fc_raw = recursive_forecast_next24(df_design[[TIME_COL, "y_true"] + ([price_col] if price_col else []) + ([demand_col] if demand_col else [])], mb)
    fc_raw.to_csv(outdir / "intensity_forecast_next24.csv", index=False)

    # ---------------- Savings (raw) ----------------
    b_blk, w_blk, kg_best, kg_avg, kg_worst = block_emissions_from_forecast(
        fc_raw, args.need_kwh, args.charger_kw, args.window_hours
    )
    print("\n=== Forecasted 24h session (RAW) ===")
    if not np.isnan(kg_best):
        print(f"- Charge in BEST block:  {kg_best:.1f} kg CO₂")
        print(f"- Charge in AVERAGE:     {kg_avg:.1f} kg CO₂")
        print(f"- Charge in WORST block: {kg_worst:.1f} kg CO₂")
        print(f"=> Potential saving (worst→best): {kg_worst - kg_best:.1f} kg CO₂")
    else:
        print("Not enough horizon to compute block savings.")

    # ---------------- Plot forecast (smooth only the line that is drawn) ----------------
    plot_forecast(df_design[[TIME_COL, "y_true"]], fc_raw, b_blk, w_blk,
                  outdir / "intensity_forecast_plot.png", smooth_k=args.smooth_plot)

if __name__ == "__main__":
    main()
