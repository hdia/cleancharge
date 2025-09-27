# src/fetch_openelectricity_master_90d.py
"""
Fetch a 90-day master dataset from OpenElectricity with correct endpoints.

Endpoints:
- /v4/data/network/{REGION}: power, energy, emissions, market_value, renewable_proportion
- /v4/market/network/{REGION}: price, demand, demand_energy, curtailment*

Metrics pulled:
- emissions (t)                      [data]
- energy (MWh)                       [data]
- price ($/MWh)                      [market]
- demand (MW)                        [market]
- demand_energy (MWh)                [market]
- renewable_proportion (%)           [data]
- curtailment_solar_utility (MW)     [market]
- curtailment_solar_utility_energy (MWh) [market]
- curtailment_wind (MW)              [market]
- curtailment_wind_energy (MWh)      [market]

Derived:
- intensity_g_per_kwh = (emissions_t / energy_mwh) * 1000
- price_$per_kwh = price_$perMWh / 1000
- renewable_share = renewable_proportion_pct / 100

Usage:
  python src/fetch_openelectricity_master_90d.py \
    --days 90 --interval 1h --region NEM \
    --out data/processed/openelectricity_master_90d.csv

Requires:
  pip install python-dotenv requests pandas
  .env with OPENELECTRICITY_API_KEY=...
"""

import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd

# Load .env so you do not need to export manually
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("OPENELECTRICITY_API_KEY")

BASE_DATA = "https://api.openelectricity.org.au/v4/data/network"
BASE_MARKET = "https://api.openelectricity.org.au/v4/market/network"

# Map metric → (endpoint_group, output_column)
# endpoint_group is either "data" or "market"
METRICS = {
    # data endpoint
    "emissions": ("data", "emissions_t"),
    "energy": ("data", "energy_mwh"),
    "renewable_proportion": ("data", "renewable_proportion_pct"),
    # market endpoint
    "price": ("market", "price_$perMWh"),
    "demand": ("market", "demand_MW"),
    "demand_energy": ("market", "demand_energy_MWh"),
    "curtailment_solar_utility": ("market", "curtailment_solar_MW"),
    "curtailment_solar_utility_energy": ("market", "curtailment_solar_MWh"),
    "curtailment_wind": ("market", "curtailment_wind_MW"),
    "curtailment_wind_energy": ("market", "curtailment_wind_MWh"),
}

def _window(days: int) -> tuple[str, str]:
    """End at the top of the last whole hour to align with hourly bins."""
    now_utc = datetime.now(timezone.utc)
    end = now_utc.replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    return start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")

def _session() -> requests.Session:
    if not API_KEY:
        raise RuntimeError("Set OPENELECTRICITY_API_KEY in your environment or .env")
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    s.headers.update({"Authorization": f"Bearer {API_KEY}"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def _get_metric(sess: requests.Session, endpoint_group: str, metric: str,
                region: str, start_iso: str, end_iso: str, interval: str, out_col: str) -> pd.DataFrame:
    """
    Fetch a single metric and return a ts/value dataframe.
    """
    base = BASE_DATA if endpoint_group == "data" else BASE_MARKET
    url = f"{base}/{region}"
    params = {
        "metrics": metric,
        "interval": interval,
        "date_start": start_iso,
        "date_end": end_iso,
    }
    try:
        r = sess.get(url, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        print(f"!! HTTP {code} for metric '{metric}' at {endpoint_group}. Skipping.")
        return pd.DataFrame(columns=["ts", out_col])
    except Exception as e:
        print(f"!! Error fetching metric '{metric}': {e}. Skipping.")
        return pd.DataFrame(columns=["ts", out_col])

    rows = payload.get("results", [])
    if not rows:
        print(f"!! Empty results for metric '{metric}'.")
        return pd.DataFrame(columns=["ts", out_col])
    data = rows[0].get("data", [])
    if not data:
        print(f"!! No data array for metric '{metric}'.")
        return pd.DataFrame(columns=["ts", out_col])

    df = pd.DataFrame(data, columns=["ts", out_col])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df

def build_master(days: int, interval: str, region: str) -> pd.DataFrame:
    start_iso, end_iso = _window(days)
    print(f"> Pulling {days} days for {region} from {start_iso} to {end_iso} at {interval} resolution.")

    sess = _session()

    frames = []
    for metric, (group, out_col) in METRICS.items():
        print(f"… fetching {metric:<32} -> {out_col:<28} [{group}]")
        df_m = _get_metric(sess, group, metric, region, start_iso, end_iso, interval, out_col)
        frames.append(df_m)

    # Outer-join everything on timestamp, then sort
    master = None
    for df in frames:
        master = df if master is None else master.merge(df, on="ts", how="outer")
    if master is None or master.empty:
        raise RuntimeError("No data retrieved. Check API key, region, or interval.")
    master = master.sort_values("ts").reset_index(drop=True)

    # Derived columns
    with pd.option_context("mode.use_inf_as_na", True):
        if {"emissions_t", "energy_mwh"}.issubset(master.columns):
            master["intensity_g_per_kwh"] = (master["emissions_t"] / master["energy_mwh"]) * 1000.0
        else:
            master["intensity_g_per_kwh"] = pd.NA

        if "price_$perMWh" in master.columns:
            master["price_$per_kwh"] = master["price_$perMWh"] / 1000.0
        else:
            master["price_$per_kwh"] = pd.NA

        if "renewable_proportion_pct" in master.columns:
            master["renewable_share"] = master["renewable_proportion_pct"] / 100.0
        else:
            master["renewable_share"] = pd.NA

    # Local time helpers
    master["local_time"] = master["ts"].dt.tz_convert("Australia/Melbourne")
    master["local_compact"] = master["local_time"].dt.strftime("%d-%b %H:%M")

    return master

def parse_args():
    p = argparse.ArgumentParser(description="Fetch OpenElectricity 90-day master dataset with correct endpoints.")
    p.add_argument("--days", type=int, default=90, help="Number of days to fetch, default 90.")
    p.add_argument("--interval", type=str, default="1h", help="Interval, e.g., 1h or 30m. Default 1h.")
    p.add_argument("--region", type=str, default="NEM", help="Network region, default NEM.")
    p.add_argument("--out", type=str, default="data/processed/openelectricity_master_90d.csv", help="Output CSV path.")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_master(days=args.days, interval=args.interval, region=args.region)
    df.to_csv(out_path, index=False)
    print(f">> Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
