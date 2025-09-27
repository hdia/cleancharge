# src/zzfill_market_metrics.py
from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

BASE = os.environ.get("OPENELECTRICITY_API_BASE", "https://api.openelectricity.org.au")
NETWORK = "NEM"  # case-sensitive
INTERVAL_DEFAULT = "1h"

# Endpoint split per API: market vs data
MARKET_METRICS = {
    "price",
    "demand",
    "demand_energy",
    "curtailment",
    "curtailment_energy",
    "curtailment_solar_utility",
    "curtailment_solar_utility_energy",
    "curtailment_wind",
    "curtailment_wind_energy",
}
DATA_METRICS = {
    "power",
    "energy",
    "emissions",
    "market_value",
    "renewable_proportion",
    "storage_battery",
    "pollution",
}

@dataclass
class Opts:
    inp: str
    out: str
    metrics: List[str]
    tz: str = "Australia/Melbourne"
    interval: str = INTERVAL_DEFAULT
    max_retries: int = 5
    timeout_s: int = 60
    backoff: List[int] = None

    def __post_init__(self):
        if self.backoff is None:
            self.backoff = [2, 3, 5, 8, 13]


def load_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENELECTRICITY_API_KEY")
    if not key:
        raise SystemExit("!! OPENELECTRICITY_API_KEY is missing. Add it to your .env")
    return key


def read_master(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # detect UTC timestamp column
    for c in ["timestamp", "ts_utc", "utc_time", "time_utc", "time", "ts"]:
        if c in df.columns:
            df = df.rename(columns={c: "timestamp"})
            break
    if "timestamp" not in df.columns:
        raise SystemExit(f"!! Could not find a UTC timestamp column in {path}. Columns: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def day_span(df: pd.DataFrame) -> List[dt.date]:
    start = df["timestamp"].min().date()
    end = df["timestamp"].max().date()
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += dt.timedelta(days=1)
    return days


def endpoint_for(metric: str) -> str:
    if metric in MARKET_METRICS:
        return f"{BASE}/v4/market/network/{NETWORK}"
    elif metric in DATA_METRICS:
        return f"{BASE}/v4/data/network/{NETWORK}"
    else:
        # default to data; API will 422 if unsupported
        return f"{BASE}/v4/data/network/{NETWORK}"


def fetch_day(session: requests.Session, api_key: str, metric: str, day: dt.date, interval: str, timeout_s: int) -> Dict[pd.Timestamp, float]:
    url = endpoint_for(metric)
    params = {
        "metrics": metric,
        "interval": interval,
        "dateStart": day.isoformat(),
        "dateEnd": (day + dt.timedelta(days=1)).isoformat(),
        "pageSize": 1000,
        "page": 1,
    }
    r = session.get(url, headers={"Authorization": f"Bearer {api_key}"}, params=params, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    if not j.get("success", False):
        return {}
    out: Dict[pd.Timestamp, float] = {}
    for block in j.get("data", []):
        for res in block.get("results", []):
            for ts_str, val in res.get("data", []):
                if val is None:
                    continue
                ts = pd.to_datetime(ts_str, utc=True)
                out[ts] = float(val)
    return out


def fetch_metric_with_retries(metric: str, days: List[dt.date], api_key: str, interval: str,
                              max_retries: int, backoff: List[int], timeout_s: int) -> pd.DataFrame:
    out: Dict[pd.Timestamp, float] = {}
    with requests.Session() as s:
        for d in days:
            ok = False
            for attempt in range(1, max_retries + 1):
                try:
                    rows = fetch_day(s, api_key, metric, d, interval, timeout_s)
                    if rows:
                        out.update(rows)
                    ok = True
                    break
                except requests.HTTPError as e:
                    code = getattr(e.response, "status_code", "HTTPError")
                    print(f"!! {metric} {d} attempt {attempt}/{max_retries}: {code} {e}")
                except requests.RequestException as e:
                    print(f"!! {metric} {d} attempt {attempt}/{max_retries}: {e}")
                if attempt < max_retries:
                    time.sleep(backoff[min(attempt - 1, len(backoff) - 1)])
            if not ok:
                print(f"!! {metric} {d}: giving up after {max_retries} attempts")

    if not out:
        print(f"!! {metric}: no rows retrieved.")
        return pd.DataFrame(columns=["timestamp", metric])

    df = (
        pd.DataFrame({"timestamp": list(out.keys()), metric: list(out.values())})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return df


def build_local(df_utc: pd.DataFrame, tz: str) -> pd.DataFrame:
    df = df_utc.copy()
    df["local_time"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
    cols = ["local_time"] + [c for c in df.columns if c != "local_time"]
    return df[cols]


def parse_args() -> Opts:
    p = argparse.ArgumentParser(description="Back-fill market/data metrics into existing master CSV (UTC).")
    p.add_argument("--in", dest="inp", required=True, help="Existing UTC master CSV")
    p.add_argument("--out", dest="out", required=True, help="Output UTC master CSV (overwrite)")
    p.add_argument("--metrics", default="price,demand", help="Comma-separated (e.g., price,demand)")
    p.add_argument("--interval", default=INTERVAL_DEFAULT, help="API interval (default 1h)")
    p.add_argument("--timezone", default="Australia/Melbourne", help="Local timezone for *_local.csv")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=60)
    args = p.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    return Opts(inp=args.inp, out=args.out, metrics=metrics, tz=args.timezone, interval=args.interval,
                max_retries=args.max_retries, timeout_s=args.timeout)


def main():
    api_key = load_key()
    opts = parse_args()

    print(f">> Reading UTC master: {opts.inp}")
    df_master = read_master(opts.inp)
    days = day_span(df_master)
    print(f">> Date span in UTC master: {days[0]} → {days[-1]}  ({len(days)} days)")

    for m in opts.metrics:
        print(f">> Filling metric: {m} via {'market' if m in MARKET_METRICS else 'data'} endpoint …")
        df_m = fetch_metric_with_retries(
            metric=m,
            days=days,
            api_key=api_key,
            interval=opts.interval,
            max_retries=opts.max_retries,
            backoff=opts.backoff,
            timeout_s=opts.timeout_s,
        )
        if df_m.empty:
            print(f"!! Skipping merge for {m}: no data.")
            continue
        df_master = df_master.merge(df_m, on="timestamp", how="left")

    # Save UTC + local
    os.makedirs(os.path.dirname(opts.out), exist_ok=True)
    df_master.sort_values("timestamp").to_csv(opts.out, index=False)
    print(f">> Saved UTC master: {opts.out}  ({len(df_master)} rows)")

    out_local = opts.out.replace(".csv", "_local.csv")
    df_local = build_local(df_master, opts.tz)
    df_local.to_csv(out_local, index=False)
    print(f">> Saved local master: {out_local}")
    print(df_local.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
