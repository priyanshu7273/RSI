#!/usr/bin/env python3
"""Ingest Kite historical-data result file(s) into the per-symbol cache.

Each --file is either (a) an auto-saved tool-result .txt containing the JSON
array of day candles, or (b) a plain .json array. Multiple files (e.g. two
date chunks) are concatenated, de-duplicated by date, and written to
data_cache/<SYMBOL>.parquet in the normalized OHLCV schema.

Usage:
  python tools/ingest_kite.py --symbol RELIANCE --file a.txt b.txt [--data-dir data_cache]
"""
import argparse, json, os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rsi_backtest.config import Config
from rsi_backtest import data as d


def load_candles(path: str):
    raw = open(path, "r").read().strip()
    # tolerate a trailing/leading noise; find the JSON array
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"{path}: no JSON array found")
    return json.loads(raw[start:end + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--file", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data_cache")
    args = ap.parse_args()

    cfg = Config(data_dir=args.data_dir)
    all_candles = []
    for f in args.file:
        all_candles.extend(load_candles(f))
    if not all_candles:
        print(f"{args.symbol}: NO DATA"); sys.exit(2)
    df = d.ingest_kite_candles(all_candles, args.symbol, cfg)
    print(f"{args.symbol}: {len(df)} rows  {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()
