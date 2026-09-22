#!/usr/bin/env python3
"""Load a real OurAirports snapshot into the repository's airport database."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIRPORTS_CSV = ROOT / "data" / "airports.csv"


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else AIRPORTS_CSV
    if not source.exists():
        raise SystemExit(f"source data not found: {source}")

    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "icao",
            "iata",
            "name",
            "municipality",
            "country",
            "latitude_deg",
            "longitude_deg",
            "elevation_ft",
            "airport_type",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"source CSV missing required columns: {missing}")
        for row in reader:
            print(row["icao"], row["name"], row["latitude_deg"], row["longitude_deg"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
