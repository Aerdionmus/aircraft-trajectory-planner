"""Real-airport data layer with provenance and deterministic lookup."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

DATASET_SOURCE = "OurAirports"
DATASET_URL = "https://ourairports.com/data/airports.csv"
DATASET_DATE = "2025-01-01"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Airport:
    """A real-world airport record from the authoritative public dataset."""

    icao: str
    iata: str | None
    name: str
    municipality: str | None
    country: str | None
    latitude_deg: float
    longitude_deg: float
    elevation_ft: float | None = None
    airport_type: str | None = None
    source: str = DATASET_SOURCE
    source_url: str = DATASET_URL
    source_date: str = DATASET_DATE
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("raw", None)
        return payload

    @property
    def latitude(self) -> float:
        return self.latitude_deg

    @property
    def longitude(self) -> float:
        return self.longitude_deg


DEFAULT_AIRPORTS: tuple[Airport, ...] = (
    Airport(
        icao="VOMM",
        iata="MAA",
        name="Chennai International Airport",
        municipality="Chennai",
        country="IN",
        latitude_deg=12.9900,
        longitude_deg=80.1696,
        elevation_ft=52.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="VABB",
        iata="BOM",
        name="Chhatrapati Shivaji Maharaj International Airport",
        municipality="Mumbai",
        country="IN",
        latitude_deg=19.0896,
        longitude_deg=72.8656,
        elevation_ft=39.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="VIDP",
        iata="DEL",
        name="Indira Gandhi International Airport",
        municipality="Delhi",
        country="IN",
        latitude_deg=28.5665,
        longitude_deg=77.1031,
        elevation_ft=777.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="VOBL",
        iata="BLR",
        name="Kempegowda International Airport",
        municipality="Bengaluru",
        country="IN",
        latitude_deg=13.1989,
        longitude_deg=77.7063,
        elevation_ft=3000.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="LFPG",
        iata="CDG",
        name="Charles de Gaulle International Airport",
        municipality="Paris",
        country="FR",
        latitude_deg=49.0097,
        longitude_deg=2.5479,
        elevation_ft=392.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="LFPO",
        iata="ORY",
        name="Paris Orly Airport",
        municipality="Paris",
        country="FR",
        latitude_deg=48.7262,
        longitude_deg=2.3652,
        elevation_ft=291.0,
        airport_type="large_airport",
    ),
    Airport(
        icao="EGLL",
        iata="LHR",
        name="London Heathrow Airport",
        municipality="London",
        country="GB",
        latitude_deg=51.4700,
        longitude_deg=-0.4543,
        elevation_ft=83.0,
        airport_type="large_airport",
    ),
)


class AirportDatabase:
    """Deterministic airport lookup with provenance metadata."""

    def __init__(self, records: Iterable[Airport] | None = None) -> None:
        self._records = tuple(records or DEFAULT_AIRPORTS)
        self._by_icao = {airport.icao.upper(): airport for airport in self._records}
        self._by_iata = {airport.iata.upper(): airport for airport in self._records if airport.iata}
        self._by_name = {
            airport.name.upper(): airport for airport in self._records
        }
        self.source = DATASET_SOURCE
        self.source_url = DATASET_URL
        self.source_date = DATASET_DATE

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "AirportDatabase":
        path = Path(csv_path)
        records: list[Airport] = []
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                airport = Airport(
                    icao=(row.get("icao") or "").strip(),
                    iata=(row.get("iata") or "").strip() or None,
                    name=(row.get("name") or "").strip(),
                    municipality=(row.get("municipality") or "").strip() or None,
                    country=(row.get("country") or "").strip() or None,
                    latitude_deg=float(row["latitude_deg"]),
                    longitude_deg=float(row["longitude_deg"]),
                    elevation_ft=float(row["elevation_ft"]) if row.get("elevation_ft") else None,
                    airport_type=(row.get("airport_type") or "").strip() or None,
                    source=(row.get("source") or DATASET_SOURCE).strip() or DATASET_SOURCE,
                    source_url=(row.get("source_url") or DATASET_URL).strip() or DATASET_URL,
                    source_date=(row.get("source_date") or DATASET_DATE).strip() or DATASET_DATE,
                    raw=dict(row),
                )
                records.append(airport)
        return cls(records)

    @classmethod
    def default(cls) -> "AirportDatabase":
        csv_path = _repo_root() / "data" / "airports.csv"
        if csv_path.exists():
            return cls.from_csv(csv_path)
        return cls(DEFAULT_AIRPORTS)

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def lookup(self, code_or_name: str) -> Airport:
        if not isinstance(code_or_name, str):
            raise KeyError("airport lookup requires a string value")
        token = code_or_name.strip()
        if not token:
            raise KeyError("airport lookup requires a non-empty code or name")
        normalised = token.upper()
        if normalised in self._by_icao:
            return self._by_icao[normalised]
        if normalised in self._by_iata:
            return self._by_iata[normalised]
        if normalised in self._by_name:
            return self._by_name[normalised]
        matches = [airport for airport in self._records if normalised in airport.name.upper()]
        if matches:
            return matches[0]
        raise KeyError(f"airport not found: {code_or_name!r}")

    def search(self, query: str) -> list[Airport]:
        if not query:
            return []
        token = query.strip().upper()
        matches: list[Airport] = []
        for airport in self._records:
            haystack = " ".join(
                part for part in (airport.icao, airport.iata or "", airport.name, airport.municipality or "", airport.country or "")
            ).upper()
            if token in haystack:
                matches.append(airport)
        return matches

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_url": self.source_url,
            "source_date": self.source_date,
            "airports": [airport.as_dict() for airport in self._records],
        }


def resolve_airport(code_or_name: str) -> Airport:
    return default_airport_database().lookup(code_or_name)


def default_airport_database() -> AirportDatabase:
    return AirportDatabase.default()
