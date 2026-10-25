"""Parse and validate DJI-style SRT cues without associating video positions.

UTF-8 (with or without BOM) is supported. Additional telemetry text is allowed.
Coordinates are selected from the nearest cue start, never interpolated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class SrtValidationError(ValueError):
    """The SRT text, cue structure, timing, or coordinate data is invalid."""


TIME_RE = re.compile(
    r"(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{1,3})\s*-->\s*"
    r"(\d{2,}):([0-5]\d):([0-5]\d)[,.](\d{1,3})"
)
NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
COORD_RE = re.compile(r"\b(latitude|longitude|longtude|long)\s*:\s*([^\s\]<>;,]*)", re.IGNORECASE)
ALT_RE = re.compile(r"\b(?:rel_alt|altitude|abs_alt)\s*:\s*([^\s\]<>;,]+)", re.IGNORECASE)
GPS_MARKER_RE = re.compile(r"\bGPS\s*\(", re.IGNORECASE)
GPS_RE = re.compile(r"\bGPS\s*\(([^)]*)\)", re.IGNORECASE)


@dataclass(frozen=True)
class TelemetryPoint:
    t: float
    lat: float
    lon: float
    alt: float | None


def _hms_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _coordinate(value, limit, cue_number):
    if not NUMBER_RE.fullmatch(value):
        raise SrtValidationError(f"Cue {cue_number} contains an invalid coordinate.")
    number = float(value)
    if not math.isfinite(number) or not -limit <= number <= limit:
        raise SrtValidationError(f"Cue {cue_number} contains an out-of-range coordinate.")
    return number


def _optional_altitude(value):
    if value is not None and NUMBER_RE.fullmatch(value):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _cue_coordinate(payload, cue_number):
    fields = {}
    for match in COORD_RE.finditer(payload):
        key = "lat" if match[1].lower() == "latitude" else "lon"
        value = _coordinate(match[2], 90 if key == "lat" else 180, cue_number)
        if key in fields:
            raise SrtValidationError(f"Cue {cue_number} contains duplicate coordinate fields.")
        fields[key] = value

    gps_matches = list(GPS_RE.finditer(payload))
    if len(gps_matches) != len(GPS_MARKER_RE.findall(payload)):
        raise SrtValidationError(f"Cue {cue_number} contains malformed GPS data.")
    if len(gps_matches) > 1:
        raise SrtValidationError(f"Cue {cue_number} contains multiple GPS records.")
    gps = None
    if gps_matches:
        parts = [part.strip() for part in gps_matches[0][1].split(",")]
        if len(parts) not in (2, 3):
            raise SrtValidationError(f"Cue {cue_number} contains malformed GPS data.")
        gps = (_coordinate(parts[1], 90, cue_number), _coordinate(parts[0], 180, cue_number),
               _optional_altitude(parts[2]) if len(parts) == 3 else None)

    if fields:
        if set(fields) != {"lat", "lon"}:
            raise SrtValidationError(f"Cue {cue_number} must contain both latitude and longitude.")
        if gps and (fields["lat"], fields["lon"]) != gps[:2]:
            raise SrtValidationError(f"Cue {cue_number} contains conflicting GPS records.")
        altitude = ALT_RE.search(payload)
        return fields["lat"], fields["lon"], _optional_altitude(altitude[1] if altitude else None)
    return gps


def parse_srt(path: str) -> list[TelemetryPoint]:
    """Validate every cue and return records; one usable GPS record is enough."""
    try:
        with open(path, "r", encoding="utf-8-sig") as source:
            content = source.read()
    except UnicodeError as exc:
        raise SrtValidationError("SRT must be readable UTF-8 text (a BOM is permitted).") from exc
    except OSError as exc:
        raise SrtValidationError("SRT file could not be read.") from exc
    if not content.strip():
        raise SrtValidationError("SRT file is empty.")
    if "\x00" in content:
        raise SrtValidationError("SRT contains invalid NUL characters.")

    points = []
    for cue_number, block in enumerate(re.split(r"\n(?:[ \t]*\n)+", content.strip()), start=1):
        lines = block.strip().splitlines()
        if (len(lines) < 3 or not re.fullmatch(r"[0-9]+", lines[0].strip())
                or not lines[0].strip().strip("0")):
            raise SrtValidationError(f"Cue {cue_number} must contain a number, timing line, and telemetry text.")
        timing = TIME_RE.fullmatch(lines[1].strip())
        if timing is None:
            raise SrtValidationError(f"Cue {cue_number} has invalid timing.")
        try:
            start = _hms_to_seconds(*timing.groups()[:4])
            end = _hms_to_seconds(*timing.groups()[4:])
        except (ValueError, OverflowError) as exc:
            raise SrtValidationError(f"Cue {cue_number} has invalid timing.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise SrtValidationError(f"Cue {cue_number} must end after it starts.")
        payload = "\n".join(lines[2:]).strip()
        if not payload:
            raise SrtValidationError(f"Cue {cue_number} has no telemetry text.")
        coordinate = _cue_coordinate(payload, cue_number)
        if coordinate is not None:
            lat, lon, alt = coordinate
            points.append(TelemetryPoint(t=start, lat=lat, lon=lon, alt=alt))
    if not points:
        raise SrtValidationError("No usable latitude/longitude record was found in the SRT.")
    points.sort(key=lambda point: point.t)
    return points


def coord_at(points: list[TelemetryPoint], t: float) -> tuple[float, float]:
    """Nearest cue start; ties prefer earlier time, then original file order."""
    if not points or not math.isfinite(t) or t < 0:
        raise SrtValidationError("A usable video timestamp and telemetry record are required.")
    point = min(points, key=lambda point: (abs(point.t - t), point.t))
    return point.lat, point.lon
