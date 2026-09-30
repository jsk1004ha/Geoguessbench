"""Distances in km; public API coordinates are degrees, in lat/lon order."""
import math

EARTH_RADIUS_KM = 6371.0
RADII_KM = (1, 25, 200, 750, 2500)


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    p, q = map(math.radians, (a[0], b[0]))
    dp = q - p
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def bearing(a: tuple[float, float], b: tuple[float, float]) -> float:
    p, q = map(math.radians, (a[0], b[0]))
    dl = math.radians(b[1] - a[1])
    return math.degrees(math.atan2(math.sin(dl) * math.cos(q),
                                  math.cos(p) * math.sin(q) - math.sin(p) * math.cos(q) * math.cos(dl))) % 360


def geo_score(distance_km: float) -> float:
    """OSV-5M exponential score; NOT an exact reproduction of GeoGuessr's live scoring."""
    if not math.isfinite(distance_km) or distance_km < 0:
        raise ValueError("distance must be finite and nonnegative")
    return 5000.0 * math.exp(-distance_km / 1492.7)
