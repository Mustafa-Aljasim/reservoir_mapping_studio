"""Coordinate reference system helpers."""

from __future__ import annotations

from dataclasses import dataclass


LOCAL_CRS_MODE = "Local / Unknown XY"
EPSG_CRS_MODE = "EPSG Code"


@dataclass(frozen=True)
class CRSInfo:
    mode: str
    epsg: int | None = None
    name: str = ""
    authority: str = ""

    @property
    def is_known(self) -> bool:
        return self.mode == EPSG_CRS_MODE and self.epsg is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "epsg": self.epsg,
            "name": self.name,
            "authority": self.authority,
        }


def local_crs() -> CRSInfo:
    return CRSInfo(mode=LOCAL_CRS_MODE, name="Local / Unknown XY")


def validate_epsg(epsg_value: int | str | None) -> CRSInfo:
    """Resolve and validate an EPSG code with pyproj."""

    try:
        epsg = int(str(epsg_value).replace("EPSG:", "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("EPSG code must be a whole number, for example 32638.") from exc
    if epsg <= 0:
        raise ValueError("EPSG code must be a positive number.")
    try:
        from pyproj import CRS
    except ImportError as exc:  # pragma: no cover - dependency is installed in V1 requirements
        raise ValueError("pyproj is required to validate EPSG codes.") from exc

    try:
        crs = CRS.from_epsg(epsg)
    except Exception as exc:
        raise ValueError(f"EPSG:{epsg} could not be resolved.") from exc
    authority = crs.to_authority()
    authority_text = f"{authority[0]}:{authority[1]}" if authority else f"EPSG:{epsg}"
    return CRSInfo(mode=EPSG_CRS_MODE, epsg=epsg, name=crs.name or authority_text, authority=authority_text)


def normalize_crs_config(config: dict[str, object] | None) -> dict[str, object]:
    if not config:
        return local_crs().to_dict()
    mode = str(config.get("mode") or LOCAL_CRS_MODE)
    if mode != EPSG_CRS_MODE:
        return local_crs().to_dict()
    return validate_epsg(config.get("epsg")).to_dict()


def crs_display_name(config: dict[str, object] | None) -> str:
    normalized = normalize_crs_config(config)
    if normalized["mode"] != EPSG_CRS_MODE:
        return LOCAL_CRS_MODE
    authority = normalized.get("authority") or f"EPSG:{normalized.get('epsg')}"
    name = normalized.get("name") or authority
    return f"{authority} - {name}"


def crs_are_compatible(first: dict[str, object] | None, second: dict[str, object] | None) -> bool:
    left = normalize_crs_config(first)
    right = normalize_crs_config(second)
    if left["mode"] != right["mode"]:
        return False
    if left["mode"] == EPSG_CRS_MODE:
        return int(left["epsg"]) == int(right["epsg"])
    return True
