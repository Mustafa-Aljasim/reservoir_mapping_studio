"""Coordinate reference system helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from utils.units import coordinate_unit_symbol


LOCAL_CRS_MODE = "Local / Unknown XY"
EPSG_32638_CRS_MODE = "EPSG:32638 - WGS 84 / UTM zone 38N"
CUSTOM_EPSG_CRS_MODE = "Custom EPSG"
LEGACY_EPSG_CRS_MODE = "EPSG Code"
EPSG_CRS_MODE = CUSTOM_EPSG_CRS_MODE
CRS_LIBRARY_MODE = "CRS Library"
CRS_MODE_OPTIONS = (LOCAL_CRS_MODE, CRS_LIBRARY_MODE, CUSTOM_EPSG_CRS_MODE)
EPSG_32638 = 32638
EPSG_32638_NAME = "WGS 84 / UTM zone 38N"


@dataclass(frozen=True)
class CRSInfo:
    mode: str
    epsg: int | None = None
    name: str = ""
    authority: str = ""
    coordinate_unit: str = ""
    crs_type: str = ""

    @property
    def is_known(self) -> bool:
        return self.epsg is not None and self.mode in {
            EPSG_32638_CRS_MODE,
            CUSTOM_EPSG_CRS_MODE,
            LEGACY_EPSG_CRS_MODE,
            CRS_LIBRARY_MODE,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "epsg": self.epsg,
            "name": self.name,
            "authority": self.authority,
            "coordinate_unit": self.coordinate_unit,
            "crs_type": self.crs_type,
        }


def local_crs() -> CRSInfo:
    return CRSInfo(mode=LOCAL_CRS_MODE, name="Local / Unknown XY")


def preset_crs_32638() -> CRSInfo:
    resolved = validate_epsg(EPSG_32638, mode=EPSG_32638_CRS_MODE)
    return CRSInfo(
        mode=EPSG_32638_CRS_MODE,
        epsg=EPSG_32638,
        name=resolved.name or EPSG_32638_NAME,
        authority="EPSG:32638",
        coordinate_unit=resolved.coordinate_unit,
        crs_type=resolved.crs_type,
    )


@lru_cache(maxsize=256)
def validate_epsg(epsg_value: int | str | None, mode: str | None = None) -> CRSInfo:
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
    selected_mode = mode if mode in {EPSG_32638_CRS_MODE, CUSTOM_EPSG_CRS_MODE, CRS_LIBRARY_MODE} else CUSTOM_EPSG_CRS_MODE
    return CRSInfo(mode=selected_mode, epsg=epsg, name=crs.name or authority_text, authority=authority_text,
                   coordinate_unit=crs.axis_info[0].unit_name if crs.axis_info else "", crs_type=crs.type_name)


def normalize_crs_config(config: dict[str, object] | None) -> dict[str, object]:
    if not config:
        return local_crs().to_dict()
    mode = str(config.get("mode") or LOCAL_CRS_MODE)
    if config.get("mode") == LOCAL_CRS_MODE and config.get("epsg") not in (None, ""):
        raise ValueError("Local / Unknown XY cannot specify an EPSG code.")
    if mode == EPSG_32638_CRS_MODE:
        if config.get("epsg") not in (None, "", EPSG_32638, str(EPSG_32638)):
            raise ValueError("CRS preset and EPSG code disagree.")
        return preset_crs_32638().to_dict()
    if mode in {EPSG_CRS_MODE, CUSTOM_EPSG_CRS_MODE, LEGACY_EPSG_CRS_MODE, CRS_LIBRARY_MODE} or config.get("epsg"):
        if config.get("epsg") in (None, ""):
            raise ValueError("A known CRS requires an explicit EPSG code.")
        return validate_epsg(config.get("epsg"), mode=mode).to_dict()
    if mode != LOCAL_CRS_MODE:
        return local_crs().to_dict()
    return local_crs().to_dict()


@lru_cache(maxsize=1)
def _crs_catalog():
    from pyproj.database import query_crs_info
    return tuple((int(info.code), info.name, f"epsg:{info.code} {info.name}".casefold())
                 for info in query_crs_info(auth_name="EPSG", allow_deprecated=False))


def search_crs(query: str, limit: int = 50) -> list[dict[str, object]]:
    """Search one cached PROJ catalog; bound the options sent to the browser."""
    terms = query.casefold().strip().split()
    if not terms:
        return [validate_epsg(EPSG_32638, CRS_LIBRARY_MODE).to_dict()]
    matches = [(code, name) for code, name, text in _crs_catalog() if all(term in text for term in terms)]
    matches.sort(key=lambda item: (str(item[0]) != query.strip().removeprefix("EPSG:"), item[0] != EPSG_32638, item[0]))
    return [{"epsg": code, "name": name, "label": f"EPSG:{code} — {name}"}
            for code, name in matches[:max(0, min(limit, 100))]]


def map_crs_snapshot(map_result: dict[str, object]) -> dict[str, object]:
    """Resolve only map-owned state and reject contradictory saved metadata."""
    seed = map_result.get("export_metadata") or map_result.get("metadata") or {}
    snapshot = map_result.get("crs")
    metadata_epsg = seed.get("CRS_EPSG") or seed.get("EPSG")
    if seed.get("CRS_EPSG") not in (None, "") and seed.get("EPSG") not in (None, ""):
        if validate_epsg(seed["CRS_EPSG"]).epsg != validate_epsg(seed["EPSG"]).epsg:
            raise ValueError("Map export metadata contains conflicting EPSG codes.")
    if snapshot:
        resolved = normalize_crs_config(snapshot)
        if seed.get("CRS_Mode") == LOCAL_CRS_MODE and resolved.get("epsg"):
            raise ValueError("Map CRS snapshot disagrees with local export metadata. Regenerate the map.")
        if metadata_epsg not in (None, "") and resolved.get("epsg") != validate_epsg(metadata_epsg).epsg:
            raise ValueError("Map CRS snapshot disagrees with its export metadata. Regenerate the map.")
        return resolved
    if metadata_epsg not in (None, "") or seed.get("CRS_Mode"):
        return normalize_crs_config({"mode": seed.get("CRS_Mode") or CUSTOM_EPSG_CRS_MODE, "epsg": metadata_epsg})
    return local_crs().to_dict()


def crs_display_name(config: dict[str, object] | None) -> str:
    normalized = normalize_crs_config(config)
    if not normalized.get("epsg"):
        return LOCAL_CRS_MODE
    authority = normalized.get("authority") or f"EPSG:{normalized.get('epsg')}"
    name = normalized.get("name") or authority
    return f"{authority} - {name}"


def crs_are_compatible(first: dict[str, object] | None, second: dict[str, object] | None) -> bool:
    left = normalize_crs_config(first)
    right = normalize_crs_config(second)
    if left.get("epsg") or right.get("epsg"):
        return bool(left.get("epsg") and right.get("epsg") and int(left["epsg"]) == int(right["epsg"]))
    if left["mode"] != right["mode"]:
        return False
    return True


def crs_coordinate_unit_warning(config: dict[str, object] | None, coordinate_unit: str | None) -> str:
    """Return a CRS/unit warning without converting coordinates."""

    normalized = normalize_crs_config(config)
    unit_symbol = coordinate_unit_symbol(coordinate_unit)
    if int(normalized.get("epsg") or 0) == EPSG_32638 and unit_symbol != "m":
        return (
            "EPSG:32638 is a projected CRS defined in meters, while the current project "
            f"Coordinate Unit is {unit_symbol}. Verify the project configuration."
        )
    return ""
