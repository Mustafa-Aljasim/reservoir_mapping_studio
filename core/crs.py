"""Coordinate reference system helpers."""

from __future__ import annotations

from dataclasses import dataclass

from utils.units import coordinate_unit_symbol


LOCAL_CRS_MODE = "Local / Unknown XY"
EPSG_32638_CRS_MODE = "EPSG:32638 - WGS 84 / UTM zone 38N"
CUSTOM_EPSG_CRS_MODE = "Custom EPSG"
LEGACY_EPSG_CRS_MODE = "EPSG Code"
EPSG_CRS_MODE = CUSTOM_EPSG_CRS_MODE
CRS_MODE_OPTIONS = (LOCAL_CRS_MODE, EPSG_32638_CRS_MODE, CUSTOM_EPSG_CRS_MODE)
EPSG_32638 = 32638
EPSG_32638_NAME = "WGS 84 / UTM zone 38N"


@dataclass(frozen=True)
class CRSInfo:
    mode: str
    epsg: int | None = None
    name: str = ""
    authority: str = ""

    @property
    def is_known(self) -> bool:
        return self.epsg is not None and self.mode in {
            EPSG_32638_CRS_MODE,
            CUSTOM_EPSG_CRS_MODE,
            LEGACY_EPSG_CRS_MODE,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "epsg": self.epsg,
            "name": self.name,
            "authority": self.authority,
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
    )


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
    selected_mode = mode if mode in {EPSG_32638_CRS_MODE, CUSTOM_EPSG_CRS_MODE} else CUSTOM_EPSG_CRS_MODE
    return CRSInfo(mode=selected_mode, epsg=epsg, name=crs.name or authority_text, authority=authority_text)


def normalize_crs_config(config: dict[str, object] | None) -> dict[str, object]:
    if not config:
        return local_crs().to_dict()
    mode = str(config.get("mode") or LOCAL_CRS_MODE)
    if mode == EPSG_32638_CRS_MODE:
        return preset_crs_32638().to_dict()
    if mode in {EPSG_CRS_MODE, CUSTOM_EPSG_CRS_MODE, LEGACY_EPSG_CRS_MODE} or config.get("epsg"):
        if config.get("epsg") in (None, ""):
            return local_crs().to_dict()
        return validate_epsg(config.get("epsg"), mode=CUSTOM_EPSG_CRS_MODE).to_dict()
    if mode != LOCAL_CRS_MODE:
        return local_crs().to_dict()
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
