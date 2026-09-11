"""Application constants and default options."""

APP_NAME = "Reservoir Mapping Studio"

INTERNAL_ROW_ID = "__rms_row_id"
INCLUDE_COLUMN = "Include"
OUTLIER_COLUMN = "Potential_Outlier"

MANDATORY_MAPPING_KEYS = ("x", "y")

SEMANTIC_FIELDS = {
    "x": "X Coordinate",
    "y": "Y Coordinate",
    "well": "Well Name",
    "field": "Field",
    "reservoir": "Reservoir",
    "panel": "Panel",
    "layer": "Layer",
    "formation": "Formation",
    "measurement_date": "Original Measurement Date",
    "map_reference_date": "Pressure Map Reference Date",
    "well_type": "Well Type",
    "well_status": "Well Status",
}

METADATA_FILTER_ORDER = (
    "field",
    "reservoir",
    "panel",
    "layer",
    "formation",
    "well_type",
    "well_status",
)

COLUMN_ALIASES = {
    "x": (
        "X",
        "X_COORD",
        "X_COORDINATE",
        "EASTING",
        "UTM_X",
        "E",
        "LONGITUDE",
        "LON",
    ),
    "y": (
        "Y",
        "Y_COORD",
        "Y_COORDINATE",
        "NORTHING",
        "UTM_Y",
        "N",
        "LATITUDE",
        "LAT",
    ),
    "well": (
        "WELL",
        "WELL_NAME",
        "WELLNAME",
        "WELLBORE",
        "WELL_ID",
        "WELLID",
    ),
    "reservoir": ("RESERVOIR", "RES", "FORMATION"),
    "panel": ("PANEL", "COMPARTMENT", "BLOCK"),
    "layer": ("LAYER", "ZONE", "INTERVAL"),
    "formation": ("FORMATION", "FM", "RESERVOIR"),
    "measurement_date": (
        "MEASUREMENT_DATE",
        "PRESSURE_MEASUREMENT_DATE",
        "PRESSURE_DATE",
        "SURVEY_DATE",
        "TEST_DATE",
        "ORIGINAL_DATE",
        "MEASURED_DATE",
        "RECORDED_DATE",
        "DATE",
    ),
    "map_reference_date": (
        "MAP_DATE",
        "REFERENCE_DATE",
        "MAP_REFERENCE_DATE",
        "PRESSURE_REFERENCE_DATE",
        "PRESSURE_MAP_REFERENCE_DATE",
        "PRESSURE_MAP_DATE",
        "EXTRAPOLATED_DATE",
    ),
    "field": ("FIELD", "FIELD_NAME", "ASSET"),
    "well_type": ("WELL_TYPE", "TYPE", "WELL_CLASS"),
    "well_status": ("WELL_STATUS", "STATUS"),
}

COORDINATE_UNIT_OPTIONS = {
    "meters": {"label": "Meters (m)", "symbol": "m"},
    "feet": {"label": "Feet (ft)", "symbol": "ft"},
    "kilometers": {"label": "Kilometers (km)", "symbol": "km"},
}

DEFAULT_COORDINATE_UNIT = "meters"

GRID_PRESETS = {
    "Fast": (75, 75),
    "Standard": (150, 150),
    "Fine": (250, 250),
    "Custom": None,
}

INTERPOLATION_METHOD_FAMILIES = {
    "Basic": ("IDW", "Linear", "Cubic", "RBF"),
    "Geological / Surface": ("Natural Neighbor", "Minimum Curvature", "Convergent Interpolation"),
    "Geostatistical": ("Ordinary Kriging", "Universal Kriging"),
}

INTERPOLATION_METHODS = tuple(
    method
    for methods in INTERPOLATION_METHOD_FAMILIES.values()
    for method in methods
)

DUPLICATE_METHODS = ("Average", "Median", "Keep first", "Keep last")

MASK_OPTIONS = (
    "Reservoir Boundary",
    "Selected Panel Union",
    "Convex Hull",
    "Maximum Distance",
    "Reservoir Boundary + Maximum Distance",
    "Selected Panel Union + Maximum Distance",
    "Convex Hull + Maximum Distance",
    "No Mask",
)

COLOR_SCALES = (
    "Turbo",
    "Viridis",
    "Jet",
    "Rainbow",
    "Plasma",
    "Cividis",
    "RdBu",
    "Spectral",
)

RBF_KERNELS = {
    "Thin Plate Spline": "thin_plate_spline",
    "Linear": "linear",
    "Cubic": "cubic",
    "Multiquadric": "multiquadric",
    "Gaussian": "gaussian",
}

UNIT_PRESETS = ("", "psi", "bar", "m", "ft", "mD", "mD.ft", "fraction", "%")
