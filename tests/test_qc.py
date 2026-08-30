import pandas as pd

from core.data_qc import OUTLIER_COLUMN, build_qc_summary, duplicate_coordinate_rows, flag_outliers


def test_qc_detects_missing_coordinates_and_duplicates():
    df = pd.DataFrame(
        {
            "X": [0.0, 0.0, None, 3.0],
            "Y": [1.0, 1.0, 2.0, None],
            "Pressure": [100.0, 102.0, 103.0, 104.0],
            "Well": ["A", "B", "C", "D"],
        }
    )
    qc = build_qc_summary(df, "X", "Y", "Pressure", "Well")
    assert qc.missing_x == 1
    assert qc.missing_y == 1
    assert qc.duplicate_xy_rows == 2
    assert qc.unique_wells == 4
    assert len(duplicate_coordinate_rows(df, "X", "Y")) == 2


def test_outlier_flag_is_generated():
    df = pd.DataFrame({"Pressure": [100, 101, 99, 102, 98, 100, 500]})
    flagged = flag_outliers(df, "Pressure")
    assert flagged[OUTLIER_COLUMN].sum() == 1
    assert flagged.loc[6, OUTLIER_COLUMN]

