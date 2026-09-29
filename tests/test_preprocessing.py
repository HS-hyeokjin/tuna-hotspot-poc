import math

import numpy as np
import pandas as pd

from src.preprocessing import (
    coordinate_audit_summary,
    coordinate_minute_distribution,
    parse_deg_min,
)


def test_parse_lat_south():
    value = parse_deg_min("S0925", "lat")
    assert abs(value - (-(9 + 25 / 60))) < 1e-9


def test_parse_lon_west():
    value = parse_deg_min("W17755", "lon")
    assert abs(value - (-(177 + 55 / 60))) < 1e-9


def test_invalid_coordinate():
    assert math.isnan(parse_deg_min("RE1602", "lon"))
    assert math.isnan(parse_deg_min("S0965", "lat"))


def test_coordinate_audit_detects_minute_precision_and_multi_set_rows():
    df = pd.DataFrame(
        {
            "lat_raw": [
                "S0925",
                "S0003",
                "N1850",
                "BAD",
            ],
            "lon_raw": [
                "E15958",
                "E15240",
                "E13922",
                "E16000",
            ],
            "lat": [
                -(9 + 25 / 60),
                -(0 + 3 / 60),
                18 + 50 / 60,
                np.nan,
            ],
            "lon": [
                159 + 58 / 60,
                152 + 40 / 60,
                139 + 22 / 60,
                160.0,
            ],
            "set_count": [1, 2, 0, 1],
            "is_set": [True, True, False, True],
        }
    )

    summary = coordinate_audit_summary(df)

    assert summary["rows"] == 4
    assert summary["valid_pairs"] == 3
    assert summary["set_rows"] == 3
    assert summary["valid_set_pairs"] == 2
    assert summary["multi_set_rows"] == 1
    assert summary["lat_minute_unique"] == 3
    assert summary["lon_minute_unique"] == 4
    assert (
        summary["encoded_lat_resolution_km"]
        > 1.8
    )

    dist = coordinate_minute_distribution(df)
    assert len(dist) == 60
    assert int(
        dist.loc[
            dist["minute"] == 25,
            "latitude_count",
        ].iloc[0]
    ) == 1
    assert int(
        dist.loc[
            dist["minute"] == 58,
            "longitude_count",
        ].iloc[0]
    ) == 1
