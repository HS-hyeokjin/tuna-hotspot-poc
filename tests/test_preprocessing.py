import math

from src.preprocessing import parse_deg_min


def test_parse_lat_south():
    value = parse_deg_min("S0925", "lat")
    assert abs(value - (-(9 + 25 / 60))) < 1e-9


def test_parse_lon_west():
    value = parse_deg_min("W17755", "lon")
    assert abs(value - (-(177 + 55 / 60))) < 1e-9


def test_invalid_coordinate():
    assert math.isnan(parse_deg_min("RE1602", "lon"))
    assert math.isnan(parse_deg_min("S0965", "lat"))
