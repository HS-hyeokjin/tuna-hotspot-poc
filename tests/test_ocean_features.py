import pandas as pd

from src.ocean.extractor import (
    _cache_matches,
    _normalise_points,
    feature_coverage,
    merge_ocean_features,
)


def test_normalise_points_splits_dateline_blocks():
    df = pd.DataFrame(
        {
            "source_row_id": [0, 1],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-01"]
            ),
            "lat": [-5.0, -5.0],
            "lon": [179.5, -179.5],
        }
    )

    points = _normalise_points(df)

    assert len(points) == 2
    assert points["lon_band"].nunique() == 2


def test_merge_ocean_features_left_join():
    fishing = pd.DataFrame(
        {
            "source_row_id": [0, 1, 2],
            "date": pd.to_datetime(
                ["2026-01-01"] * 3
            ),
        }
    )
    features = pd.DataFrame(
        {
            "source_row_id": [0, 2],
            "ocean_sst": [29.1, 28.7],
            "ocean_current_speed": [0.4, 0.7],
        }
    )

    merged = merge_ocean_features(
        fishing,
        features,
    )

    assert len(merged) == 3
    assert merged.loc[
        merged["source_row_id"] == 1,
        "ocean_sst",
    ].isna().all()


def test_feature_coverage():
    features = pd.DataFrame(
        {
            "source_row_id": [0, 1],
            "ocean_sst": [29.0, None],
            "ocean_current_u": [0.1, 0.2],
        }
    )

    coverage = feature_coverage(features)
    row = coverage[
        coverage["feature"] == "ocean_sst"
    ].iloc[0]

    assert row["coverage_pct"] == 50.0


def test_cache_matches_requires_same_rows_and_coordinates():
    requested = pd.DataFrame(
        {
            "source_row_id": [1, 2],
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02"]
            ),
            "lat": [-5.0, -6.0],
            "lon": [170.0, 171.0],
        }
    )
    cached = requested.copy()
    cached["ocean_sst"] = [29.0, 28.0]

    assert _cache_matches(
        cached,
        requested,
    )

    changed = requested.copy()
    changed.loc[1, "lon"] = 172.0

    assert not _cache_matches(
        cached,
        changed,
    )
