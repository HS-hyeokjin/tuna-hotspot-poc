import pandas as pd

from src.analysis import hotspot_grid


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                ]
            ),
            "is_set": [True, True, True],
            "lat": [-9.41, -9.43, -9.91],
            "lon": [159.96, 159.98, 159.91],
            "set_count": [1, 1, 1],
            "is_success": [True, False, True],
            "catch_total": [20.0, 0.0, 15.0],
            "cpue_per_set": [20.0, 0.0, 15.0],
        }
    )


def test_hotspot_grid_uses_requested_resolution():
    df = _sample_frame()

    fine = hotspot_grid(
        df,
        min_sets=1,
        grid_deg=0.1,
    )
    coarse = hotspot_grid(
        df,
        min_sets=1,
        grid_deg=1.0,
    )

    assert not fine.empty
    assert not coarse.empty
    assert set(fine["grid_deg"]) == {0.1}
    assert set(coarse["grid_deg"]) == {1.0}
    assert len(fine) >= len(coarse)


def test_hotspot_grid_rejects_non_positive_resolution():
    df = _sample_frame()

    try:
        hotspot_grid(
            df,
            min_sets=1,
            grid_deg=0,
        )
    except ValueError as exc:
        assert "grid_deg" in str(exc)
    else:
        raise AssertionError(
            "grid_deg=0 should raise ValueError"
        )
