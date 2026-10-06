import numpy as np
import pandas as pd

from src.setlevel_validation import (
    run_setlevel_regression,
    run_setlevel_suite,
)


def _sample_events() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    train_dates = pd.date_range(
        "2026-01-01",
        periods=240,
        freq="D",
    )
    test_dates = pd.date_range(
        "2026-07-01",
        periods=90,
        freq="D",
    )
    dates = train_dates.append(
        test_dates
    )
    n = len(dates)

    lat = -5 + rng.normal(
        0,
        1.5,
        n,
    )
    lon = 165 + rng.normal(
        0,
        2.0,
        n,
    )
    method = np.where(
        np.arange(n) % 3 == 0,
        "PA",
        "SC",
    )
    vessel = np.where(
        np.arange(n) % 2 == 0,
        "AX",
        "CO",
    )

    signal = (
        (lat + 6) * 3
        + (lon - 163) * 1.5
        + np.where(
            method == "PA",
            8,
            0,
        )
    )
    catch = np.maximum(
        5,
        35
        + signal
        + rng.normal(
            0,
            4,
            n,
        ),
    )

    return pd.DataFrame(
        {
            "date": dates,
            "lat": lat,
            "lon": lon,
            "month": dates.month,
            "dayofyear": dates.dayofyear,
            "method": method,
            "vessel": vessel,
            "catch_total": catch,
            "has_positive_catch": True,
        }
    )


def test_run_setlevel_regression_smoke():
    df = _sample_events()

    result = run_setlevel_regression(
        df,
        spec_key="spatial_context",
        iterations=30,
    )

    metrics = result["metrics"]
    assert metrics["train_rows"] >= 200
    assert metrics["test_rows"] >= 80
    assert metrics["model_top10_lift"] > 1.0
    assert not result[
        "feature_importance"
    ].empty


def test_setlevel_suite_without_ocean():
    df = _sample_events()

    summary, results = (
        run_setlevel_suite(
            df,
            include_ocean=False,
            iterations=20,
        )
    )

    assert {
        "spatial",
        "spatial_context",
    }.issubset(
        set(summary["spec_key"])
    )
    assert (
        summary["상태"] == "완료"
    ).all()
    assert "spatial_context" in results
