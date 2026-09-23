import numpy as np
import pandas as pd

from src.model_validation import (
    EXPERIMENTS,
    experiment_catalog,
    run_experiment,
    top_k_lift,
)


def test_top_k_lift_prefers_high_actual_values():
    actual = pd.Series(
        [1, 1, 1, 1, 10, 20]
    )
    score = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    )

    lift = top_k_lift(
        actual,
        score,
        fraction=0.34,
    )

    assert lift > 1.0


def test_experiment_catalog_contains_core_comparisons():
    catalog = experiment_catalog()

    assert {
        "all_full",
        "env_only",
        "school_full",
        "school_env",
        "pa_full",
        "paired_v2_full",
        "ocean_only",
        "ocean_full",
        "onboard_plus_ocean",
    }.issubset(set(EXPERIMENTS))

    assert set(catalog["key"]) == set(EXPERIMENTS)


def test_run_experiment_smoke():
    rng = np.random.default_rng(42)
    train_dates = pd.date_range(
        "2025-01-01",
        periods=240,
        freq="D",
    )
    test_dates = pd.date_range(
        "2026-01-01",
        periods=80,
        freq="D",
    )
    dates = train_dates.append(test_dates)
    n = len(dates)

    signal = rng.normal(size=n)
    success = signal > 0
    catch = np.where(
        success,
        20 + signal * 5 + rng.uniform(0, 10, n),
        0,
    )
    catch = np.maximum(catch, 0)

    df = pd.DataFrame(
        {
            "date": dates,
            "year": dates.year,
            "month": dates.month,
            "quarter": dates.quarter,
            "lat": -5 + signal,
            "lon": 165 + rng.normal(0, 2, n),
            "water_temp": 28 + signal * 0.5,
            "current": rng.normal(0.5, 0.1, n),
            "vessel": np.where(
                np.arange(n) % 2 == 0,
                "V1",
                "V2",
            ),
            "captain": np.where(
                np.arange(n) % 3 == 0,
                "C1",
                "C2",
            ),
            "fishing_ground": np.where(
                np.arange(n) % 2 == 0,
                "A",
                "B",
            ),
            "method": "school_fish",
            "is_set": True,
            "catch_total": catch,
            "catch_sj": catch * 0.8,
            "catch_yf": catch * 0.2,
            "catch_be": 0.0,
        }
    )

    result = run_experiment(
        df,
        "all_full",
        target_col="catch_total",
        iterations=20,
        with_shap=True,
    )

    assert result.metrics["test_rows"] == 80
    assert result.metrics["roc_auc"] > 0.5
    assert not result.shap_importance.empty
    assert "model_score" in result.scored_test
