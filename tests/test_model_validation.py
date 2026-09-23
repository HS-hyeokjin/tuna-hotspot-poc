import pandas as pd

from src.model_validation import (
    EXPERIMENTS,
    experiment_catalog,
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
    } == set(EXPERIMENTS)

    assert set(catalog["key"]) == set(EXPERIMENTS)
