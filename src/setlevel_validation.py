from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
)

from .model_validation import (
    OCEAN_FEATURES,
    top_k_lift,
)


@dataclass(frozen=True)
class SetLevelSpec:
    key: str
    label: str
    description: str
    features: tuple[str, ...]
    cat_features: tuple[str, ...]
    required_non_null: tuple[str, ...] = ()


SETLEVEL_SPECS: dict[str, SetLevelSpec] = {
    "spatial": SetLevelSpec(
        key="spatial",
        label="위치·계절·방법",
        description=(
            "정확한 이벤트 좌표, 계절, 조업방법만 사용"
        ),
        features=(
            "lat",
            "lon",
            "month",
            "dayofyear",
            "method",
        ),
        cat_features=("method",),
    ),
    "spatial_context": SetLevelSpec(
        key="spatial_context",
        label="위치·계절·방법 + 선박",
        description=(
            "이벤트 좌표와 계절에 선박 컨텍스트를 추가"
        ),
        features=(
            "lat",
            "lon",
            "month",
            "dayofyear",
            "method",
            "vessel",
        ),
        cat_features=(
            "method",
            "vessel",
        ),
    ),
    "spatial_context_ocean": SetLevelSpec(
        key="spatial_context_ocean",
        label="위치·계절·방법·선박 + Ocean",
        description=(
            "set-level 이벤트에 Copernicus 일자료를 결합한 실험"
        ),
        features=(
            "lat",
            "lon",
            "month",
            "dayofyear",
            "method",
            "vessel",
            *OCEAN_FEATURES,
        ),
        cat_features=(
            "method",
            "vessel",
        ),
        required_non_null=(
            "ocean_sst",
            "ocean_current_speed",
            "ocean_ssh",
            "ocean_chl",
        ),
    ),
}


def _prepare(
    df: pd.DataFrame,
    spec: SetLevelSpec,
    target_col: str,
) -> pd.DataFrame:
    required = [
        "date",
        "has_positive_catch",
        target_col,
        *spec.features,
        *spec.required_non_null,
    ]
    missing = [
        col
        for col in required
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            "필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    work = df[
        df["has_positive_catch"]
    ].copy()
    work = work.dropna(
        subset=[
            "date",
            "lat",
            "lon",
            target_col,
            *spec.required_non_null,
        ]
    )

    for col in spec.cat_features:
        work[col] = (
            work[col]
            .astype("string")
            .fillna("UNKNOWN")
            .astype(str)
        )

    numeric_features = [
        col
        for col in spec.features
        if col not in spec.cat_features
    ]
    for col in numeric_features:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work[target_col] = pd.to_numeric(
        work[target_col],
        errors="coerce",
    ).fillna(0.0)

    return (
        work.sort_values("date")
        .reset_index(drop=True)
    )


def _split(
    work: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    cutoff = pd.Timestamp("2026-07-01")
    train = work[
        work["date"] < cutoff
    ].copy()
    test = work[
        work["date"] >= cutoff
    ].copy()

    if (
        len(train) >= 200
        and len(test) >= 80
    ):
        return (
            train,
            test,
            "2026-01~06 학습 / 2026-07~09 검증",
        )

    cut = max(
        int(len(work) * 0.75),
        1,
    )
    return (
        work.iloc[:cut].copy(),
        work.iloc[cut:].copy(),
        "시간순 75% 학습 / 최근 25% 검증",
    )


def _rank_corr(
    actual: pd.Series,
    predicted: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(
                actual,
                errors="coerce",
            ),
            "predicted": pd.to_numeric(
                predicted,
                errors="coerce",
            ),
        }
    ).dropna()

    if (
        len(frame) < 3
        or frame["actual"].nunique() < 2
        or frame["predicted"].nunique() < 2
    ):
        return np.nan

    return float(
        frame["actual"]
        .rank(method="average")
        .corr(
            frame["predicted"]
            .rank(method="average")
        )
    )


def _method_median_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    global_median = float(
        train[target_col].median()
    )
    medians = (
        train.groupby("method")[
            target_col
        ]
        .median()
        .to_dict()
    )

    return (
        test["method"]
        .map(medians)
        .fillna(global_median)
        .to_numpy(dtype=float)
    )


def run_setlevel_regression(
    df: pd.DataFrame,
    spec_key: str = "spatial_context",
    target_col: str = "catch_total",
    iterations: int = 220,
) -> dict[str, Any]:
    if spec_key not in SETLEVEL_SPECS:
        raise KeyError(spec_key)

    spec = SETLEVEL_SPECS[spec_key]
    work = _prepare(
        df,
        spec,
        target_col,
    )
    train, test, split_note = _split(work)

    if len(train) < 100 or len(test) < 30:
        raise ValueError(
            "set-level 회귀 검증 표본이 부족합니다."
        )

    features = list(spec.features)
    cat_features = list(
        spec.cat_features
    )

    model = CatBoostRegressor(
        iterations=iterations,
        depth=7,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train[features],
        np.log1p(
            train[target_col]
        ),
        cat_features=cat_features,
    )

    scored = test.copy()
    scored["prediction"] = np.maximum(
        np.expm1(
            model.predict(
                scored[features]
            )
        ),
        0.0,
    )
    scored["baseline_prediction"] = (
        _method_median_baseline(
            train,
            scored,
            target_col,
        )
    )

    actual = scored[target_col]

    metrics = {
        "spec_key": spec.key,
        "model": spec.label,
        "split": split_note,
        "rows": int(len(work)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline_mae": float(
            mean_absolute_error(
                actual,
                scored[
                    "baseline_prediction"
                ],
            )
        ),
        "model_mae": float(
            mean_absolute_error(
                actual,
                scored["prediction"],
            )
        ),
        "baseline_medae": float(
            median_absolute_error(
                actual,
                scored[
                    "baseline_prediction"
                ],
            )
        ),
        "model_medae": float(
            median_absolute_error(
                actual,
                scored["prediction"],
            )
        ),
        "rank_corr": _rank_corr(
            actual,
            scored["prediction"],
        ),
        "top10_lift": top_k_lift(
            actual,
            scored["prediction"],
            fraction=0.10,
        ),
        "top20_lift": top_k_lift(
            actual,
            scored["prediction"],
            fraction=0.20,
        ),
    }
    metrics["delta_mae"] = (
        metrics["model_mae"]
        - metrics["baseline_mae"]
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": (
                    model.get_feature_importance()
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "spec": spec,
        "model": model,
        "metrics": metrics,
        "feature_importance": (
            feature_importance
        ),
        "scored_test": scored,
    }


def run_setlevel_suite(
    df: pd.DataFrame,
    include_ocean: bool = False,
    target_col: str = "catch_total",
    iterations: int = 220,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    keys = [
        "spatial",
        "spatial_context",
    ]
    if include_ocean:
        keys.append(
            "spatial_context_ocean"
        )

    results: dict[str, dict] = {}
    rows: list[dict] = []

    for key in keys:
        try:
            result = run_setlevel_regression(
                df,
                spec_key=key,
                target_col=target_col,
                iterations=iterations,
            )
            results[key] = result
            rows.append(
                {
                    **result["metrics"],
                    "상태": "완료",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "spec_key": key,
                    "model": (
                        SETLEVEL_SPECS[
                            key
                        ].label
                    ),
                    "상태": f"실패: {exc}",
                }
            )

    return pd.DataFrame(rows), results
