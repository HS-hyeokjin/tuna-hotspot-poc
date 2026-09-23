from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    median_absolute_error,
    roc_auc_score,
)


ENV_FEATURES = [
    "lat",
    "lon",
    "water_temp",
    "current",
    "month",
    "quarter",
]

CONTEXT_FEATURES = [
    "vessel",
    "captain",
    "fishing_ground",
]

OCEAN_FEATURES = [
    "ocean_sst",
    "ocean_current_u",
    "ocean_current_v",
    "ocean_current_speed",
    "ocean_ssh",
    "ocean_chl",
    "ocean_sst_gradient_c_per_100km",
]

OCEAN_ENV_FEATURES = [
    "lat",
    "lon",
    "month",
    "quarter",
    *OCEAN_FEATURES,
]

SPECIES_TARGETS = {
    "전체 어획": "catch_total",
    "S/J": "catch_sj",
    "Y/F": "catch_yf",
    "B/E": "catch_be",
}


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    label: str
    description: str
    method_filter: str | None
    features: tuple[str, ...]
    cat_features: tuple[str, ...]


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    target_col: str
    classifier: Any | None
    regressor: Any | None
    metrics: dict
    feature_importance: pd.DataFrame
    shap_importance: pd.DataFrame
    shap_detail: pd.DataFrame
    scored_test: pd.DataFrame
    split_note: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "all_full": ExperimentSpec(
        key="all_full",
        label="전체 · 모든 변수",
        description="조업방법·선박·선장·어장·위치·수온·조류를 모두 사용",
        method_filter=None,
        features=tuple(
            ENV_FEATURES
            + CONTEXT_FEATURES
            + ["method"]
        ),
        cat_features=tuple(
            CONTEXT_FEATURES
            + ["method"]
        ),
    ),
    "env_only": ExperimentSpec(
        key="env_only",
        label="전체 · 위치/환경 Only",
        description="선박·선장·조업방법을 제외하고 위치·수온·조류·계절만 사용",
        method_filter=None,
        features=tuple(ENV_FEATURES),
        cat_features=(),
    ),
    "school_full": ExperimentSpec(
        key="school_full",
        label="School Fish · 모든 변수",
        description="School Fish 단독 투망만 대상으로 선박·선장 효과까지 포함",
        method_filter="school_fish",
        features=tuple(
            ENV_FEATURES
            + CONTEXT_FEATURES
        ),
        cat_features=tuple(CONTEXT_FEATURES),
    ),
    "school_env": ExperimentSpec(
        key="school_env",
        label="School Fish · 위치/환경 Only",
        description="School Fish 단독 투망에서 위치·수온·조류·계절 신호만 검증",
        method_filter="school_fish",
        features=tuple(ENV_FEATURES),
        cat_features=(),
    ),
    "pa_full": ExperimentSpec(
        key="pa_full",
        label="PA · 모든 변수",
        description="PA 단독 투망만 대상으로 선박·선장·위치·환경 변수를 사용",
        method_filter="pa",
        features=tuple(
            ENV_FEATURES
            + CONTEXT_FEATURES
        ),
        cat_features=tuple(CONTEXT_FEATURES),
    ),
    "ocean_only": ExperimentSpec(
        key="ocean_only",
        label="외부 해양 · 환경 Only",
        description=(
            "Copernicus SST·해류·SSH·Chl-a·SST Gradient와 "
            "위치·계절만 사용"
        ),
        method_filter=None,
        features=tuple(OCEAN_ENV_FEATURES),
        cat_features=(),
    ),
    "ocean_full": ExperimentSpec(
        key="ocean_full",
        label="외부 해양 · 모든 변수",
        description=(
            "Copernicus 해양 특징에 선박·선장·어장·조업방법까지 결합"
        ),
        method_filter=None,
        features=tuple(
            OCEAN_ENV_FEATURES
            + CONTEXT_FEATURES
            + ["method"]
        ),
        cat_features=tuple(
            CONTEXT_FEATURES
            + ["method"]
        ),
    ),
    "onboard_plus_ocean": ExperimentSpec(
        key="onboard_plus_ocean",
        label="기존 + 외부 해양 · 모든 변수",
        description=(
            "기존 수온·조류와 Copernicus 해양 특징을 함께 사용해 "
            "V2 대비 추가 정보의 개선폭을 확인"
        ),
        method_filter=None,
        features=tuple(
            ENV_FEATURES
            + OCEAN_FEATURES
            + CONTEXT_FEATURES
            + ["method"]
        ),
        cat_features=tuple(
            CONTEXT_FEATURES
            + ["method"]
        ),
    ),
}


def experiment_catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "key": spec.key,
                "실험": spec.label,
                "대상": (
                    "전체 투망"
                    if spec.method_filter is None
                    else spec.method_filter
                ),
                "설명": spec.description,
            }
            for spec in EXPERIMENTS.values()
        ]
    )


def _prepare_frame(
    df: pd.DataFrame,
    spec: ExperimentSpec,
    target_col: str,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target 컬럼이 없습니다: {target_col}")

    work = df[df["is_set"]].copy()

    if spec.method_filter is not None:
        work = work[
            work["method"] == spec.method_filter
        ].copy()

    required = ["date", target_col, *spec.features]
    missing = [
        col
        for col in required
        if col not in work.columns
    ]
    if missing:
        raise ValueError(
            "필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    work = work.dropna(
        subset=["date", "lat", "lon"]
    )
    work = work.sort_values("date").reset_index(
        drop=True
    )

    work["_target_catch"] = pd.to_numeric(
        work[target_col],
        errors="coerce",
    ).fillna(0)
    work["_target_success"] = (
        work["_target_catch"] > 0
    ).astype(int)

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

    return work


def _temporal_split(
    work: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train = work[
        work["date"] < pd.Timestamp("2026-01-01")
    ].copy()
    test = work[
        work["date"] >= pd.Timestamp("2026-01-01")
    ].copy()

    if len(train) >= 200 and len(test) >= 50:
        return (
            train,
            test,
            "2023~2025 학습 / 2026 검증",
        )

    cut = max(int(len(work) * 0.8), 1)
    return (
        work.iloc[:cut].copy(),
        work.iloc[cut:].copy(),
        "시간순 80% 학습 / 최근 20% 검증",
    )


def top_k_lift(
    actual: pd.Series,
    score: pd.Series,
    fraction: float = 0.10,
) -> float:
    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(
                actual,
                errors="coerce",
            ),
            "score": pd.to_numeric(
                score,
                errors="coerce",
            ),
        }
    ).dropna()

    if frame.empty:
        return np.nan

    baseline = frame["actual"].mean()
    if baseline <= 0:
        return np.nan

    k = max(
        1,
        int(np.ceil(len(frame) * fraction)),
    )
    top_mean = (
        frame.nlargest(k, "score")["actual"].mean()
    )
    return float(top_mean / baseline)


def _safe_classification_metrics(
    y_true: pd.Series,
    y_prob: pd.Series,
) -> dict:
    metrics = {
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "brier": np.nan,
    }

    if len(y_true) == 0:
        return metrics

    metrics["brier"] = float(
        brier_score_loss(y_true, y_prob)
    )

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob)
        )
        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                y_prob,
            )
        )

    return metrics


def _fit_once(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: ExperimentSpec,
    target_col: str,
    iterations: int,
    with_shap: bool,
) -> ExperimentResult:
    if train["_target_success"].nunique() < 2:
        raise ValueError(
            "학습 구간의 성공/실패가 한 종류뿐이라 "
            "분류 모델을 학습할 수 없습니다."
        )

    features = list(spec.features)
    cat_features = list(spec.cat_features)

    classifier = CatBoostClassifier(
        iterations=iterations,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    classifier.fit(
        train[features],
        train["_target_success"],
        cat_features=cat_features,
    )

    scored = test.copy()
    scored["success_prob"] = (
        classifier.predict_proba(
            scored[features]
        )[:, 1]
    )

    class_metrics = _safe_classification_metrics(
        scored["_target_success"],
        scored["success_prob"],
    )

    positive_train = train[
        train["_target_catch"] > 0
    ].copy()
    positive_test = scored[
        scored["_target_catch"] > 0
    ].copy()

    regressor = None
    mae = np.nan
    medae = np.nan

    min_positive_train = max(
        50,
        len(features) * 5,
    )

    if (
        len(positive_train) >= min_positive_train
        and len(positive_test) >= 10
    ):
        regressor = CatBoostRegressor(
            iterations=iterations,
            depth=7,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        regressor.fit(
            positive_train[features],
            np.log1p(
                positive_train["_target_catch"]
            ),
            cat_features=cat_features,
        )

        pred_positive = np.expm1(
            regressor.predict(
                positive_test[features]
            )
        ).clip(min=0)

        mae = float(
            mean_absolute_error(
                positive_test["_target_catch"],
                pred_positive,
            )
        )
        medae = float(
            median_absolute_error(
                positive_test["_target_catch"],
                pred_positive,
            )
        )

        scored[
            "pred_catch_if_success"
        ] = np.expm1(
            regressor.predict(
                scored[features]
            )
        ).clip(min=0)
        scored["model_score"] = (
            scored["success_prob"]
            * scored["pred_catch_if_success"]
        )
    else:
        scored[
            "pred_catch_if_success"
        ] = np.nan
        scored["model_score"] = (
            scored["success_prob"]
        )

    catch_lift_10 = top_k_lift(
        scored["_target_catch"],
        scored["model_score"],
        0.10,
    )
    catch_lift_20 = top_k_lift(
        scored["_target_catch"],
        scored["model_score"],
        0.20,
    )
    success_lift_10 = top_k_lift(
        scored["_target_success"],
        scored["success_prob"],
        0.10,
    )

    feature_importance = (
        pd.DataFrame(
            {
                "feature": features,
                "importance": (
                    classifier
                    .get_feature_importance()
                ),
            }
        )
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    shap_importance = pd.DataFrame(
        columns=[
            "feature",
            "mean_abs_shap",
            "mean_shap",
        ]
    )
    shap_detail = pd.DataFrame()

    if with_shap and not scored.empty:
        sample_n = min(500, len(scored))
        shap_sample = scored.sample(
            sample_n,
            random_state=42,
        ).reset_index(drop=True)

        pool = Pool(
            shap_sample[features],
            cat_features=cat_features,
        )
        shap_values = (
            classifier.get_feature_importance(
                pool,
                type="ShapValues",
            )
        )
        shap_matrix = shap_values[:, :-1]

        shap_importance = (
            pd.DataFrame(
                {
                    "feature": features,
                    "mean_abs_shap": np.mean(
                        np.abs(shap_matrix),
                        axis=0,
                    ),
                    "mean_shap": np.mean(
                        shap_matrix,
                        axis=0,
                    ),
                }
            )
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        detail_cols = [
            "date",
            "_target_catch",
            "_target_success",
            *features,
        ]
        shap_detail = shap_sample[
            detail_cols
        ].copy()

        for i, feature in enumerate(features):
            shap_detail[
                f"shap__{feature}"
            ] = shap_matrix[:, i]

    metrics = {
        "rows": len(train) + len(test),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_success_rate": float(
            train["_target_success"].mean()
        ),
        "test_success_rate": float(
            scored["_target_success"].mean()
        ),
        "positive_train_rows": len(
            positive_train
        ),
        "positive_test_rows": len(
            positive_test
        ),
        "roc_auc": class_metrics["roc_auc"],
        "pr_auc": class_metrics["pr_auc"],
        "brier": class_metrics["brier"],
        "mae_positive_catch": mae,
        "median_ae_positive_catch": medae,
        "catch_lift_top10": catch_lift_10,
        "catch_lift_top20": catch_lift_20,
        "success_lift_top10": success_lift_10,
    }

    return ExperimentResult(
        spec=spec,
        target_col=target_col,
        classifier=classifier,
        regressor=regressor,
        metrics=metrics,
        feature_importance=feature_importance,
        shap_importance=shap_importance,
        shap_detail=shap_detail,
        scored_test=scored,
        split_note="",
    )


def run_experiment(
    df: pd.DataFrame,
    experiment_key: str,
    target_col: str = "catch_total",
    iterations: int = 250,
    with_shap: bool = True,
) -> ExperimentResult:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(
            f"알 수 없는 실험: {experiment_key}"
        )

    spec = EXPERIMENTS[experiment_key]
    work = _prepare_frame(
        df,
        spec,
        target_col,
    )

    if len(work) < 150:
        raise ValueError(
            f"유효 데이터가 {len(work):,}건으로 "
            "독립 모델 검증에 부족합니다."
        )

    train, test, split_note = _temporal_split(
        work
    )
    if len(test) < 30:
        raise ValueError(
            f"검증 데이터가 {len(test):,}건으로 "
            "너무 적습니다."
        )

    result = _fit_once(
        train=train,
        test=test,
        spec=spec,
        target_col=target_col,
        iterations=iterations,
        with_shap=with_shap,
    )
    result.split_note = split_note
    return result


def run_experiment_suite(
    df: pd.DataFrame,
    target_col: str = "catch_total",
    experiment_keys: list[str] | None = None,
    iterations: int = 220,
) -> tuple[pd.DataFrame, dict[str, ExperimentResult]]:
    keys = experiment_keys or list(
        EXPERIMENTS.keys()
    )

    rows = []
    results: dict[str, ExperimentResult] = {}

    for key in keys:
        spec = EXPERIMENTS[key]
        try:
            result = run_experiment(
                df,
                key,
                target_col=target_col,
                iterations=iterations,
                with_shap=True,
            )
            results[key] = result
            m = result.metrics
            rows.append(
                {
                    "key": key,
                    "실험": spec.label,
                    "상태": "완료",
                    "rows": m["rows"],
                    "test_rows": m["test_rows"],
                    "test_success_rate": (
                        m["test_success_rate"]
                    ),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "brier": m["brier"],
                    "mae": (
                        m["mae_positive_catch"]
                    ),
                    "median_ae": (
                        m[
                            "median_ae_positive_catch"
                        ]
                    ),
                    "top10_lift": (
                        m["catch_lift_top10"]
                    ),
                    "top20_lift": (
                        m["catch_lift_top20"]
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "key": key,
                    "실험": spec.label,
                    "상태": f"제외: {exc}",
                    "rows": np.nan,
                    "test_rows": np.nan,
                    "test_success_rate": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "brier": np.nan,
                    "mae": np.nan,
                    "median_ae": np.nan,
                    "top10_lift": np.nan,
                    "top20_lift": np.nan,
                }
            )

    return pd.DataFrame(rows), results


def walk_forward_validate(
    df: pd.DataFrame,
    experiment_key: str,
    target_col: str = "catch_total",
    iterations: int = 180,
) -> pd.DataFrame:
    if experiment_key not in EXPERIMENTS:
        raise ValueError(
            f"알 수 없는 실험: {experiment_key}"
        )

    spec = EXPERIMENTS[experiment_key]
    work = _prepare_frame(
        df,
        spec,
        target_col,
    )
    years = sorted(
        int(y)
        for y in work["year"].dropna().unique()
    )

    rows = []
    for test_year in years[1:]:
        train = work[
            work["year"] < test_year
        ].copy()
        test = work[
            work["year"] == test_year
        ].copy()

        if (
            len(train) < 200
            or len(test) < 30
        ):
            continue

        try:
            result = _fit_once(
                train=train,
                test=test,
                spec=spec,
                target_col=target_col,
                iterations=iterations,
                with_shap=False,
            )
            m = result.metrics
            rows.append(
                {
                    "test_year": test_year,
                    "train_years": (
                        f"{int(train['year'].min())}"
                        f"~{int(train['year'].max())}"
                    ),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "success_rate": (
                        m["test_success_rate"]
                    ),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "mae": (
                        m["mae_positive_catch"]
                    ),
                    "median_ae": (
                        m[
                            "median_ae_positive_catch"
                        ]
                    ),
                    "top10_lift": (
                        m["catch_lift_top10"]
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "test_year": test_year,
                    "train_years": (
                        f"{int(train['year'].min())}"
                        f"~{int(train['year'].max())}"
                    ),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "success_rate": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "mae": np.nan,
                    "median_ae": np.nan,
                    "top10_lift": np.nan,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def run_species_suite(
    df: pd.DataFrame,
    experiment_key: str = "all_full",
    iterations: int = 200,
) -> tuple[pd.DataFrame, dict[str, ExperimentResult]]:
    rows = []
    results: dict[str, ExperimentResult] = {}

    for label, target_col in SPECIES_TARGETS.items():
        if label == "전체 어획":
            continue

        try:
            result = run_experiment(
                df,
                experiment_key,
                target_col=target_col,
                iterations=iterations,
                with_shap=False,
            )
            results[label] = result
            m = result.metrics
            rows.append(
                {
                    "어종": label,
                    "상태": "완료",
                    "test_rows": m["test_rows"],
                    "test_success_rate": (
                        m["test_success_rate"]
                    ),
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "mae": (
                        m["mae_positive_catch"]
                    ),
                    "top10_lift": (
                        m["catch_lift_top10"]
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "어종": label,
                    "상태": f"제외: {exc}",
                    "test_rows": np.nan,
                    "test_success_rate": np.nan,
                    "roc_auc": np.nan,
                    "pr_auc": np.nan,
                    "mae": np.nan,
                    "top10_lift": np.nan,
                }
            )

    return pd.DataFrame(rows), results
