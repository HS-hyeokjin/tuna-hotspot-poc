from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, roc_auc_score


FEATURES = [
    "lat",
    "lon",
    "water_temp",
    "current",
    "month",
    "quarter",
    "vessel",
    "captain",
    "fishing_ground",
    "method",
]
CAT_FEATURES = ["vessel", "captain", "fishing_ground", "method"]


@dataclass
class BaselineResult:
    classifier: Any | None
    regressor: Any | None
    metrics: dict
    feature_importance: pd.DataFrame
    scored_test: pd.DataFrame
    split_note: str


def _clean_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["is_set"]].copy()
    work = work.dropna(subset=["date", "lat", "lon"])
    work = work.sort_values("date").reset_index(drop=True)

    for col in CAT_FEATURES:
        work[col] = work[col].astype("string").fillna("UNKNOWN").astype(str)

    for col in ["water_temp", "current", "month", "quarter"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    return work


def _temporal_split(
    work: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    train = work[work["date"] < pd.Timestamp("2026-01-01")].copy()
    test = work[work["date"] >= pd.Timestamp("2026-01-01")].copy()

    if len(train) >= 200 and len(test) >= 50:
        return train, test, "2023~2025 학습 / 2026 검증"

    cut = max(int(len(work) * 0.8), 1)
    return (
        work.iloc[:cut].copy(),
        work.iloc[cut:].copy(),
        "시간순 80% 학습 / 최근 20% 검증",
    )


def train_baseline(df: pd.DataFrame) -> BaselineResult:
    work = _clean_model_frame(df)

    if len(work) < 200:
        raise ValueError("Baseline 학습에 사용할 유효 투망 데이터가 200건 미만입니다.")

    train, test, split_note = _temporal_split(work)
    if test.empty:
        raise ValueError("검증 데이터가 없습니다.")

    classifier = CatBoostClassifier(
        iterations=350,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    classifier.fit(
        train[FEATURES],
        train["is_success"].astype(int),
        cat_features=CAT_FEATURES,
    )

    test = test.copy()
    test["success_prob"] = classifier.predict_proba(test[FEATURES])[:, 1]

    auc = np.nan
    if test["is_success"].nunique() > 1:
        auc = float(
            roc_auc_score(
                test["is_success"].astype(int),
                test["success_prob"],
            )
        )

    positive_train = train[train["catch_total"] > 0].copy()
    positive_test = test[test["catch_total"] > 0].copy()

    regressor = None
    mae = np.nan

    if len(positive_train) >= 100 and len(positive_test) >= 20:
        regressor = CatBoostRegressor(
            iterations=350,
            depth=7,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        regressor.fit(
            positive_train[FEATURES],
            np.log1p(positive_train["catch_total"]),
            cat_features=CAT_FEATURES,
        )

        positive_pred = np.expm1(
            regressor.predict(positive_test[FEATURES])
        ).clip(min=0)

        mae = float(
            mean_absolute_error(
                positive_test["catch_total"],
                positive_pred,
            )
        )

        test["pred_catch_if_success"] = np.expm1(
            regressor.predict(test[FEATURES])
        ).clip(min=0)

        test["model_score"] = (
            test["success_prob"] * test["pred_catch_if_success"]
        )
    else:
        test["pred_catch_if_success"] = np.nan
        test["model_score"] = test["success_prob"]

    fi = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": classifier.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)

    metrics = {
        "train_rows": len(train),
        "test_rows": len(test),
        "positive_train_rows": len(positive_train),
        "positive_test_rows": len(positive_test),
        "auc": auc,
        "mae_positive_catch": mae,
        "test_success_rate": float(test["is_success"].mean()),
    }

    return BaselineResult(
        classifier,
        regressor,
        metrics,
        fi.reset_index(drop=True),
        test,
        split_note,
    )
