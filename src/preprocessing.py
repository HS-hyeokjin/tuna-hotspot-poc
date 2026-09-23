from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BE_COLUMNS, CATCH_COLUMNS, METHOD_COLUMNS, SJ_COLUMNS, YF_COLUMNS

_COORD_RE = re.compile(r"^([NSEW])(\d+)$", re.IGNORECASE)


def parse_deg_min(value: object, kind: str) -> float:
    """S0925, E15958 같은 좌표를 decimal degree로 변환한다."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan

    text = str(value).strip().upper().replace(" ", "")
    match = _COORD_RE.match(text)
    if not match:
        return np.nan

    direction, digits = match.groups()
    if len(digits) < 3:
        return np.nan

    try:
        degrees = int(digits[:-2])
        minutes = int(digits[-2:])
    except ValueError:
        return np.nan

    if minutes >= 60:
        return np.nan

    value_dd = degrees + minutes / 60.0

    if kind == "lat":
        if direction not in {"N", "S"} or value_dd > 90:
            return np.nan
        return -value_dd if direction == "S" else value_dd

    if kind == "lon":
        if direction not in {"E", "W"} or value_dd > 180:
            return np.nan
        return -value_dd if direction == "W" else value_dd

    raise ValueError("kind must be 'lat' or 'lon'")


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def derive_method(row: pd.Series) -> str:
    active = [name for name in METHOD_COLUMNS if float(row.get(name, 0) or 0) > 0]
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        return "mixed"
    return "none"


def preprocess_fishing_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # 외부 해양 특징과 재결합하기 위한 원본 행 식별자
    df["source_row_id"] = np.arange(len(df), dtype=int)

    for col in ["vessel", "captain", "fishing_ground", "lat_raw", "lon_raw"]:
        df[col] = df[col].astype("string").str.strip()

    date_text = df["date_raw"].astype("string").str.replace(r"\.0$", "", regex=True)
    df["date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")

    numeric_cols = (
        METHOD_COLUMNS
        + ["prev_cum", "daily", "cum"]
        + CATCH_COLUMNS
        + ["water_temp", "current"]
    )
    _to_numeric(df, numeric_cols)
    df[METHOD_COLUMNS + CATCH_COLUMNS] = df[METHOD_COLUMNS + CATCH_COLUMNS].fillna(0)

    df["lat"] = df["lat_raw"].map(lambda x: parse_deg_min(x, "lat"))
    df["lon"] = df["lon_raw"].map(lambda x: parse_deg_min(x, "lon"))

    df["set_count"] = df[METHOD_COLUMNS].sum(axis=1)
    df["catch_total"] = df[CATCH_COLUMNS].sum(axis=1)
    df["catch_sj"] = df[SJ_COLUMNS].sum(axis=1)
    df["catch_yf"] = df[YF_COLUMNS].sum(axis=1)
    df["catch_be"] = df[BE_COLUMNS].sum(axis=1)

    # S/H(PS)는 원본 정의 확인 전까지 S/J에 임의 합산하지 않는다.
    df["catch_sh_ps"] = df["sh_ps"]

    df["method"] = df.apply(derive_method, axis=1)
    df["is_set"] = df["set_count"] > 0
    df["is_success"] = df["is_set"] & (df["catch_total"] > 0)
    df["cpue_per_set"] = np.where(
        df["set_count"] > 0,
        df["catch_total"] / df["set_count"],
        np.nan,
    )

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["dayofyear"] = df["date"].dt.dayofyear

    df["lat_grid"] = df["lat"].round(0)
    df["lon_grid"] = df["lon"].round(0)
    df["grid_id"] = np.where(
        df["lat_grid"].notna() & df["lon_grid"].notna(),
        df["lat_grid"].map(lambda x: f"{x:.0f}")
        + ","
        + df["lon_grid"].map(lambda x: f"{x:.0f}"),
        pd.NA,
    )

    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """오류 탐지용 휴리스틱 품질 지표."""
    checks = [
        ("유효 날짜 없음", df["date"].isna()),
        ("위도 파싱 실패", df["lat"].isna()),
        ("경도 파싱 실패", df["lon"].isna()),
        ("수온 누락", df["water_temp"].isna()),
        ("수온 0 이하", df["water_temp"].notna() & (df["water_temp"] <= 0)),
        ("수온 40 초과", df["water_temp"].notna() & (df["water_temp"] > 40)),
        ("조류 누락", df["current"].isna()),
        (
            "조류 절대값 10 초과(휴리스틱)",
            df["current"].notna() & (df["current"].abs() > 10),
        ),
        (
            "복수 투망방법 동시 기록",
            (df[["school_fish", "log_fish", "pa"]] > 0).sum(axis=1) > 1,
        ),
        ("투망횟수 음수", df["set_count"] < 0),
        ("어획량 음수", df["catch_total"] < 0),
    ]

    total = max(len(df), 1)
    rows = []
    for name, mask in checks:
        count = int(mask.fillna(False).sum())
        rows.append(
            {
                "check": name,
                "count": count,
                "rate_pct": round(count / total * 100, 2),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["count", "check"], ascending=[False, True])
        .reset_index(drop=True)
    )
