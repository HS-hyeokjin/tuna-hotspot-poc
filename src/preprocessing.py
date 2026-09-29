from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import BE_COLUMNS, CATCH_COLUMNS, METHOD_COLUMNS, SJ_COLUMNS, YF_COLUMNS

_COORD_RE = re.compile(r"^([NSEW])(\d+)$", re.IGNORECASE)


def _coord_parts(
    value: object,
    kind: str,
) -> tuple[str, int, int] | None:
    """도분(DMM) 좌표를 방향/도/분으로 분해한다."""
    if value is None or (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return None

    text = (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
    )
    match = _COORD_RE.match(text)
    if not match:
        return None

    direction, digits = match.groups()
    if len(digits) < 3:
        return None

    try:
        degrees = int(digits[:-2])
        minutes = int(digits[-2:])
    except ValueError:
        return None

    if minutes >= 60:
        return None

    if kind == "lat":
        if direction not in {"N", "S"}:
            return None
        if degrees > 90:
            return None
    elif kind == "lon":
        if direction not in {"E", "W"}:
            return None
        if degrees > 180:
            return None
    else:
        raise ValueError(
            "kind must be 'lat' or 'lon'"
        )

    return direction, degrees, minutes


def parse_deg_min(value: object, kind: str) -> float:
    """S0925/E15958 같은 도분 좌표를 decimal degree로 변환한다."""
    parts = _coord_parts(value, kind)
    if parts is None:
        return np.nan

    direction, degrees, minutes = parts
    value_dd = degrees + minutes / 60.0
    if direction in {"S", "W"}:
        value_dd *= -1

    return value_dd

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

    # 모델 및 Copernicus 결합은 위의 원본 도분 좌표를 변환한
    # lat/lon(decimal degree)을 그대로 사용한다.
    # 아래 1도 grid는 기존 Historical Hotspot 집계 호환용 파생값이며
    # 원본 좌표의 정밀도를 의미하지 않는다.
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


def _coord_minute(
    value: object,
    kind: str,
) -> float:
    parts = _coord_parts(value, kind)
    if parts is None:
        return np.nan
    return float(parts[2])


def coordinate_audit_summary(
    df: pd.DataFrame,
) -> dict:
    """원본 도분 좌표의 표현 정밀도와 set-level 한계를 요약한다."""
    total_rows = len(df)
    valid_pair = (
        df["lat"].notna()
        & df["lon"].notna()
    )

    if "is_set" in df.columns:
        set_mask = df["is_set"].fillna(False)
    else:
        set_mask = (
            pd.to_numeric(
                df.get("set_count", 0),
                errors="coerce",
            ).fillna(0)
            > 0
        )

    set_count = pd.to_numeric(
        df.get("set_count", 0),
        errors="coerce",
    ).fillna(0)
    multi_set = set_mask & (set_count > 1)

    lat_minutes = df["lat_raw"].map(
        lambda x: _coord_minute(x, "lat")
    )
    lon_minutes = df["lon_raw"].map(
        lambda x: _coord_minute(x, "lon")
    )

    median_abs_lat = pd.to_numeric(
        df.loc[valid_pair, "lat"],
        errors="coerce",
    ).abs().median()

    lat_resolution_km = 111.32 / 60.0
    if pd.notna(median_abs_lat):
        lon_resolution_km = (
            lat_resolution_km
            * math.cos(
                math.radians(
                    float(median_abs_lat)
                )
            )
        )
    else:
        lon_resolution_km = np.nan

    set_rows = int(set_mask.sum())
    valid_set_pairs = int(
        (set_mask & valid_pair).sum()
    )

    return {
        "rows": int(total_rows),
        "valid_pairs": int(valid_pair.sum()),
        "valid_pair_rate": (
            float(valid_pair.mean())
            if total_rows
            else np.nan
        ),
        "set_rows": set_rows,
        "valid_set_pairs": valid_set_pairs,
        "valid_set_pair_rate": (
            valid_set_pairs / set_rows
            if set_rows
            else np.nan
        ),
        "lat_minute_unique": int(
            lat_minutes.dropna().nunique()
        ),
        "lon_minute_unique": int(
            lon_minutes.dropna().nunique()
        ),
        "lat_minute_zero_rate": (
            float(
                (lat_minutes.dropna() == 0).mean()
            )
            if lat_minutes.notna().any()
            else np.nan
        ),
        "lon_minute_zero_rate": (
            float(
                (lon_minutes.dropna() == 0).mean()
            )
            if lon_minutes.notna().any()
            else np.nan
        ),
        "multi_set_rows": int(
            multi_set.sum()
        ),
        "multi_set_rate": (
            float(multi_set.sum() / set_rows)
            if set_rows
            else np.nan
        ),
        "median_abs_lat": (
            float(median_abs_lat)
            if pd.notna(median_abs_lat)
            else np.nan
        ),
        "encoded_lat_resolution_km": float(
            lat_resolution_km
        ),
        "encoded_lon_resolution_km_at_median_lat": (
            float(lon_resolution_km)
            if pd.notna(lon_resolution_km)
            else np.nan
        ),
    }


def coordinate_minute_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """위/경도 분(minute) 00~59 사용 분포를 반환한다."""
    lat_minutes = (
        df["lat_raw"]
        .map(
            lambda x: _coord_minute(
                x,
                "lat",
            )
        )
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(range(60), fill_value=0)
    )
    lon_minutes = (
        df["lon_raw"]
        .map(
            lambda x: _coord_minute(
                x,
                "lon",
            )
        )
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(range(60), fill_value=0)
    )

    return pd.DataFrame(
        {
            "minute": list(range(60)),
            "latitude_count": (
                lat_minutes.to_numpy()
            ),
            "longitude_count": (
                lon_minutes.to_numpy()
            ),
        }
    )


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """오류 탐지용 휴리스틱 품질 지표."""
    checks = [
        ("유효 날짜 없음", df["date"].isna()),
        ("위도 파싱 실패", df["lat"].isna()),
        ("경도 파싱 실패", df["lon"].isna()),
        (
            "좌표쌍 파싱 실패",
            df["lat"].isna()
            | df["lon"].isna(),
        ),
        (
            "동일 행 투망 2회 이상(좌표 1개)",
            df["set_count"] > 1,
        ),
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
