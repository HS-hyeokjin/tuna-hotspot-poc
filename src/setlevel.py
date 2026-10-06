from __future__ import annotations

import math
import re
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


_SET_COLUMNS = [
    "event_no",
    "date",
    "lat_raw",
    "_lat_blank",
    "lon_raw",
    "_lon_blank",
    "vessel",
    "method",
    "gg_yf20",
    "gg_yf75",
    "ps_yf20",
    "ps_yf75",
    "ps_sj75",
    "ps_sj40",
    "yf20",
    "yf75",
    "yf4",
    "sj75",
    "sj4",
    "sj3",
    "sj_m3",
    "be",
    "other",
    "catch_total",
    "advance",
]

_CATCH_COMPONENTS = [
    "gg_yf20",
    "gg_yf75",
    "ps_yf20",
    "ps_yf75",
    "ps_sj75",
    "ps_sj40",
    "yf20",
    "yf75",
    "yf4",
    "sj75",
    "sj4",
    "sj3",
    "sj_m3",
    "be",
    "other",
]

_COORD_SEPARATED_RE = re.compile(
    r"^([NSEW])\s*(\d{1,3})\s*[-°:]\s*(\d{1,3}(?:\.\d+)?)['′]?$",
    re.IGNORECASE,
)
_COORD_COMPACT_RE = re.compile(
    r"^([NSEW])(\d+)$",
    re.IGNORECASE,
)


def parse_set_coordinate(
    value: object,
    kind: str,
) -> float:
    """2026 어획 sheet의 DMM 좌표를 decimal degree로 변환한다.

    지원 예:
    - S04-08
    - W165-18
    - S0529
    - E15958

    방향 문자가 없는 값은 임의 보정하지 않고 NaN 처리한다.
    """
    if value is None or (
        isinstance(value, float)
        and math.isnan(value)
    ):
        return np.nan

    text = (
        str(value)
        .strip()
        .upper()
        .replace(" ", "")
    )

    match = _COORD_SEPARATED_RE.match(text)
    if match:
        direction, degree_text, minute_text = (
            match.groups()
        )
        degrees = int(degree_text)
        minutes = float(minute_text)
    else:
        match = _COORD_COMPACT_RE.match(text)
        if not match:
            return np.nan

        direction, digits = match.groups()
        if len(digits) < 3:
            return np.nan

        degree_digits = 2 if kind == "lat" else 3
        if len(digits) <= degree_digits:
            return np.nan

        try:
            degrees = int(
                digits[:degree_digits]
            )
            minutes = float(
                digits[degree_digits:]
            )
        except ValueError:
            return np.nan

        # 예: S0925, E15958
        if minutes >= 60:
            # 일부 데이터의 050 같은 zero-padded minute를 허용
            minute_digits = digits[degree_digits:]
            if (
                len(minute_digits) == 3
                and minute_digits.startswith("0")
            ):
                minutes = float(
                    minute_digits[1:]
                )

    if minutes >= 60:
        return np.nan

    direction = direction.upper()

    if kind == "lat":
        if direction not in {"N", "S"}:
            return np.nan
        if degrees > 90:
            return np.nan
    elif kind == "lon":
        if direction not in {"E", "W"}:
            return np.nan
        if degrees > 180:
            return np.nan
    else:
        raise ValueError(
            "kind must be 'lat' or 'lon'"
        )

    value_dd = degrees + minutes / 60.0
    if direction in {"S", "W"}:
        value_dd *= -1.0

    return float(value_dd)


def _find_set_header(raw: pd.DataFrame) -> int:
    required = {
        "No.",
        "날짜",
        "위도",
        "경도",
        "선명",
        "구분",
        "Total",
    }

    for idx in raw.index:
        values = {
            str(value).strip()
            for value in raw.loc[idx].dropna()
        }
        if required.issubset(values):
            return int(idx)

    raise ValueError(
        "26어획 sheet에서 set-level 헤더를 찾지 못했습니다."
    )


def load_setlevel_2026_excel(
    source: str | Path | BinaryIO,
) -> pd.DataFrame:
    """26년 운항.xlsx의 '26어획' set-level 블록을 읽는다."""
    raw = pd.read_excel(
        source,
        sheet_name="26어획",
        header=None,
    )
    header_row = _find_set_header(raw)

    work = (
        raw.iloc[
            header_row + 1 :,
            : len(_SET_COLUMNS),
        ]
        .copy()
    )
    work.columns = _SET_COLUMNS

    work["date"] = pd.to_datetime(
        work["date"],
        errors="coerce",
    )
    work["event_no"] = pd.to_numeric(
        work["event_no"],
        errors="coerce",
    )

    work = work[
        work["date"].notna()
        & work["vessel"].notna()
    ].copy()

    work["source_row_id"] = np.arange(
        len(work),
        dtype=int,
    )
    work["vessel"] = (
        work["vessel"]
        .astype("string")
        .str.strip()
    )
    work["method"] = (
        work["method"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    work["method"] = work["method"].replace(
        {"<NA>": pd.NA, "NAN": pd.NA}
    )

    work["lat_raw"] = (
        work["lat_raw"]
        .astype("string")
        .str.strip()
    )
    work["lon_raw"] = (
        work["lon_raw"]
        .astype("string")
        .str.strip()
    )

    work["lat"] = work["lat_raw"].map(
        lambda x: parse_set_coordinate(
            x,
            "lat",
        )
    )
    work["lon"] = work["lon_raw"].map(
        lambda x: parse_set_coordinate(
            x,
            "lon",
        )
    )

    for col in [
        *_CATCH_COMPONENTS,
        "catch_total",
        "advance",
    ]:
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work[_CATCH_COMPONENTS] = (
        work[_CATCH_COMPONENTS]
        .fillna(0.0)
    )
    work["catch_total"] = (
        work["catch_total"]
        .fillna(0.0)
    )

    work["catch_yf"] = (
        work[
            [
                "gg_yf20",
                "gg_yf75",
                "ps_yf20",
                "ps_yf75",
                "yf20",
                "yf75",
                "yf4",
            ]
        ]
        .sum(axis=1)
    )
    work["catch_sj"] = (
        work[
            [
                "ps_sj75",
                "ps_sj40",
                "sj75",
                "sj4",
                "sj3",
                "sj_m3",
            ]
        ]
        .sum(axis=1)
    )
    work["catch_be"] = work["be"]
    work["catch_other"] = work["other"]

    work["has_positive_catch"] = (
        work["catch_total"] > 0
    )
    work["year"] = work["date"].dt.year
    work["month"] = work["date"].dt.month
    work["quarter"] = work["date"].dt.quarter
    work["dayofyear"] = (
        work["date"].dt.dayofyear
    )

    return (
        work.sort_values(
            ["date", "event_no"],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def load_2026_operation_counts(
    source: str | Path | BinaryIO,
) -> pd.DataFrame:
    """'26조업'의 일자/선박/방법별 투망횟수 행렬을 long format으로 변환한다."""
    raw = pd.read_excel(
        source,
        sheet_name="26조업",
        header=None,
    )

    vessel_row = None
    for idx in raw.index:
        row = raw.loc[idx]
        values = {
            str(value).strip()
            for value in row.dropna()
        }
        if {
            "AX",
            "CC",
            "FF",
            "CO",
        }.issubset(values):
            vessel_row = int(idx)
            break

    if vessel_row is None:
        raise ValueError(
            "26조업 sheet의 선박 헤더를 찾지 못했습니다."
        )

    # 원본 구조:
    # 12~22: 합계, 23~33: SC, 34~44: PA, 45~55: LOG
    vessels = [
        str(raw.iat[vessel_row, col]).strip()
        for col in range(12, 23)
    ]

    rows: list[dict] = []
    for row_idx in range(
        vessel_row + 1,
        len(raw),
    ):
        date = pd.to_datetime(
            raw.iat[row_idx, 0],
            errors="coerce",
        )
        if pd.isna(date):
            continue

        for method, start_col in [
            ("SC", 23),
            ("PA", 34),
            ("LOG", 45),
        ]:
            for offset, vessel in enumerate(
                vessels
            ):
                sets = pd.to_numeric(
                    raw.iat[
                        row_idx,
                        start_col + offset,
                    ],
                    errors="coerce",
                )
                if pd.isna(sets):
                    sets = 0.0

                rows.append(
                    {
                        "date": date,
                        "vessel": vessel,
                        "method": method,
                        "reported_sets": float(
                            sets
                        ),
                    }
                )

    return pd.DataFrame(rows)


def setlevel_coverage(
    events: pd.DataFrame,
    operations: pd.DataFrame | None = None,
) -> tuple[dict, pd.DataFrame]:
    """set-level 이벤트 커버리지와 미위치 투망 규모를 요약한다."""
    summary = {
        "event_rows": int(len(events)),
        "positive_event_rows": int(
            events["has_positive_catch"].sum()
        ),
        "valid_coordinate_rows": int(
            (
                events["lat"].notna()
                & events["lon"].notna()
            ).sum()
        ),
        "method_known_rows": int(
            events["method"].notna().sum()
        ),
    }

    by_method = (
        events.groupby(
            "method",
            dropna=False,
        )
        .agg(
            event_rows=("date", "size"),
            positive_events=(
                "has_positive_catch",
                "sum",
            ),
            valid_coordinates=(
                "lat",
                lambda s: int(s.notna().sum()),
            ),
            catch_total=(
                "catch_total",
                "sum",
            ),
        )
        .reset_index()
    )

    if operations is None or operations.empty:
        return summary, by_method

    event_counts = (
        events[
            events["method"].notna()
        ]
        .groupby(
            ["date", "vessel", "method"],
            as_index=False,
        )
        .size()
        .rename(
            columns={"size": "event_rows"}
        )
    )

    compare = (
        operations.merge(
            event_counts,
            on=[
                "date",
                "vessel",
                "method",
            ],
            how="outer",
        )
        .fillna(
            {
                "reported_sets": 0.0,
                "event_rows": 0.0,
            }
        )
    )
    compare["unlocated_set_gap"] = (
        compare["reported_sets"]
        - compare["event_rows"]
    )

    # 음수 gap은 날짜/기록 정합성 이슈로 별도 보존하고,
    # 실패 위치로 임의 해석하지 않는다.
    summary["reported_sets"] = float(
        compare["reported_sets"].sum()
    )
    summary["matched_event_rows"] = float(
        compare["event_rows"].sum()
    )
    summary["net_unlocated_gap"] = float(
        compare["unlocated_set_gap"].sum()
    )
    summary["negative_gap_groups"] = int(
        (
            compare["unlocated_set_gap"] < 0
        ).sum()
    )

    operation_method = (
        compare.groupby(
            "method",
            as_index=False,
        )
        .agg(
            reported_sets=(
                "reported_sets",
                "sum",
            ),
            matched_event_rows=(
                "event_rows",
                "sum",
            ),
            net_unlocated_gap=(
                "unlocated_set_gap",
                "sum",
            ),
        )
    )

    by_method = by_method.merge(
        operation_method,
        on="method",
        how="outer",
    )

    return summary, by_method
