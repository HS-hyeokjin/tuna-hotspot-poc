from io import BytesIO

import numpy as np
import pandas as pd

from src.setlevel import (
    load_2026_operation_counts,
    load_setlevel_2026_excel,
    parse_set_coordinate,
    setlevel_coverage,
)


def _workbook_bytes() -> BytesIO:
    catch = pd.DataFrame(
        np.nan,
        index=range(10),
        columns=range(25),
    )
    header = [
        "No.",
        "날짜",
        "위도",
        np.nan,
        "경도",
        np.nan,
        "선명",
        "구분",
        "GG\nYF+20",
        "GG\nYF+7.5",
        "PS\nYF+20",
        "PS\nYF+7.5",
        "PS\nSJ+7.5",
        "PS\nSJ+4.0",
        "YF+20",
        "YF+7.5",
        "YF+4",
        "SJ+7.5",
        "SJ+4",
        "SJ+3",
        "SJ-3",
        "BE",
        "기타",
        "Total",
        "가불",
    ]
    catch.iloc[1, :] = header
    catch.iloc[2, :8] = [
        1,
        pd.Timestamp("2026-01-01"),
        "S04-08",
        np.nan,
        "W165-18",
        np.nan,
        "AX",
        "SC",
    ]
    catch.iloc[2, 17] = 20
    catch.iloc[2, 23] = 20
    catch.iloc[3, :8] = [
        2,
        pd.Timestamp("2026-01-01"),
        "S04-09",
        np.nan,
        "W165-20",
        np.nan,
        "AX",
        "PA",
    ]
    catch.iloc[3, 14] = 10
    catch.iloc[3, 23] = 10

    ops = pd.DataFrame(
        np.nan,
        index=range(8),
        columns=range(67),
    )
    vessels = [
        "AX",
        "CC",
        "FF",
        "CO",
        "PP",
        "PO",
        "ST",
        "TA",
        "SR1",
        "SR2",
        "MR",
    ]
    ops.iloc[5, 12:23] = vessels
    ops.iloc[6, 0] = pd.Timestamp(
        "2026-01-01"
    )
    ops.iloc[6, 23] = 2
    ops.iloc[6, 34] = 1

    buffer = BytesIO()
    with pd.ExcelWriter(
        buffer,
        engine="openpyxl",
    ) as writer:
        catch.to_excel(
            writer,
            sheet_name="26어획",
            header=False,
            index=False,
        )
        ops.to_excel(
            writer,
            sheet_name="26조업",
            header=False,
            index=False,
        )
    buffer.seek(0)
    return buffer


def test_parse_set_coordinate_dmm():
    assert abs(
        parse_set_coordinate(
            "S04-08",
            "lat",
        )
        - (-(4 + 8 / 60))
    ) < 1e-9
    assert abs(
        parse_set_coordinate(
            "W165-18",
            "lon",
        )
        - (-(165 + 18 / 60))
    ) < 1e-9
    assert abs(
        parse_set_coordinate(
            "E15958",
            "lon",
        )
        - (159 + 58 / 60)
    ) < 1e-9


def test_setlevel_workbook_load_and_gap():
    buffer = _workbook_bytes()
    events = load_setlevel_2026_excel(
        buffer
    )
    buffer.seek(0)
    ops = load_2026_operation_counts(
        buffer
    )

    assert len(events) == 2
    assert events[
        "has_positive_catch"
    ].all()
    assert events[
        ["lat", "lon"]
    ].notna().all().all()

    summary, by_method = setlevel_coverage(
        events,
        ops,
    )
    assert summary["reported_sets"] == 3
    assert summary[
        "matched_event_rows"
    ] == 2
    assert summary[
        "net_unlocated_gap"
    ] == 1
    assert {
        "SC",
        "PA",
    }.issubset(
        set(
            by_method["method"]
            .dropna()
            .astype(str)
        )
    )
