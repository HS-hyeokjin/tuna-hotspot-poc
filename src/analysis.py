from __future__ import annotations

import pandas as pd


def kpi_summary(df: pd.DataFrame) -> dict:
    set_rows = df[df["is_set"]]
    return {
        "rows": len(df),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "vessels": int(df["vessel"].nunique(dropna=True)),
        "captains": int(df["captain"].nunique(dropna=True)),
        "grounds": int(df["fishing_ground"].nunique(dropna=True)),
        "set_rows": int(df["is_set"].sum()),
        "success_rows": int(df["is_success"].sum()),
        "success_rate": float(set_rows["is_success"].mean()) if len(set_rows) else 0.0,
        "catch_total": float(df["catch_total"].sum()),
        "mean_cpue": float(set_rows["cpue_per_set"].mean()) if len(set_rows) else 0.0,
    }


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.dropna(subset=["year"]).copy()
    return (
        work.groupby("year", as_index=False)
        .agg(
            records=("date", "size"),
            set_rows=("is_set", "sum"),
            catch_total=("catch_total", "sum"),
            catch_sj=("catch_sj", "sum"),
            catch_yf=("catch_yf", "sum"),
            catch_be=("catch_be", "sum"),
        )
        .sort_values("year")
    )


def method_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["is_set"]].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "records",
                "sets",
                "catch_total",
                "success_rate",
                "mean_cpue",
            ]
        )

    return (
        work.groupby("method", as_index=False)
        .agg(
            records=("date", "size"),
            sets=("set_count", "sum"),
            catch_total=("catch_total", "sum"),
            success_rate=("is_success", "mean"),
            mean_cpue=("cpue_per_set", "mean"),
        )
        .sort_values("records", ascending=False)
    )


def daily_species_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """당일 총어획량과 어종별 합계를 선박·일자 단위로 비교한다.

    원본의 '당일' 정의가 실제 일일 총어획량이라는 전제는 아직
    현업 확인이 필요하므로, 이 결과는 데이터 정의 확인용이다.
    """
    work = df[
        df["date"].notna()
        & df["vessel"].notna()
    ].copy()

    if work.empty:
        return pd.DataFrame()

    grouped = (
        work.groupby(
            ["date", "vessel"],
            as_index=False,
        )
        .agg(
            official_daily=("daily", "max"),
            species_sum=("catch_total", "sum"),
            set_count=("set_count", "sum"),
            records=("date", "size"),
        )
    )

    grouped = grouped[
        grouped["official_daily"].notna()
    ].copy()

    if grouped.empty:
        return grouped

    grouped["difference"] = (
        grouped["official_daily"]
        - grouped["species_sum"]
    )
    grouped["abs_difference"] = (
        grouped["difference"].abs()
    )
    grouped["exact_match"] = (
        grouped["abs_difference"] < 1e-9
    )

    return grouped.sort_values(
        "abs_difference",
        ascending=False,
    ).reset_index(drop=True)


def hotspot_grid(
    df: pd.DataFrame,
    min_sets: int = 3,
) -> pd.DataFrame:
    work = df[
        df["is_set"]
        & df["lat_grid"].notna()
        & df["lon_grid"].notna()
    ].copy()

    if work.empty:
        return pd.DataFrame()

    result = (
        work.groupby(
            [
                "lat_grid",
                "lon_grid",
                "grid_id",
            ],
            as_index=False,
        )
        .agg(
            records=("date", "size"),
            sets=("set_count", "sum"),
            successes=("is_success", "sum"),
            catch_total=("catch_total", "sum"),
            mean_cpue=("cpue_per_set", "mean"),
            median_cpue=("cpue_per_set", "median"),
            last_date=("date", "max"),
        )
    )

    result = result[
        result["sets"] >= min_sets
    ].copy()
    if result.empty:
        return result

    result["success_rate"] = (
        result["successes"]
        / result["records"].clip(lower=1)
    )

    # 과거 성과 탐색용 점수이며 미래예측 점수가 아니다.
    result["historical_score"] = (
        result["success_rate"].rank(pct=True)
        * 0.45
        + result["mean_cpue"].rank(pct=True)
        * 0.45
        + result["sets"].rank(pct=True)
        * 0.10
    ) * 100

    return (
        result.sort_values(
            "historical_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )
