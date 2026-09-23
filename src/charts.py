from __future__ import annotations

import pandas as pd
import plotly.express as px


def catch_by_year_chart(annual: pd.DataFrame):
    long = annual.melt(
        id_vars="year",
        value_vars=["catch_sj", "catch_yf", "catch_be"],
        var_name="species",
        value_name="catch",
    )
    labels = {"catch_sj": "S/J", "catch_yf": "Y/F", "catch_be": "B/E"}
    long["species"] = long["species"].map(labels)

    fig = px.bar(
        long,
        x="year",
        y="catch",
        color="species",
        barmode="group",
        title="연도별 어종 그룹 어획량",
    )
    fig.update_layout(xaxis_title="연도", yaxis_title="어획량(원본 단위)")
    return fig


def cpue_by_temp_chart(df: pd.DataFrame):
    work = df[
        df["is_set"]
        & df["water_temp"].between(5, 40)
        & df["cpue_per_set"].notna()
    ].copy()

    if work.empty:
        return None

    work["temp_bin"] = (work["water_temp"] * 2).round() / 2
    agg = (
        work.groupby("temp_bin", as_index=False)
        .agg(records=("date", "size"), mean_cpue=("cpue_per_set", "mean"))
    )
    agg = agg[agg["records"] >= 10]

    if agg.empty:
        return None

    fig = px.line(
        agg,
        x="temp_bin",
        y="mean_cpue",
        markers=True,
        title="수온 구간별 평균 CPUE (투망 1회당)",
    )
    fig.update_layout(xaxis_title="수온(℃)", yaxis_title="평균 CPUE")
    return fig


def historical_hotspot_map(grid: pd.DataFrame):
    if grid.empty:
        return None

    fig = px.scatter_geo(
        grid,
        lat="lat_grid",
        lon="lon_grid",
        size="sets",
        color="historical_score",
        hover_name="grid_id",
        hover_data={
            "success_rate": ":.1%",
            "mean_cpue": ":.1f",
            "sets": True,
            "historical_score": ":.1f",
            "lat_grid": False,
            "lon_grid": False,
        },
        projection="natural earth",
        title="과거 조업 Hotspot (미래예측 아님)",
    )
    fig.update_geos(showcoastlines=True, showcountries=True, showland=True)
    fig.update_layout(height=580)
    return fig
