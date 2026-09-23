from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import annual_summary, hotspot_grid, kpi_summary, method_summary
from src.charts import catch_by_year_chart, cpue_by_temp_chart, historical_hotspot_map
from src.config import DEFAULT_DATA_PATH
from src.loader import load_fishing_excel
from src.model import train_baseline
from src.preprocessing import preprocess_fishing_data, quality_report


st.set_page_config(
    page_title="Tuna Hotspot AI PoC",
    page_icon="🐟",
    layout="wide",
)

st.title("🐟 Tuna Hotspot AI PoC")
st.caption(
    "참치 선망 과거 조업데이터 분석 · 데이터 품질 진단 · "
    "ML baseline · Historical Hotspot"
)
st.info(
    "V1은 24~48시간 미래 어장을 확정 예측하지 않습니다. "
    "현재 조업데이터에 예측 신호가 있는지 검증하는 PoC입니다. "
    "향후 SST 예보·수온전선·Chl-a·해류·풍속·파고 등 외부 데이터를 "
    "결합한 뒤 미래 후보 해역 평가로 확장합니다."
)


@st.cache_data(show_spinner=False)
def load_and_prepare(
    uploaded_bytes: bytes | None,
    local_path: str | None,
):
    import io

    if uploaded_bytes is not None:
        raw = load_fishing_excel(io.BytesIO(uploaded_bytes))
    elif local_path:
        raw = load_fishing_excel(local_path)
    else:
        raise ValueError("데이터 소스가 없습니다.")

    return preprocess_fishing_data(raw)


with st.sidebar:
    st.header("데이터")
    uploaded = st.file_uploader(
        "조업보고 Excel 업로드",
        type=["xlsx"],
    )
    use_local = st.checkbox(
        "로컬 data 폴더 파일 사용",
        value=DEFAULT_DATA_PATH.exists(),
    )
    st.caption("원본 사내 데이터는 GitHub에 커밋하지 마세요.")


source_bytes = uploaded.getvalue() if uploaded else None
local_source = (
    str(DEFAULT_DATA_PATH)
    if (
        uploaded is None
        and use_local
        and DEFAULT_DATA_PATH.exists()
    )
    else None
)

if source_bytes is None and local_source is None:
    st.warning(
        "왼쪽에서 Excel 파일을 업로드하거나 "
        "`data/선망_조업보고 데이터.xlsx`를 로컬에 배치하세요."
    )
    st.stop()

try:
    df = load_and_prepare(source_bytes, local_source)
except Exception as exc:
    st.error(f"데이터 로딩 실패: {exc}")
    st.stop()


with st.sidebar:
    st.divider()
    st.header("공통 필터")

    years = sorted(
        int(x)
        for x in df["year"].dropna().unique()
    )
    selected_years = st.multiselect(
        "연도",
        years,
        default=years,
    )

    methods = sorted(
        str(x)
        for x in df["method"].dropna().unique()
    )
    selected_methods = st.multiselect(
        "조업방법",
        methods,
        default=methods,
    )

    vessels = sorted(
        str(x)
        for x in df["vessel"].dropna().unique()
    )
    selected_vessels = st.multiselect(
        "선박",
        vessels,
        default=vessels,
    )


filtered = df[
    df["year"].isin(selected_years)
    & df["method"].isin(selected_methods)
    & df["vessel"].isin(selected_vessels)
].copy()

if filtered.empty:
    st.warning("필터 결과가 없습니다.")
    st.stop()

summary = kpi_summary(filtered)

(
    tab_overview,
    tab_quality,
    tab_species,
    tab_method,
    tab_hotspot,
    tab_model,
) = st.tabs(
    [
        "개요",
        "데이터 품질",
        "어종 분석",
        "조업방법/CPUE",
        "Historical Hotspot",
        "ML Baseline",
    ]
)


with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("전체 기록", f"{summary['rows']:,}")
    c2.metric("투망 기록", f"{summary['set_rows']:,}")
    c3.metric("성공 투망", f"{summary['success_rows']:,}")
    c4.metric(
        "투망 성공률",
        f"{summary['success_rate']:.1%}",
    )
    c5.metric(
        "평균 CPUE",
        f"{summary['mean_cpue']:.1f}",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("선박", f"{summary['vessels']:,}")
    c2.metric("선장", f"{summary['captains']:,}")
    c3.metric("어장명", f"{summary['grounds']:,}")

    date_text = "-"
    if (
        pd.notna(summary["start_date"])
        and pd.notna(summary["end_date"])
    ):
        date_text = (
            f"{summary['start_date']:%Y-%m-%d} ~ "
            f"{summary['end_date']:%Y-%m-%d}"
        )
    c4.metric("기간", date_text)

    annual = annual_summary(filtered)
    st.plotly_chart(
        catch_by_year_chart(annual),
        use_container_width=True,
    )
    st.dataframe(
        annual,
        use_container_width=True,
        hide_index=True,
    )


with tab_quality:
    st.subheader("데이터 품질 진단")
    st.caption(
        "수온/조류 범위는 오류 탐지용 휴리스틱이며, "
        "현업 단위·센서 규격 확인 후 조정해야 합니다."
    )

    q = quality_report(filtered)
    st.dataframe(
        q,
        use_container_width=True,
        hide_index=True,
    )

    bad_coord = filtered[
        filtered["lat"].isna()
        | filtered["lon"].isna()
    ][
        [
            "date",
            "vessel",
            "fishing_ground",
            "lat_raw",
            "lon_raw",
            "water_temp",
            "current",
        ]
    ].head(100)

    with st.expander(
        "좌표 파싱 실패 예시 (최대 100건)"
    ):
        st.dataframe(
            bad_coord,
            use_container_width=True,
            hide_index=True,
        )


with tab_species:
    st.subheader("어종 그룹별 분석")
    st.caption(
        "S/H(PS)는 정의 확인 전까지 "
        "S/J에 합산하지 않고 별도 보존합니다."
    )

    species = filtered[filtered["is_set"]].copy()
    long = species.melt(
        id_vars=[
            "date",
            "year",
            "month",
            "water_temp",
            "method",
        ],
        value_vars=[
            "catch_sj",
            "catch_yf",
            "catch_be",
            "catch_sh_ps",
        ],
        var_name="species",
        value_name="catch",
    )

    label_map = {
        "catch_sj": "S/J",
        "catch_yf": "Y/F",
        "catch_be": "B/E",
        "catch_sh_ps": "S/H (PS)",
    }
    long["species"] = long["species"].map(label_map)

    species_summary = (
        long.groupby("species", as_index=False)
        .agg(
            records=("catch", "size"),
            total_catch=("catch", "sum"),
            mean_catch=("catch", "mean"),
        )
    )

    st.dataframe(
        species_summary.sort_values(
            "total_catch",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    positive_species = long[long["catch"] > 0]
    if not positive_species.empty:
        fig = px.box(
            positive_species,
            x="species",
            y="catch",
            points=False,
            log_y=True,
            title="어종 그룹별 양(양수 어획, 로그축)",
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )


with tab_method:
    st.subheader("조업방법별 성과와 CPUE")

    method_df = method_summary(filtered)
    method_df_display = method_df.copy()

    if not method_df_display.empty:
        method_df_display["success_rate"] = (
            method_df_display["success_rate"]
            .map(lambda x: f"{x:.1%}")
        )

    st.dataframe(
        method_df_display,
        use_container_width=True,
        hide_index=True,
    )

    fig = cpue_by_temp_chart(filtered)
    if fig is not None:
        st.plotly_chart(
            fig,
            use_container_width=True,
        )
    else:
        st.info(
            "수온별 CPUE 그래프를 만들 수 있는 "
            "유효 데이터가 부족합니다."
        )


with tab_hotspot:
    st.subheader("Historical Hotspot Explorer")
    st.warning(
        "이 점수는 과거 조업기록의 성공률·CPUE·표본수를 "
        "요약한 탐색 점수이며, 미래 어장 예측값이 아닙니다."
    )

    min_sets = st.slider(
        "격자 최소 투망횟수",
        min_value=1,
        max_value=30,
        value=5,
    )

    grid = hotspot_grid(
        filtered,
        min_sets=min_sets,
    )
    map_fig = historical_hotspot_map(grid)

    if map_fig is not None:
        st.plotly_chart(
            map_fig,
            use_container_width=True,
        )

        top_cols = [
            "grid_id",
            "sets",
            "success_rate",
            "mean_cpue",
            "catch_total",
            "historical_score",
            "last_date",
        ]
        top = grid[top_cols].head(30).copy()
        top["success_rate"] = (
            top["success_rate"]
            .map(lambda x: f"{x:.1%}")
        )

        st.dataframe(
            top,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "지도화할 유효 좌표/투망 데이터가 없습니다."
        )


with tab_model:
    st.subheader("CatBoost 시간분할 Baseline")
    st.caption(
        "모델은 현재 데이터에 신호가 존재하는지 확인하기 위한 기준선입니다. "
        "AUC/MAE가 낮으면 외부 환경·장비 데이터가 필요하다는 근거가 됩니다."
    )

    if st.button(
        "Baseline 학습 실행",
        type="primary",
    ):
        try:
            with st.spinner(
                "시간순 학습/검증 중..."
            ):
                result = train_baseline(filtered)

            st.success(
                f"완료: {result.split_note}"
            )

            m = result.metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Train",
                f"{m['train_rows']:,}",
            )
            c2.metric(
                "Test",
                f"{m['test_rows']:,}",
            )
            c3.metric(
                "Success AUC",
                (
                    "N/A"
                    if pd.isna(m["auc"])
                    else f"{m['auc']:.3f}"
                ),
            )
            c4.metric(
                "Positive Catch MAE",
                (
                    "N/A"
                    if pd.isna(
                        m["mae_positive_catch"]
                    )
                    else (
                        f"{m['mae_positive_catch']:.2f}"
                    )
                ),
            )

            st.markdown(
                "#### Feature Importance"
            )
            fi_fig = px.bar(
                result.feature_importance,
                x="importance",
                y="feature",
                orientation="h",
            )
            fi_fig.update_layout(
                yaxis={
                    "categoryorder":
                    "total ascending"
                }
            )
            st.plotly_chart(
                fi_fig,
                use_container_width=True,
            )

            st.markdown(
                "#### 검증 데이터 일부"
            )
            cols = [
                "date",
                "vessel",
                "captain",
                "fishing_ground",
                "method",
                "catch_total",
                "is_success",
                "success_prob",
                "pred_catch_if_success",
                "model_score",
            ]
            preview = (
                result.scored_test[cols]
                .sort_values(
                    "model_score",
                    ascending=False,
                )
                .head(100)
            )
            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(
                f"Baseline 학습 실패: {exc}"
            )


st.divider()
st.caption(
    "Next: SST/SST gradient/Chl-a/current/wind/wave 데이터 결합 "
    "→ 24~48시간 후보 해역 ranking 검증"
)
