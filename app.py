from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src.analysis import (
    annual_summary,
    daily_species_consistency,
    hotspot_grid,
    kpi_summary,
    method_summary,
)
from src.charts import (
    catch_by_year_chart,
    cpue_by_temp_chart,
    historical_hotspot_map,
)
from src.config import DEFAULT_DATA_PATH
from src.loader import load_fishing_excel
from src.model_validation import (
    ABLATION_GROUPS,
    EXPERIMENTS,
    SPECIES_TARGETS,
    experiment_catalog,
    run_experiment_suite,
    run_ocean_ablation,
    run_ocean_ablation_walk_forward,
    run_species_suite,
    walk_forward_validate,
)
from src.ocean.config import (
    OCEAN_FEATURE_COLUMNS,
    dataset_catalog_rows,
)
from src.ocean.store import (
    DEFAULT_FEATURE_PATH,
    attach_feature_store,
    load_feature_store,
)
from src.preprocessing import (
    coordinate_audit_summary,
    coordinate_minute_distribution,
    preprocess_fishing_data,
    quality_report,
)


APP_VERSION = "0.3.2"


st.set_page_config(
    page_title="Tuna Hotspot AI PoC",
    page_icon="🐟",
    layout="wide",
)

st.title("🐟 Tuna Hotspot AI PoC")
st.caption(
    f"v{APP_VERSION} · Coordinate Precision Audit · "
    "어장 예측 가능성 검증 PoC"
)
st.info(
    "V3.2는 원본 조업좌표가 1° 단위가 아니라 S0925/E15958 같은 "
    "도분(DMM) 형식이라는 점을 명확히 검증합니다. "
    "모델과 Copernicus 매칭은 변환된 decimal-degree 원좌표를 사용하고, "
    "Historical Hotspot 격자는 시각화/집계 크기로만 사용합니다."
)


@st.cache_data(show_spinner=False)
def load_and_prepare(
    uploaded_bytes: bytes | None,
    local_path: str | None,
):
    if uploaded_bytes is not None:
        raw = load_fishing_excel(
            io.BytesIO(uploaded_bytes)
        )
    elif local_path:
        raw = load_fishing_excel(local_path)
    else:
        raise ValueError(
            "데이터 소스가 없습니다."
        )

    return preprocess_fishing_data(raw)


@st.cache_data(show_spinner=False)
def load_ocean_feature_view(
    path_str: str,
    mtime_ns: int,
) -> pd.DataFrame:
    _ = mtime_ns
    return load_feature_store(path_str)


def fmt_num(
    value,
    digits: int = 3,
    suffix: str = "",
) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def clear_model_state() -> None:
    for key in [
        "v2_suite_summary",
        "v2_suite_results",
        "v2_suite_signature",
        "v2_species_summary",
        "v2_species_results",
        "v2_species_signature",
        "v2_walk_forward",
        "v2_walk_forward_signature",
        "v31_ablation_summary",
        "v31_ablation_signature",
        "v31_ablation_walk",
        "v31_ablation_walk_signature",
    ]:
        st.session_state.pop(key, None)


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
    st.caption(
        "사내 원본 데이터는 GitHub에 커밋하지 않습니다."
    )


source_bytes = (
    uploaded.getvalue()
    if uploaded
    else None
)
local_source = (
    str(DEFAULT_DATA_PATH)
    if (
        uploaded is None
        and use_local
        and DEFAULT_DATA_PATH.exists()
    )
    else None
)

if (
    source_bytes is None
    and local_source is None
):
    st.warning(
        "왼쪽에서 Excel 파일을 업로드하거나 "
        "`data/선망_조업보고 데이터.xlsx`를 로컬에 배치하세요."
    )
    st.stop()

try:
    df = load_and_prepare(
        source_bytes,
        local_source,
    )
except Exception as exc:
    st.error(
        f"데이터 로딩 실패: {exc}"
    )
    st.stop()

ocean_loaded = False
ocean_coverage = pd.DataFrame()
ocean_load_error = None
ocean_signature = None

if DEFAULT_FEATURE_PATH.exists():
    try:
        df, ocean_coverage = attach_feature_store(
            df,
            DEFAULT_FEATURE_PATH,
        )
        ocean_loaded = True
        ocean_signature = (
            DEFAULT_FEATURE_PATH
            .stat()
            .st_mtime_ns
        )
    except Exception as exc:
        ocean_load_error = str(exc)

with st.sidebar:
    st.divider()
    st.header("Copernicus")
    if ocean_loaded:
        st.success(
            "외부 해양 특징 결합됨"
        )
        st.caption(
            str(DEFAULT_FEATURE_PATH)
        )
    elif ocean_load_error:
        st.error(
            "외부 특징 파일 로딩 실패"
        )
        st.caption(ocean_load_error)
    else:
        st.info(
            "외부 특징 파일 없음"
        )
        st.caption(
            "Ocean Data 탭의 명령으로 생성하세요."
        )


with st.sidebar:
    st.divider()
    st.header("공통 필터")

    years = sorted(
        int(x)
        for x in df["year"]
        .dropna()
        .unique()
    )
    selected_years = st.multiselect(
        "연도",
        years,
        default=years,
    )

    methods = sorted(
        str(x)
        for x in df["method"]
        .dropna()
        .unique()
    )
    selected_methods = st.multiselect(
        "조업방법",
        methods,
        default=methods,
    )

    vessels = sorted(
        str(x)
        for x in df["vessel"]
        .dropna()
        .unique()
    )
    selected_vessels = st.multiselect(
        "선박",
        vessels,
        default=vessels,
    )

    st.divider()
    if st.button(
        "AI 결과 초기화",
        use_container_width=True,
    ):
        clear_model_state()
        st.success("초기화 완료")


filtered = df[
    df["year"].isin(selected_years)
    & df["method"].isin(
        selected_methods
    )
    & df["vessel"].isin(
        selected_vessels
    )
].copy()

if filtered.empty:
    st.warning(
        "필터 결과가 없습니다."
    )
    st.stop()

data_signature = (
    len(filtered),
    str(filtered["date"].min()),
    str(filtered["date"].max()),
    tuple(selected_years),
    tuple(selected_methods),
    tuple(selected_vessels),
    ocean_signature,
)

summary = kpi_summary(filtered)

(
    tab_overview,
    tab_quality,
    tab_species,
    tab_method,
    tab_hotspot,
    tab_ocean,
    tab_ablation,
    tab_lab,
    tab_walk,
) = st.tabs(
    [
        "개요",
        "품질/좌표",
        "어종 분석",
        "조업방법/CPUE",
        "Historical Hotspot",
        "Ocean Data",
        "Ocean Ablation",
        "AI 실험실",
        "Walk-forward",
    ]
)


with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "전체 기록",
        f"{summary['rows']:,}",
    )
    c2.metric(
        "투망 기록",
        f"{summary['set_rows']:,}",
    )
    c3.metric(
        "성공 투망",
        f"{summary['success_rows']:,}",
    )
    c4.metric(
        "투망 성공률",
        f"{summary['success_rate']:.1%}",
    )
    c5.metric(
        "평균 CPUE",
        f"{summary['mean_cpue']:.1f}",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "선박",
        f"{summary['vessels']:,}",
    )
    c2.metric(
        "선장",
        f"{summary['captains']:,}",
    )
    c3.metric(
        "어장명",
        f"{summary['grounds']:,}",
    )

    date_text = "-"
    if (
        pd.notna(
            summary["start_date"]
        )
        and pd.notna(
            summary["end_date"]
        )
    ):
        date_text = (
            f"{summary['start_date']:%Y-%m-%d} ~ "
            f"{summary['end_date']:%Y-%m-%d}"
        )
    c4.metric(
        "기간",
        date_text,
    )

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
        "수온·조류 범위는 오류 확정이 아니라 확인 필요 항목입니다. "
        "특히 조류 컬럼의 단위/정의가 확정되기 전에는 값이 크다고 자동 제거하지 않습니다."
    )

    q = quality_report(filtered)
    st.dataframe(
        q,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 좌표 정밀도 감사"
    )
    coord_audit = coordinate_audit_summary(
        filtered
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "유효 좌표쌍",
        (
            f"{coord_audit['valid_pairs']:,} / "
            f"{coord_audit['rows']:,}"
        ),
    )
    c2.metric(
        "좌표쌍 유효율",
        (
            f"{coord_audit['valid_pair_rate']:.1%}"
            if pd.notna(
                coord_audit[
                    "valid_pair_rate"
                ]
            )
            else "N/A"
        ),
    )
    c3.metric(
        "위도 분(minute) 종류",
        (
            f"{coord_audit['lat_minute_unique']} / 60"
        ),
    )
    c4.metric(
        "경도 분(minute) 종류",
        (
            f"{coord_audit['lon_minute_unique']} / 60"
        ),
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "투망행 유효 좌표율",
        (
            f"{coord_audit['valid_set_pair_rate']:.1%}"
            if pd.notna(
                coord_audit[
                    "valid_set_pair_rate"
                ]
            )
            else "N/A"
        ),
    )
    c2.metric(
        "1행 2회 이상 투망",
        (
            f"{coord_audit['multi_set_rows']:,} "
            f"({coord_audit['multi_set_rate']:.1%})"
            if pd.notna(
                coord_audit[
                    "multi_set_rate"
                ]
            )
            else "N/A"
        ),
    )
    c3.metric(
        "좌표 표현 해상도",
        (
            "약 "
            f"{coord_audit['encoded_lat_resolution_km']:.2f} km / 1′"
        ),
    )

    st.success(
        "원본 좌표는 S0925 = S 09°25′, "
        "E15958 = E 159°58′ 형태입니다. "
        "분(minute) 정보가 존재하므로 원본을 1° 단위 좌표로 "
        "해석하면 안 됩니다."
    )
    st.info(
        "AI 모델과 Copernicus Feature Store는 lat/lon의 "
        "decimal-degree 변환값을 사용합니다. "
        "기존 lat_grid/lon_grid 1° 값은 Historical Hotspot "
        "집계를 위해 만든 파생값이었으며 모델 입력 좌표가 아닙니다."
    )
    st.warning(
        "다만 좌표 표기가 1′ 단위라는 것과 실제 투망 GPS 정확도는 "
        "같은 의미가 아닙니다. 특히 한 행에 투망이 2회 이상 기록되면 "
        "여러 set에 좌표 하나만 존재하므로 set별 정확한 위치는 알 수 없습니다."
    )

    valid_coord_example = (
        filtered[
            filtered["lat"].notna()
            & filtered["lon"].notna()
        ][
            [
                "date",
                "vessel",
                "lat_raw",
                "lon_raw",
                "lat",
                "lon",
                "set_count",
            ]
        ]
        .head(10)
    )
    with st.expander(
        "원본 좌표 → decimal degree 예시"
    ):
        st.dataframe(
            valid_coord_example,
            use_container_width=True,
            hide_index=True,
        )

    minute_dist = coordinate_minute_distribution(
        filtered
    )
    with st.expander(
        "분(minute) 00~59 사용 분포"
    ):
        st.dataframe(
            minute_dist,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "### 당일 총어획량 ↔ 어종별 합계 확인"
    )
    st.caption(
        "원본 '당일' 컬럼이 공식 일일 총어획량이라는 가정하에 "
        "선박·일자 단위로 어종별 합계와 비교합니다. "
        "컬럼 정의가 다르면 이 비교는 해석하지 않아야 합니다."
    )

    consistency = daily_species_consistency(
        filtered
    )
    if not consistency.empty:
        exact_rate = (
            consistency["exact_match"].mean()
        )
        median_gap = (
            consistency["abs_difference"]
            .median()
        )
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "비교 가능 선박·일",
            f"{len(consistency):,}",
        )
        c2.metric(
            "정확 일치 비율",
            f"{exact_rate:.1%}",
        )
        c3.metric(
            "차이 중앙값",
            f"{median_gap:.2f}",
        )

        with st.expander(
            "차이가 큰 기록 확인"
        ):
            st.dataframe(
                consistency.head(100),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "'당일' 컬럼과 비교 가능한 데이터가 없습니다."
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

    st.warning(
        "PA / S/H(PS) / Y/F(PS) / Y/F(GG) 및 중량구간의 정확한 "
        "사내 정의는 별도 확인이 필요합니다. "
        "V2에서는 정의가 확정되지 않은 S/H(PS)를 S/J에 임의 합산하지 않습니다."
    )


with tab_species:
    st.subheader("어종 그룹별 분석")

    species = filtered[
        filtered["is_set"]
    ].copy()

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
    long["species"] = (
        long["species"].map(
            label_map
        )
    )

    species_summary = (
        long.groupby(
            "species",
            as_index=False,
        )
        .agg(
            records=("catch", "size"),
            positive_records=(
                "catch",
                lambda s: int(
                    (s > 0).sum()
                ),
            ),
            total_catch=("catch", "sum"),
            mean_catch=("catch", "mean"),
        )
    )
    species_summary[
        "positive_rate"
    ] = (
        species_summary[
            "positive_records"
        ]
        / species_summary["records"]
    )

    st.dataframe(
        species_summary.sort_values(
            "total_catch",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    positive_species = long[
        long["catch"] > 0
    ]
    if not positive_species.empty:
        fig = px.box(
            positive_species,
            x="species",
            y="catch",
            points=False,
            log_y=True,
            title=(
                "어종 그룹별 양수 어획량 "
                "(로그축)"
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    st.markdown(
        "### 어종별 AI 모델 비교"
    )
    st.caption(
        "동일한 전체 변수 모델에서 S/J, Y/F, B/E의 "
        "'해당 어종이 잡혔는가 + 잡혔다면 얼마나 잡혔는가'를 각각 검증합니다. "
        "표본이 부족한 어종은 자동으로 제외됩니다."
    )

    if st.button(
        "어종별 모델 비교 실행",
        key="run_species_suite",
    ):
        with st.spinner(
            "어종별 모델 검증 중..."
        ):
            species_ml, species_results = (
                run_species_suite(
                    filtered,
                    experiment_key="all_full",
                )
            )
        st.session_state[
            "v2_species_summary"
        ] = species_ml
        st.session_state[
            "v2_species_results"
        ] = species_results
        st.session_state[
            "v2_species_signature"
        ] = data_signature

    if (
        st.session_state.get(
            "v2_species_signature"
        )
        == data_signature
        and "v2_species_summary"
        in st.session_state
    ):
        st.dataframe(
            st.session_state[
                "v2_species_summary"
            ],
            use_container_width=True,
            hide_index=True,
        )


with tab_method:
    st.subheader(
        "조업방법별 성과와 CPUE"
    )
    st.caption(
        "조업방법은 어획 성공과 강하게 연결될 수 있으므로 "
        "환경효과와 섞어서 해석하지 않습니다."
    )

    method_df = method_summary(
        filtered
    )
    method_df_display = (
        method_df.copy()
    )

    if not method_df_display.empty:
        method_df_display[
            "success_rate"
        ] = (
            method_df_display[
                "success_rate"
            ].map(
                lambda x: f"{x:.1%}"
            )
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
            "수온별 CPUE 그래프를 만들 수 있는 유효 데이터가 부족합니다."
        )


with tab_hotspot:
    st.subheader(
        "Historical Hotspot Explorer"
    )
    st.warning(
        "이 점수는 과거 조업기록의 성공률·CPUE·표본수를 요약한 "
        "탐색 점수이며, 미래 어장 예측값이 아닙니다."
    )

    grid_deg = st.select_slider(
        "집계 격자 크기",
        options=[
            0.1,
            0.25,
            0.5,
            1.0,
        ],
        value=0.25,
        format_func=lambda x: (
            f"{x:g}° "
            f"(약 {x * 111:.0f} km)"
        ),
    )
    st.caption(
        "이 설정은 Historical Hotspot 표시용 집계 크기입니다. "
        "모델/Copernicus는 원본 도분 좌표를 decimal degree로 "
        "변환한 값을 그대로 사용합니다."
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
        grid_deg=grid_deg,
    )
    map_fig = historical_hotspot_map(
        grid
    )

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
        top = (
            grid[top_cols]
            .head(30)
            .copy()
        )
        top["success_rate"] = (
            top["success_rate"].map(
                lambda x: f"{x:.1%}"
            )
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


with tab_ocean:
    st.subheader(
        "Copernicus Marine 외부 해양데이터"
    )

    st.dataframe(
        pd.DataFrame(
            dataset_catalog_rows()
        ),
        use_container_width=True,
        hide_index=True,
    )

    if ocean_loaded:
        st.success(
            "외부 특징 파일이 현재 조업데이터에 결합되어 있습니다."
        )

        if not ocean_coverage.empty:
            st.markdown(
                "### Feature Coverage"
            )
            st.dataframe(
                ocean_coverage,
                use_container_width=True,
                hide_index=True,
            )

        available = [
            col
            for col in OCEAN_FEATURE_COLUMNS
            if col in filtered.columns
            and filtered[col].notna().any()
        ]

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "해양 Feature",
            f"{len(available)} / {len(OCEAN_FEATURE_COLUMNS)}",
        )
        if "ocean_sst" in filtered:
            c2.metric(
                "SST 연결",
                f"{filtered['ocean_sst'].notna().mean():.1%}",
            )
        if "ocean_chl" in filtered:
            c3.metric(
                "Chl-a 연결",
                f"{filtered['ocean_chl'].notna().mean():.1%}",
            )

        if (
            "ocean_sst" in filtered
            and filtered["ocean_sst"].notna().any()
            and filtered["water_temp"].notna().any()
        ):
            compare = filtered[
                filtered["ocean_sst"].notna()
                & filtered["water_temp"].notna()
                & filtered["water_temp"].between(
                    5,
                    40,
                )
            ].copy()

            if not compare.empty:
                compare["sst_gap"] = (
                    compare["water_temp"]
                    - compare["ocean_sst"]
                )

                st.markdown(
                    "### 선박 기록 수온 vs Copernicus SST"
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "비교 건수",
                    f"{len(compare):,}",
                )
                c2.metric(
                    "평균 차이",
                    f"{compare['sst_gap'].mean():.2f} ℃",
                )
                c3.metric(
                    "차이 중앙값",
                    f"{compare['sst_gap'].median():.2f} ℃",
                )

                sst_fig = px.scatter(
                    compare.sample(
                        min(
                            len(compare),
                            1500,
                        ),
                        random_state=42,
                    ),
                    x="ocean_sst",
                    y="water_temp",
                    hover_data=[
                        "date",
                        "vessel",
                        "fishing_ground",
                    ],
                    title=(
                        "Copernicus SST와 "
                        "조업일지 수온 비교"
                    ),
                )
                st.plotly_chart(
                    sst_fig,
                    use_container_width=True,
                )

        if available:
            feature_cols = [
                "date",
                "vessel",
                "lat",
                "lon",
                *available,
            ]
            with st.expander(
                "외부 특징 데이터 예시"
            ):
                st.dataframe(
                    filtered[
                        feature_cols
                    ].dropna(
                        how="all",
                        subset=available,
                    ).head(100),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning(
                "외부 특징 파일은 있으나 유효한 해양 Feature가 없습니다. "
                "생성 로그의 ocean_error를 확인하세요."
            )

        st.markdown("### Parquet Data Viewer")
        st.caption(
            "fishing_ocean_features.parquet 원본을 화면에서 확인합니다. "
            "이 표는 예측결과가 아니라 조업 날짜·좌표에 붙인 AI 입력 Feature입니다."
        )

        column_guide = pd.DataFrame(
            [
                ["source_row_id", "원본 조업행 연결 ID", "-"],
                ["date", "조업 날짜", "date"],
                ["lat / lon", "조업 위도 / 경도", "degree"],
                ["ocean_sst", "Copernicus 해수면 수온", "℃"],
                ["ocean_current_u", "동서방향 해류 성분", "m/s"],
                ["ocean_current_v", "남북방향 해류 성분", "m/s"],
                ["ocean_current_speed", "U/V를 합친 해류 속력", "m/s"],
                ["ocean_current_dir_deg", "해류 방향각", "degree"],
                ["ocean_ssh", "해수면 높이(SSH)", "m"],
                ["ocean_chl", "Chlorophyll-a 농도", "제품 단위"],
                [
                    "ocean_sst_gradient_c_per_100km",
                    "주변 수온 변화량 근사치",
                    "℃ / 100km",
                ],
                ["ocean_complete", "핵심 해양 Feature 확보 여부", "bool"],
                ["ocean_error", "부분 조회 실패/오류 기록", "text"],
            ],
            columns=["컬럼", "뜻", "단위"],
        )
        with st.expander(
            "컬럼 뜻 보기",
            expanded=False,
        ):
            st.dataframe(
                column_guide,
                use_container_width=True,
                hide_index=True,
            )

        try:
            ocean_store = load_ocean_feature_view(
                str(DEFAULT_FEATURE_PATH),
                int(ocean_signature or 0),
            )
            view = ocean_store.copy()

            c1, c2, c3 = st.columns(3)
            complete_filter = c1.selectbox(
                "데이터 상태",
                ["전체", "완료만", "미완료만"],
                key="ocean_view_complete",
            )
            error_only = c2.checkbox(
                "오류 기록만",
                value=False,
                key="ocean_view_error_only",
            )
            view_limit = c3.selectbox(
                "화면 표시 행수",
                [50, 100, 300, 1000],
                index=1,
                key="ocean_view_limit",
            )

            if (
                complete_filter != "전체"
                and "ocean_complete" in view.columns
            ):
                complete_mask = (
                    view["ocean_complete"].fillna(False)
                    == True
                )
                if complete_filter == "완료만":
                    view = view[complete_mask]
                else:
                    view = view[~complete_mask]

            if (
                error_only
                and "ocean_error" in view.columns
            ):
                error_text = (
                    view["ocean_error"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
                view = view[error_text != ""]

            st.caption(
                f"필터 결과 {len(view):,}건 / "
                f"전체 {len(ocean_store):,}건"
            )
            st.dataframe(
                view.head(int(view_limit)),
                use_container_width=True,
                hide_index=True,
            )

            csv_bytes = view.to_csv(
                index=False,
            ).encode("utf-8-sig")
            st.download_button(
                "현재 필터 결과 CSV 다운로드",
                data=csv_bytes,
                file_name="ocean_features_filtered.csv",
                mime="text/csv",
                key="download_ocean_csv",
            )
        except Exception as exc:
            st.error(
                f"Parquet Viewer 로딩 실패: {exc}"
            )
    else:
        st.warning(
            "아직 외부 특징 파일이 없습니다. "
            "Copernicus 로그인 성공 상태에서 아래 명령을 실행하세요."
        )
        st.code(
            "python scripts/build_ocean_features.py --limit 100",
            language="powershell",
        )
        st.caption(
            "먼저 100건으로 연결을 확인한 뒤 문제가 없으면 전체 생성합니다."
        )
        st.code(
            "python scripts/build_ocean_features.py",
            language="powershell",
        )
        st.caption(
            "완료 파일: "
            "data/external/processed/fishing_ocean_features.parquet"
        )


with tab_ablation:
    st.subheader(
        "Ocean Ablation · 어떤 해양변수가 실제로 도움됐나?"
    )
    st.write(
        "각 해양변수마다 Copernicus 값이 존재하는 동일 행을 고정한 뒤, "
        "① 기존 V2 변수만 사용한 모델과 ② 해당 해양변수를 추가한 모델을 비교합니다. "
        "따라서 표본 차이 때문에 성능이 달라지는 문제를 줄입니다."
    )
    st.caption(
        "delta_auc > 0이면 해당 해양변수 추가 후 ROC-AUC가 상승, "
        "delta_top10 > 0이면 상위 10% 어획 후보 선별력이 상승한 것입니다. "
        "delta_mae는 음수일수록 어획량 오차가 감소한 것입니다."
    )

    if not ocean_loaded:
        st.warning(
            "먼저 Ocean Data를 전체 생성해 Parquet을 연결하세요."
        )
    else:
        ablation_target_label = st.selectbox(
            "Ablation Target",
            list(SPECIES_TARGETS.keys()),
            index=0,
            key="ablation_target",
        )
        ablation_target_col = SPECIES_TARGETS[
            ablation_target_label
        ]

        ablation_keys = list(
            ABLATION_GROUPS.keys()
        )
        selected_ablation = st.multiselect(
            "비교할 해양변수",
            options=ablation_keys,
            default=ablation_keys,
            format_func=lambda k: (
                ABLATION_GROUPS[k]["label"]
            ),
            key="ablation_groups",
        )

        c1, c2 = st.columns(2)
        run_quick = c1.button(
            "2026 기준 Ablation 실행",
            type="primary",
            use_container_width=True,
            key="run_ablation",
        )
        run_walk = c2.button(
            "Walk-forward Ablation 실행",
            use_container_width=True,
            key="run_ablation_walk",
        )

        if not selected_ablation:
            st.info(
                "최소 1개 해양변수를 선택하세요."
            )

        if run_quick and selected_ablation:
            with st.spinner(
                "동일표본 기준 해양변수별 모델 비교 중..."
            ):
                ablation = run_ocean_ablation(
                    filtered,
                    target_col=ablation_target_col,
                    group_keys=selected_ablation,
                )
            st.session_state[
                "v31_ablation_summary"
            ] = ablation
            st.session_state[
                "v31_ablation_signature"
            ] = (
                data_signature,
                ablation_target_col,
                tuple(selected_ablation),
            )

        ablation_signature = (
            data_signature,
            ablation_target_col,
            tuple(selected_ablation),
        )
        if (
            st.session_state.get(
                "v31_ablation_signature"
            )
            == ablation_signature
            and "v31_ablation_summary"
            in st.session_state
        ):
            ablation = st.session_state[
                "v31_ablation_summary"
            ]
            st.markdown(
                "### 2026 동일표본 비교"
            )
            st.dataframe(
                ablation,
                use_container_width=True,
                hide_index=True,
            )

            completed_ablation = ablation[
                ablation["상태"] == "완료"
            ].copy()

            if not completed_ablation.empty:
                auc_long = completed_ablation.melt(
                    id_vars=["해양변수"],
                    value_vars=[
                        "baseline_auc",
                        "enhanced_auc",
                    ],
                    var_name="모델",
                    value_name="ROC-AUC",
                )
                lift_long = completed_ablation.melt(
                    id_vars=["해양변수"],
                    value_vars=[
                        "baseline_top10",
                        "enhanced_top10",
                    ],
                    var_name="모델",
                    value_name="Top10 Lift",
                )

                c1, c2 = st.columns(2)
                with c1:
                    auc_fig = px.bar(
                        auc_long,
                        x="해양변수",
                        y="ROC-AUC",
                        color="모델",
                        barmode="group",
                        title="동일표본 ROC-AUC",
                        range_y=[0, 1],
                    )
                    st.plotly_chart(
                        auc_fig,
                        use_container_width=True,
                    )
                with c2:
                    lift_fig = px.bar(
                        lift_long,
                        x="해양변수",
                        y="Top10 Lift",
                        color="모델",
                        barmode="group",
                        title="동일표본 Top10 Catch Lift",
                    )
                    lift_fig.add_hline(
                        y=1.0,
                        line_dash="dash",
                        annotation_text="전체 평균",
                    )
                    st.plotly_chart(
                        lift_fig,
                        use_container_width=True,
                    )

                delta_fig = px.bar(
                    completed_ablation,
                    x="해양변수",
                    y="delta_top10",
                    title=(
                        "해양변수 추가 전후 "
                        "Top10 Lift 변화량"
                    ),
                )
                delta_fig.add_hline(
                    y=0,
                    line_dash="dash",
                )
                st.plotly_chart(
                    delta_fig,
                    use_container_width=True,
                )

        if run_walk and selected_ablation:
            with st.spinner(
                "해양변수별 연도 Walk-forward 비교 중... "
                "여러 모델을 반복 학습하므로 시간이 걸릴 수 있습니다."
            ):
                ablation_walk = (
                    run_ocean_ablation_walk_forward(
                        filtered,
                        target_col=ablation_target_col,
                        group_keys=selected_ablation,
                    )
                )
            st.session_state[
                "v31_ablation_walk"
            ] = ablation_walk
            st.session_state[
                "v31_ablation_walk_signature"
            ] = ablation_signature

        if (
            st.session_state.get(
                "v31_ablation_walk_signature"
            )
            == ablation_signature
            and "v31_ablation_walk"
            in st.session_state
        ):
            ablation_walk = st.session_state[
                "v31_ablation_walk"
            ]
            st.markdown(
                "### 연도별 Walk-forward Ablation"
            )
            st.dataframe(
                ablation_walk,
                use_container_width=True,
                hide_index=True,
            )

            completed_walk = ablation_walk[
                ablation_walk["상태"] == "완료"
            ].copy()
            if not completed_walk.empty:
                available_groups = list(
                    completed_walk["key"]
                    .dropna()
                    .unique()
                )
                detail_group = st.selectbox(
                    "연도별 상세 해양변수",
                    available_groups,
                    format_func=lambda k: (
                        ABLATION_GROUPS[k]["label"]
                    ),
                    key="ablation_walk_detail",
                )
                detail = completed_walk[
                    completed_walk["key"]
                    == detail_group
                ].copy()

                detail_long = detail.melt(
                    id_vars=["test_year"],
                    value_vars=[
                        "baseline_top10",
                        "enhanced_top10",
                    ],
                    var_name="모델",
                    value_name="Top10 Lift",
                )
                wf_fig = px.line(
                    detail_long,
                    x="test_year",
                    y="Top10 Lift",
                    color="모델",
                    markers=True,
                    title=(
                        f"{ABLATION_GROUPS[detail_group]['label']} "
                        "연도별 Top10 Lift"
                    ),
                )
                wf_fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    annotation_text="전체 평균",
                )
                st.plotly_chart(
                    wf_fig,
                    use_container_width=True,
                )

                delta_walk_fig = px.bar(
                    detail,
                    x="test_year",
                    y="delta_top10",
                    title=(
                        f"{ABLATION_GROUPS[detail_group]['label']} "
                        "추가에 따른 연도별 Lift 변화"
                    ),
                )
                delta_walk_fig.add_hline(
                    y=0,
                    line_dash="dash",
                )
                st.plotly_chart(
                    delta_walk_fig,
                    use_container_width=True,
                )


with tab_lab:
    st.subheader(
        "AI 실험실 · 모델 분리 검증"
    )
    st.write(
        "전체 모델의 성능이 조업방법·선박·선장 때문에 나온 것인지, "
        "위치·환경 자체에도 신호가 있는지 같은 데이터로 비교합니다."
    )

    st.dataframe(
        experiment_catalog(),
        use_container_width=True,
        hide_index=True,
    )

    target_label = st.selectbox(
        "분석 Target",
        list(
            SPECIES_TARGETS.keys()
        ),
        index=0,
        key="lab_target",
    )
    target_col = SPECIES_TARGETS[
        target_label
    ]

    if st.button(
        "전체 실험 실행",
        type="primary",
        key="run_v2_suite",
    ):
        with st.spinner(
            "V2/V3 실험 모델 학습 및 검증 중..."
        ):
            suite_summary, suite_results = (
                run_experiment_suite(
                    filtered,
                    target_col=target_col,
                )
            )

        st.session_state[
            "v2_suite_summary"
        ] = suite_summary
        st.session_state[
            "v2_suite_results"
        ] = suite_results
        st.session_state[
            "v2_suite_signature"
        ] = (
            data_signature,
            target_col,
        )

    suite_valid = (
        st.session_state.get(
            "v2_suite_signature"
        )
        == (
            data_signature,
            target_col,
        )
        and "v2_suite_summary"
        in st.session_state
    )

    if suite_valid:
        suite_summary = st.session_state[
            "v2_suite_summary"
        ]
        suite_results = st.session_state[
            "v2_suite_results"
        ]

        st.markdown(
            "### 실험 결과 비교"
        )
        st.dataframe(
            suite_summary,
            use_container_width=True,
            hide_index=True,
        )

        completed = suite_summary[
            suite_summary["상태"]
            == "완료"
        ].copy()

        if not completed.empty:
            c1, c2 = st.columns(2)

            with c1:
                auc_fig = px.bar(
                    completed,
                    x="실험",
                    y="roc_auc",
                    title="ROC-AUC 비교",
                    range_y=[0, 1],
                )
                st.plotly_chart(
                    auc_fig,
                    use_container_width=True,
                )

            with c2:
                lift_fig = px.bar(
                    completed,
                    x="실험",
                    y="top10_lift",
                    title="Top 10% Catch Lift",
                )
                lift_fig.add_hline(
                    y=1.0,
                    line_dash="dash",
                    annotation_text="전체 평균",
                )
                st.plotly_chart(
                    lift_fig,
                    use_container_width=True,
                )

            st.caption(
                "ROC-AUC 0.5는 성공/실패 구분력이 거의 없는 수준입니다. "
                "Top 10% Lift 1.5라면 모델 점수 상위 10%의 실제 평균 어획량이 "
                "검증 전체 평균의 약 1.5배였다는 뜻입니다."
            )

            result_keys = list(
                suite_results.keys()
            )
            selected_key = st.selectbox(
                "상세 분석할 실험",
                result_keys,
                format_func=lambda k: (
                    EXPERIMENTS[k].label
                ),
                key="detail_experiment",
            )
            result = suite_results[
                selected_key
            ]
            m = result.metrics

            st.markdown(
                f"### {result.spec.label}"
            )
            st.caption(
                result.spec.description
                + " · "
                + result.split_note
            )

            c1, c2, c3, c4, c5 = (
                st.columns(5)
            )
            c1.metric(
                "ROC-AUC",
                fmt_num(m["roc_auc"]),
            )
            c2.metric(
                "PR-AUC",
                fmt_num(m["pr_auc"]),
            )
            c3.metric(
                "Brier",
                fmt_num(m["brier"]),
            )
            c4.metric(
                "MAE",
                fmt_num(
                    m[
                        "mae_positive_catch"
                    ],
                    2,
                ),
            )
            c5.metric(
                "Top10 Lift",
                fmt_num(
                    m[
                        "catch_lift_top10"
                    ],
                    2,
                    "x",
                ),
            )

            c1, c2 = st.columns(2)
            with c1:
                fi = result.feature_importance
                fi_fig = px.bar(
                    fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title=(
                        "CatBoost Feature "
                        "Importance"
                    ),
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

            with c2:
                shap_imp = (
                    result.shap_importance
                )
                if not shap_imp.empty:
                    shap_fig = px.bar(
                        shap_imp,
                        x="mean_abs_shap",
                        y="feature",
                        orientation="h",
                        title=(
                            "SHAP Global "
                            "Importance"
                        ),
                    )
                    shap_fig.update_layout(
                        yaxis={
                            "categoryorder":
                            "total ascending"
                        }
                    )
                    st.plotly_chart(
                        shap_fig,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "SHAP 결과가 없습니다."
                    )

            shap_detail = result.shap_detail
            if not shap_detail.empty:
                st.markdown(
                    "#### SHAP 방향 확인"
                )
                shap_features = list(
                    result.spec.features
                )
                shap_feature = (
                    st.selectbox(
                        "변수 선택",
                        shap_features,
                        key="shap_feature",
                    )
                )
                shap_col = (
                    f"shap__{shap_feature}"
                )

                if pd.api.types.is_numeric_dtype(
                    shap_detail[
                        shap_feature
                    ]
                ):
                    effect = px.scatter(
                        shap_detail,
                        x=shap_feature,
                        y=shap_col,
                        hover_data=[
                            "date",
                            "_target_catch",
                        ],
                        title=(
                            f"{shap_feature} 값에 따른 "
                            "성공예측 기여도"
                        ),
                    )
                    effect.add_hline(
                        y=0,
                        line_dash="dash",
                    )
                else:
                    cat_effect = (
                        shap_detail.groupby(
                            shap_feature,
                            as_index=False,
                        )
                        .agg(
                            mean_shap=(
                                shap_col,
                                "mean",
                            ),
                            records=(
                                shap_col,
                                "size",
                            ),
                        )
                        .sort_values(
                            "records",
                            ascending=False,
                        )
                        .head(20)
                    )
                    effect = px.bar(
                        cat_effect,
                        x=shap_feature,
                        y="mean_shap",
                        hover_data=[
                            "records"
                        ],
                        title=(
                            f"{shap_feature} 범주별 "
                            "평균 SHAP"
                        ),
                    )
                    effect.add_hline(
                        y=0,
                        line_dash="dash",
                    )

                st.plotly_chart(
                    effect,
                    use_container_width=True,
                )
                st.caption(
                    "SHAP가 양수면 해당 모델에서 성공 쪽으로, "
                    "음수면 실패 쪽으로 예측을 밀었다는 뜻입니다. "
                    "인과관계를 의미하지는 않습니다."
                )

            with st.expander(
                "검증 데이터 예측값 보기"
            ):
                preview_cols = [
                    "date",
                    "vessel",
                    "captain",
                    "fishing_ground",
                    "method",
                    "_target_catch",
                    "_target_success",
                    "success_prob",
                    "pred_catch_if_success",
                    "model_score",
                ]
                preview_cols = [
                    col
                    for col in preview_cols
                    if col
                    in result.scored_test.columns
                ]
                preview = (
                    result.scored_test[
                        preview_cols
                    ]
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


with tab_walk:
    st.subheader(
        "Walk-forward Validation"
    )
    st.write(
        "과거 연도만 학습하고 그 다음 연도를 예측하는 방식으로 "
        "연도별 재현성을 확인합니다. 한 번의 80/20 분할보다 미래 적용 가능성을 "
        "더 엄격하게 볼 수 있습니다."
    )

    walk_key = st.selectbox(
        "검증 실험",
        list(EXPERIMENTS.keys()),
        format_func=lambda k: (
            EXPERIMENTS[k].label
        ),
        key="walk_experiment",
    )
    walk_target_label = st.selectbox(
        "Walk-forward Target",
        list(
            SPECIES_TARGETS.keys()
        ),
        key="walk_target",
    )
    walk_target_col = (
        SPECIES_TARGETS[
            walk_target_label
        ]
    )

    if st.button(
        "Walk-forward 실행",
        key="run_walk_forward",
    ):
        with st.spinner(
            "연도별 순차 검증 중..."
        ):
            walk = walk_forward_validate(
                filtered,
                experiment_key=walk_key,
                target_col=walk_target_col,
            )

        st.session_state[
            "v2_walk_forward"
        ] = walk
        st.session_state[
            "v2_walk_forward_signature"
        ] = (
            data_signature,
            walk_key,
            walk_target_col,
        )

    if (
        st.session_state.get(
            "v2_walk_forward_signature"
        )
        == (
            data_signature,
            walk_key,
            walk_target_col,
        )
        and "v2_walk_forward"
        in st.session_state
    ):
        walk = st.session_state[
            "v2_walk_forward"
        ]

        if walk.empty:
            st.warning(
                "Walk-forward 검증이 가능한 연도별 표본이 부족합니다."
            )
        else:
            st.dataframe(
                walk,
                use_container_width=True,
                hide_index=True,
            )

            metric_long = walk.melt(
                id_vars=[
                    "test_year"
                ],
                value_vars=[
                    "roc_auc",
                    "pr_auc",
                ],
                var_name="metric",
                value_name="value",
            )
            metric_fig = px.line(
                metric_long,
                x="test_year",
                y="value",
                color="metric",
                markers=True,
                title=(
                    "연도별 분류 성능"
                ),
            )
            metric_fig.update_yaxes(
                range=[0, 1]
            )
            st.plotly_chart(
                metric_fig,
                use_container_width=True,
            )

            lift_fig = px.bar(
                walk,
                x="test_year",
                y="top10_lift",
                title=(
                    "연도별 Top 10% "
                    "Catch Lift"
                ),
            )
            lift_fig.add_hline(
                y=1.0,
                line_dash="dash",
                annotation_text="전체 평균",
            )
            st.plotly_chart(
                lift_fig,
                use_container_width=True,
            )


st.divider()
st.caption(
    "V3.2 판단 기준: 원본 좌표의 표현 정밀도와 set-level 위치 한계를 먼저 "
    "확인한 뒤, 동일표본 Ablation과 Walk-forward에서 해양변수의 "
    "delta_auc / delta_top10이 여러 연도에 걸쳐 반복되는지 확인합니다."
)
