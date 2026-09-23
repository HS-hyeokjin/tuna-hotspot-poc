from __future__ import annotations

import argparse
from pathlib import Path

from src.loader import load_fishing_excel
from src.ocean.extractor import (
    build_ocean_feature_store,
    feature_coverage,
)
from src.ocean.store import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FEATURE_PATH,
)
from src.preprocessing import (
    preprocess_fishing_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "조업 Excel의 날짜/좌표에 Copernicus Marine "
            "해양환경 특징을 결합해 Parquet으로 저장합니다."
        )
    )
    parser.add_argument(
        "--input",
        default="data/선망_조업보고 데이터.xlsx",
        help="원본 조업 Excel 경로",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_FEATURE_PATH),
        help="완성된 Parquet 저장 경로",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="월/공간 블록별 캐시 디렉터리",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트용 최대 행 수. 예: --limit 100",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 블록 캐시를 무시하고 다시 조회",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"입력 파일이 없습니다: {input_path}"
        )

    print("[1/4] 조업 Excel 로딩")
    raw = load_fishing_excel(input_path)
    fishing = preprocess_fishing_data(raw)

    if args.limit:
        fishing = (
            fishing.sort_values("date")
            .head(args.limit)
            .copy()
        )
        print(
            f"      테스트 제한: {len(fishing):,}건"
        )

    valid = fishing[
        fishing["date"].notna()
        & fishing["lat"].notna()
        & fishing["lon"].notna()
    ]
    print(
        f"[2/4] 유효 날짜/좌표: {len(valid):,} / {len(fishing):,}"
    )

    def progress(
        current: int,
        total: int,
        label: str,
    ) -> None:
        print(
            f"[3/4] {current:>3}/{total:<3} {label}"
        )

    features = build_ocean_feature_store(
        fishing_df=fishing,
        output_path=args.output,
        cache_dir=args.cache_dir,
        force=args.force,
        progress=progress,
    )

    print("[4/4] 완료")
    print(f"      저장: {args.output}")
    print(f"      rows: {len(features):,}")

    coverage = feature_coverage(features)
    if not coverage.empty:
        print()
        print(coverage.to_string(index=False))

    errors = features[
        features["ocean_error"].astype(str) != ""
    ]
    if not errors.empty:
        print()
        print(
            f"부분/실패 요청이 있는 행: {len(errors):,}"
        )
        print(
            errors[
                ["source_row_id", "date", "ocean_error"]
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
