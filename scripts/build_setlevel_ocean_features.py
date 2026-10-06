from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocean.extractor import (
    build_ocean_feature_store,
    feature_coverage,
)
from src.ocean.store import (
    SETLEVEL_CACHE_DIR,
    SETLEVEL_FEATURE_PATH,
)
from src.setlevel import (
    load_setlevel_2026_excel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "26년 운항.xlsx의 26어획 set-level 좌표에 "
            "Copernicus Marine 일자료를 결합합니다."
        )
    )
    parser.add_argument(
        "--input",
        default="data/26년 운항.xlsx",
        help="26년 운항.xlsx 경로",
    )
    parser.add_argument(
        "--output",
        default=str(
            SETLEVEL_FEATURE_PATH
        ),
        help="set-level ocean Parquet 저장 경로",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(
            SETLEVEL_CACHE_DIR
        ),
        help="set-level Copernicus 캐시 경로",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트용 최대 이벤트 수",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 캐시를 무시하고 재조회",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"입력 파일이 없습니다: {input_path}"
        )

    print("[1/4] 26어획 set-level 로딩")
    events = load_setlevel_2026_excel(
        input_path
    )

    if args.limit:
        events = (
            events.sort_values("date")
            .head(args.limit)
            .copy()
        )
        print(
            f"      테스트 제한: {len(events):,}건"
        )

    valid = events[
        events["date"].notna()
        & events["lat"].notna()
        & events["lon"].notna()
    ]
    print(
        f"[2/4] 유효 날짜/좌표: "
        f"{len(valid):,} / {len(events):,}"
    )

    def progress(
        current: int,
        total: int,
        label: str,
    ) -> None:
        print(
            f"[3/4] {current:>3}/{total:<3} "
            f"{label}"
        )

    features = build_ocean_feature_store(
        fishing_df=events,
        output_path=args.output,
        cache_dir=args.cache_dir,
        force=args.force,
        progress=progress,
    )

    print("[4/4] 완료")
    print(f"      저장: {args.output}")
    print(f"      rows: {len(features):,}")

    coverage = feature_coverage(
        features
    )
    if not coverage.empty:
        print()
        print(
            coverage.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
