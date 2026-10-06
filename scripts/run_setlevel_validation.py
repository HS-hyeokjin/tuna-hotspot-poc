from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocean.store import (
    SETLEVEL_FEATURE_PATH,
    attach_feature_store,
)
from src.setlevel import (
    load_2026_operation_counts,
    load_setlevel_2026_excel,
    setlevel_coverage,
)
from src.setlevel_validation import (
    run_setlevel_suite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "26년 운항.xlsx의 좌표 어획 이벤트를 대상으로 "
            "positive-only set-level 회귀/Ranking 검증을 실행합니다."
        )
    )
    parser.add_argument(
        "--input",
        default="data/26년 운항.xlsx",
        help="26년 운항.xlsx 경로",
    )
    parser.add_argument(
        "--with-ocean",
        action="store_true",
        help=(
            "setlevel_2026_ocean_features.parquet을 "
            "결합해 Ocean 모델도 실행"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=220,
        help="CatBoost iterations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"입력 파일이 없습니다: {input_path}"
        )

    events = load_setlevel_2026_excel(
        input_path
    )
    operations = load_2026_operation_counts(
        input_path
    )
    summary, by_method = setlevel_coverage(
        events,
        operations,
    )

    print("[Set-level Coverage]")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("[By method]")
    print(
        by_method.to_string(
            index=False
        )
    )

    include_ocean = False
    if args.with_ocean:
        if not SETLEVEL_FEATURE_PATH.exists():
            raise SystemExit(
                "set-level Ocean Feature Store가 없습니다. "
                "먼저 build_setlevel_ocean_features.py를 실행하세요."
            )
        events, _ = attach_feature_store(
            events,
            SETLEVEL_FEATURE_PATH,
        )
        include_ocean = True

    print()
    print("[Positive-event validation]")
    result, _ = run_setlevel_suite(
        events,
        include_ocean=include_ocean,
        target_col="catch_total",
        iterations=args.iterations,
    )
    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        "주의: 실패 투망 GPS가 없으므로 "
        "이 결과는 성공확률이 아니라 양수 어획 이벤트 내부의 "
        "어획량/Ranking 검증입니다."
    )


if __name__ == "__main__":
    main()
