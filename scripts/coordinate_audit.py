from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_fishing_excel
from src.preprocessing import (
    coordinate_audit_summary,
    coordinate_minute_distribution,
    preprocess_fishing_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "원본 조업 Excel의 도분(DMM) 좌표 정밀도와 "
            "set-level 좌표 한계를 점검합니다."
        )
    )
    parser.add_argument(
        "--input",
        default="data/선망_조업보고 데이터.xlsx",
        help="원본 조업 Excel 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(
            f"입력 파일이 없습니다: {input_path}"
        )

    raw = load_fishing_excel(input_path)
    df = preprocess_fishing_data(raw)
    summary = coordinate_audit_summary(df)

    print("[Coordinate Audit]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("[Minute distribution 00~59]")
    print(
        coordinate_minute_distribution(df)
        .to_string(index=False)
    )

    print()
    print(
        "주의: 1′ 단위 표기는 좌표 표현 해상도이며 "
        "실제 set별 GPS 정확도를 보장하지 않습니다."
    )
    print(
        "모델/Copernicus는 decimal-degree lat/lon을 사용하고, "
        "Historical Hotspot grid는 별도 집계용입니다."
    )


if __name__ == "__main__":
    main()
