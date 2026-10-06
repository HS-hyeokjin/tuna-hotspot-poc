# tuna-hotspot-poc

참치 선망 조업데이터와 Copernicus Marine 해양환경 데이터를 이용해 **어장 탐색에 실제로 활용 가능한 신호가 있는지 검증**하는 Python PoC입니다.

> 특정 좌표의 미래 어획을 확정하는 시스템이 아닙니다.
> 시간순 검증과 동일표본 비교로 어떤 데이터가 실제 예측력을 추가하는지 단계적으로 확인합니다.

## 현재 브랜치

`feature/v0.3.3-positive-set-validation`

- `main`: V1 baseline
- `feature/v0.2-model-validation`: 선박/선장/조업방법 효과 분리
- `feature/v0.3-ocean-data`: Copernicus Marine 외부 해양데이터 결합
- `feature/v0.3.1-ocean-ablation`: 해양변수별 Ablation + Parquet Viewer
- `feature/v0.3.2-coordinate-audit`: 원본 도분 좌표 정밀도 감사 + 가변 Hotspot grid
- `feature/v0.3.3-positive-set-validation`: 2026 개별 어획 이벤트 기반 양수 어획량/Ranking 검증

버전별 브랜치는 병합하지 않고 유지합니다.

## V3.3 핵심 질문

`26년 운항.xlsx`의 `26어획`에는 날짜·선박·조업방법·좌표·어획량이 함께 있는
개별 어획 이벤트가 존재합니다.

하지만 실패 투망의 GPS 위치와 정확한 투망시각은 현재 확보되지 않았습니다.
따라서 V3.3은 **실패 위치를 임의 생성하지 않고**, 다음 범위만 검증합니다.

- 좌표가 있는 양수 어획 이벤트 품질 확인
- `26조업` 보고 투망횟수와 좌표 이벤트 수의 gap 진단
- 성공 이벤트 내부의 어획량 회귀
- 성공 이벤트 내부의 Top10 / Top20 Ranking
- 위치·계절·방법 모델 vs 선박 컨텍스트 추가 모델 비교
- 선택적으로 set-level Copernicus 일자료 결합

V3.3에서는 **set-level ROC-AUC / 성공확률을 산출하지 않습니다.**
실패 투망 좌표가 없는 상태에서 negative 위치를 만들어 분류하면
모델 성능을 과대평가할 수 있기 때문입니다.

### Set-level 입력

로컬 파일:

    data/26년 운항.xlsx

Streamlit에서 직접 업로드할 수도 있습니다.

### Set-level Copernicus 생성

    python scripts/build_setlevel_ocean_features.py --input "data/26년 운항.xlsx"

생성 파일:

    data/external/processed/setlevel_2026_ocean_features.parquet

기존 2023~2026 조업보고용 Feature Store와 분리해서 관리합니다.

### V3.3 검증 구조

기본 시간분할:

    2026-01~06 학습
    2026-07~09 검증

모델:

- 위치·계절·방법
- 위치·계절·방법 + 선박
- 위 모델 + Copernicus Ocean (Feature Store가 있을 때)

지표:

- MAE
- Median AE
- Rank correlation
- Top10 Catch Lift
- Top20 Catch Lift

Baseline은 학습구간의 조업방법별 median 어획량입니다.

## V3.2 핵심 질문

원본 조업위치가 `S0925 / E15958`처럼 기록되어 있다면 이는
`S 09°25′ / E 159°58′`의 도분(DMM) 좌표입니다.

V3.2는 다음을 명확히 분리합니다.

- **원본 좌표 표현 정밀도**: 분(minute) 단위, 위도 기준 약 1.85 km/1′
- **AI 모델 좌표**: DMM을 decimal degree로 변환한 `lat` / `lon`
- **Copernicus 매칭 좌표**: 동일한 `lat` / `lon`
- **Historical Hotspot grid**: 시각화/집계용 별도 격자

즉 기존 `lat_grid` / `lon_grid` 1° 파생값은
모델 입력좌표가 아니며, 원본 데이터를 1° 단위로 뭉갠 것도 아닙니다.

다만 한 행에 여러 번 투망한 기록이 존재할 수 있으므로
**좌표 표기 해상도와 실제 set별 GPS 정확도는 동일하지 않습니다.**
V3.2의 Coordinate Audit은 이 차이를 수치로 확인합니다.

### Coordinate Audit 실행

    python scripts/coordinate_audit.py --input "data/선망_조업보고 데이터.xlsx"

Streamlit의 `품질/좌표` 탭에서도 다음을 확인할 수 있습니다.

- 유효 좌표쌍 비율
- 위도/경도 minute 00~59 사용 분포
- 좌표 표현 해상도
- 투망행 중 좌표 유효율
- 한 행에 2회 이상 투망한 비율
- 원본 좌표 → decimal degree 변환 예시

### Historical Hotspot grid

V3.2부터 Historical Hotspot의 집계 격자를 선택할 수 있습니다.

- 0.1°
- 0.25°
- 0.5°
- 1.0°

이 설정은 지도 집계 해상도만 변경하며
AI 모델과 Copernicus Feature Store의 좌표는 변경하지 않습니다.

## V3.1 핵심 질문

V3 결과에서 외부 해양변수를 모두 넣었을 때 ROC-AUC가 크게 개선되지는 않았지만 일부 연도의 Top10 Catch Lift가 개선되었습니다.

V3.1은 다음 질문을 검증합니다.

> SST, Current, SSH, Chlorophyll-a, SST Gradient 중 어떤 변수가 실제로 후보 어장 Ranking을 개선했는가?

## 동일표본 Ablation

각 실험은 해당 외부 변수가 존재하는 동일한 조업행을 사용합니다.

예:

    SST 동일표본
      ├─ Baseline: 기존 V2 변수만 사용
      └─ Enhanced: 기존 V2 + SST

따라서 두 모델은 동일한 train/test 행으로 비교됩니다.

실험 그룹:

- SST
- Current (U/V/Speed)
- SSH
- Chl-a
- SST Gradient
- SST + Current
- Current + Chl-a
- All Ocean

평가지표:

- baseline / enhanced ROC-AUC
- delta_auc
- baseline / enhanced Top10 Catch Lift
- delta_top10
- baseline / enhanced MAE
- delta_mae

해석:

- `delta_auc > 0`: 성공/실패 구분 성능 상승
- `delta_top10 > 0`: 상위 어획 후보 Ranking 개선
- `delta_mae < 0`: 양수 어획량 예측오차 감소

## Walk-forward Ablation

해양변수별로 다음 구조를 반복합니다.

    2023      -> 2024 검증
    2023~2024 -> 2025 검증
    2023~2025 -> 2026 검증

한 연도에서만 좋아진 변수보다 여러 연도에서 delta_top10 / delta_auc가 반복되는 변수를 우선 확인합니다.

AI 실험실과 Walk-forward의 CatBoost iterations는 V3.1에서 220으로 통일했습니다.

## Ocean Data Viewer

Streamlit의 `Ocean Data` 탭에서 다음 기능을 제공합니다.

- Parquet 원본 내용 조회
- 컬럼 의미/단위 설명
- 완료/미완료 데이터 필터
- 오류 기록만 필터
- 표시 행 수 선택
- 현재 필터 결과 CSV 다운로드

대상 파일:

    data/external/processed/fishing_ocean_features.parquet

이 파일은 예측 결과가 아니라 날짜·위치별 해양환경 **입력 Feature**입니다.

## 실행

    git fetch origin
    git switch feature/v0.3.3-positive-set-validation
    pip install -r requirements.txt
    streamlit run app.py

기존 일단위 외부 데이터가 없다면:

    python scripts/build_ocean_features.py

2026 set-level Ocean Feature를 만들려면:

    python scripts/build_setlevel_ocean_features.py --input "data/26년 운항.xlsx"

## 권장 검증 순서

1. 품질/좌표에서 원본 Coordinate Audit 확인
2. Set-level 2026에서 어획 이벤트 / 보고 투망 gap 확인
3. Set-level 양수 어획량 모델 실행
4. 필요 시 set-level Copernicus Feature Store 생성
5. Ocean 모델 포함 후 MAE / Rank / Top10 Lift 비교
6. 기존 일단위 Ocean Ablation / Walk-forward 결과와 분리해서 해석
7. 향후 실패 투망 GPS와 투망시각 확보 시 성공/실패 분류를 새로 설계

## 주의

- V3.3의 26어획 좌표 이벤트는 성공/어획 기록 중심이며 실패 투망 위치를 포함하지 않습니다.
- 실패 투망 위치가 없으므로 V3.3 set-level 모델은 성공/실패 분류를 하지 않습니다.
- 원본 DMM 좌표는 분 단위로 표현되지만 실제 set별 GPS 정확도를 보장하지 않습니다.
- 한 행에 여러 set이 기록되면 좌표 하나로 각 set의 정확한 위치를 구분할 수 없습니다.
- 외부 해양자료는 현재 일자료 기반입니다.
- 정확한 투망시각이 없으므로 순간 해양상태를 표현하지 못합니다.
- 과거 데이터는 선장이 선택해 간 위치만 포함하므로 선택편향이 존재합니다.
- 단일 연도 성능 상승만으로 특정 해양변수의 효과를 확정하지 않습니다.
- 회사 조업 원본과 생성 Parquet은 GitHub에 커밋하지 않습니다.

## 문서

- `docs/V0.2_MODEL_VALIDATION.md`
- `docs/V0.3_OCEAN_DATA.md`
- `docs/V0.3.1_OCEAN_ABLATION.md`
- `docs/V0.3.2_COORDINATE_AUDIT.md`
- `docs/V0.3.3_POSITIVE_SET_VALIDATION.md`
