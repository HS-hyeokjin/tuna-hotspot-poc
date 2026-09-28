# tuna-hotspot-poc

참치 선망 조업데이터와 Copernicus Marine 해양환경 데이터를 이용해 **어장 탐색에 실제로 활용 가능한 신호가 있는지 검증**하는 Python PoC입니다.

> 특정 좌표의 미래 어획을 확정하는 시스템이 아닙니다.
> 시간순 검증과 동일표본 비교로 어떤 데이터가 실제 예측력을 추가하는지 단계적으로 확인합니다.

## 현재 브랜치

`feature/v0.3.1-ocean-ablation`

- `main`: V1 baseline
- `feature/v0.2-model-validation`: 선박/선장/조업방법 효과 분리
- `feature/v0.3-ocean-data`: Copernicus Marine 외부 해양데이터 결합
- `feature/v0.3.1-ocean-ablation`: 해양변수별 Ablation + Parquet Viewer

버전별 브랜치는 병합하지 않고 유지합니다.

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
    git switch feature/v0.3.1-ocean-ablation
    pip install -r requirements.txt
    streamlit run app.py

외부 데이터가 없다면 먼저:

    python scripts/build_ocean_features.py

## 권장 검증 순서

1. Ocean Data에서 Coverage와 오류 확인
2. AI 실험실에서 V2 / V3 전체 비교
3. Ocean Ablation에서 2026 동일표본 비교
4. Ocean Ablation에서 Walk-forward Ablation 실행
5. 여러 연도에서 반복적으로 개선되는 변수 확인
6. 투망시각 확보 가능 여부 확인
7. 이후 시간단위 해양자료 또는 +24h/+48h forecast 검증

## 주의

- 외부 해양자료는 현재 일자료 기반입니다.
- 정확한 투망시각이 없으므로 순간 해양상태를 표현하지 못합니다.
- 과거 데이터는 선장이 선택해 간 위치만 포함하므로 선택편향이 존재합니다.
- 단일 연도 성능 상승만으로 특정 해양변수의 효과를 확정하지 않습니다.
- 회사 조업 원본과 생성 Parquet은 GitHub에 커밋하지 않습니다.

## 문서

- `docs/V0.2_MODEL_VALIDATION.md`
- `docs/V0.3_OCEAN_DATA.md`
- `docs/V0.3.1_OCEAN_ABLATION.md`
