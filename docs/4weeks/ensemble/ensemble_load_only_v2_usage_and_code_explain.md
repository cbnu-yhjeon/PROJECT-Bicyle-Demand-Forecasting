# `ensemble_load_only_v2.py` 사용법 & 코드 설명

이 문서는 `src/ensemble/ensemble_load_only_v2.py` 스크립트의 **사용 방법**과  
코드가 내부에서 수행하는 **동작 흐름(로드 → 피처 정렬 → 앙상블 예측 → 저장/평가)** 을 정리합니다.

---

## 1. 목적

이미 학습해둔 모델을 다시 학습하지 않고, 아래 폴더에 저장된 **기존 모델 파일을 로드하여**:

- 회귀(Regression) 예측 수행
- 3개 모델(LGBM/XGB/RF) 예측을 **가중 평균(Weighted Averaging)** 으로 앙상블
- (선택) 예측 결과 CSV 저장
- (선택) 간단한 시계열 분할 기준 성능 평가(RMSE/MAE/R²)

을 수행합니다.

---

## 2. 전제: 프로젝트 폴더 구조

스크립트는 아래 구조를 기본으로 가정합니다.

```text
PROJECT-Bicyle-Demand-Forecasting/
├── Data/
│   └── interim/
│       ├── seoul/seoul_rental_data.csv
│       └── washington/dc_rental_data.csv
└── models/
    ├── IQR/
    │   ├── Seoul_LGBM_all.pkl
    │   ├── Seoul_XGB_all.pkl
    │   ├── Seoul_RF_all.pkl
    │   ├── WashingtonDC_LGBM_all.pkl
    │   ├── WashingtonDC_XGB_all.pkl
    │   └── WashingtonDC_RF_all.pkl
    └── no_IQR/
        ├── Seoul_LGBM_all_no_iqr.pkl
        ├── Seoul_XGB_all_no_iqr.pkl
        ├── Seoul_RF_all_no_iqr.pkl
        ├── WashingtonDC_LGBM_all_no_iqr.pkl
        ├── WashingtonDC_XGB_all_no_iqr.pkl
        └── WashingtonDC_RF_all_no_iqr.pkl
```

---

## 3. 앙상블 방식 설명 (정확한 정의)

이 스크립트의 앙상블은 다음 분류에 해당합니다.

- **Model-based**: 동일한 전체 피처셋을 입력으로 사용
- **Heterogeneous**: 서로 다른 알고리즘(LightGBM / XGBoost / RandomForest)
- **Parallel**: 모델들을 서로 독립적으로 학습/추론
- **Weighted Averaging (Regression)**: 예측값을 가중 평균으로 결합

즉,

```math
\hat{y} = w_{lgbm}\hat{y}_{lgbm} + w_{xgb}\hat{y}_{xgb} + w_{rf}\hat{y}_{rf},
\quad \sum w = 1
```

> ⚠️ 분류에서 말하는 hard/soft voting 개념이 아니라,  
> 회귀 문제에서의 **averaging ensemble** 입니다.  
> 또한 meta-model을 학습하지 않으므로 **stacking** 이 아닙니다.

---

## 4. 실행 방법

### 4.1 기본 실행 (Seoul / IQR)

```bash
python src/ensemble/ensemble_load_only_v2.py --city seoul --variant IQR
```

### 4.2 기본 실행 (WDC / no_IQR)

```bash
python src/ensemble/ensemble_load_only_v2.py --city wdc --variant no_IQR
```

### 4.3 예측 결과 CSV 저장

```bash
python src/ensemble/ensemble_load_only_v2.py --city seoul --variant IQR \
  --save_preds Data/interim/seoul/seoul_ensemble_preds.csv
```

저장되는 CSV 컬럼:
- `date` (기본 `--date_col`)
- `y_pred_ensemble`
- `y_true` (target 컬럼이 존재할 경우에만)

### 4.4 앙상블 가중치 변경

`--weights`는 순서대로 **lgbm xgb rf** 입니다.

```bash
python src/ensemble/ensemble_load_only_v2.py --city wdc --variant IQR \
  --weights 0.50 0.30 0.20
```

> 내부에서 자동으로 `sum(weights)=1`이 되도록 정규화합니다.

### 4.5 사용자 CSV 지정

```bash
python src/ensemble/ensemble_load_only_v2.py --city seoul --variant no_IQR \
  --csv Data/interim/seoul/seoul_rental_data.csv
```

---

## 5. 인자(Arguments) 정리

| 인자 | 기본값 | 설명 |
|---|---:|---|
| `--city` | (필수) | `seoul` 또는 `wdc` |
| `--variant` | `no_IQR` | `IQR` 또는 `no_IQR` |
| `--csv` | 자동 선택 | 도시별 기본 CSV 경로 사용 |
| `--date_col` | `date` | 날짜/시간 컬럼 |
| `--target_col` | `rental_count` | 타깃 컬럼(있으면 평가 수행) |
| `--weights` | `0.4 0.35 0.25` | `lgbm xgb rf` 가중치 |
| `--save_preds` | 없음 | 예측 결과 CSV 저장 경로 |

---

## 6. 코드 동작 흐름 (중요)

### Step 1) 로그 설정
- `logs/ensemble_load_only_v2_<city>_<variant>_YYYYMMDD_HHMMSS.log` 저장

### Step 2) 입력 CSV 로드 및 정렬
- `date_col` 기준으로 정렬
- WDC의 경우 `"quarter of day"` → `"quarter_flag"` rename 지원

### Step 3) 모델 경로 자동 해석
`models/<variant>/` 하위의 파일명을 규칙 기반으로 매핑합니다.

- IQR: `*_LGBM_all.pkl`, `*_XGB_all.pkl`, `*_RF_all.pkl`
- no_IQR: `*_LGBM_all_no_iqr.pkl`, `*_XGB_all_no_iqr.pkl`, `*_RF_all_no_iqr.pkl`

### Step 4) 모델 로드
로드 우선순위:
1. `src/utils/model_utils/model_io.py`의 `load_model` (가능하면)
2. `joblib.load`
3. `pickle.load`

### Step 5) **피처 불일치(35 vs 36) 문제 해결**
이 스크립트의 핵심 개선점입니다.

- 모델에서 가능한 경우 **학습 당시 feature list를 추출**
  - `feature_names_in_` (sklearn)
  - `feature_name_` (LightGBM)
  - `booster.feature_names` (XGBoost)
- 추론용 CSV에 없는 컬럼은 **새로 만들고 0.0으로 채움**
- **컬럼 순서를 학습 당시 순서로 강제**

그래서 LightGBM의 아래 오류가 방지됩니다.

```text
The number of features in data (...) is not the same as it was in training data (...)
```

### Step 6) 모델별 예측 → 가중 평균 결합
- LGBM / XGB / RF 각각 `predict`
- `weights`를 곱해 합산 → `y_pred_ensemble`

### Step 7) (선택) 평가 지표 계산
`target_col`이 CSV에 존재하면,
시계열 분할(Train/Val/Test)로 RMSE/MAE/R²를 로그에 출력합니다.

### Step 8) (선택) 예측 결과 저장
`--save_preds`를 지정하면 CSV로 저장합니다.

---

## 7. 자주 발생하는 문제 & 해결

### 7.1 LightGBM feature mismatch
- 원인: 학습 시 사용한 컬럼이 현재 CSV에 누락
- 해결: v2에서 자동으로 누락 컬럼을 0.0으로 생성하여 해결

### 7.2 모듈 미설치로 pkl 로드 실패
예: `ModuleNotFoundError: No module named 'lightgbm'`

- 해결: 모델을 저장한 환경과 동일하게 `lightgbm`, `xgboost` 설치 필요

---

## 8. 추천 운영 방식 (안전성)

추후 동일 문제를 원천 차단하려면:
- 학습 시점에 **feature_cols 리스트를 json으로 같이 저장**
- 추론 시에는 그 json을 1순위로 사용

현재 v2는 “모델 artifact에서 feature를 최대한 복구”하는 방식으로 안전성을 확보합니다.

---

## 9. 예시 커맨드 모음

```bash
# 1) WDC no_IQR: 예측 + 평가 + 저장
python src/ensemble/ensemble_load_only_v2.py --city wdc --variant no_IQR \
  --save_preds Data/interim/washington/wdc_ensemble_preds.csv

# 2) Seoul IQR: 가중치 조정
python src/ensemble/ensemble_load_only_v2.py --city seoul --variant IQR \
  --weights 0.45 0.35 0.20

# 3) Seoul no_IQR: 사용자 CSV 지정
python src/ensemble/ensemble_load_only_v2.py --city seoul --variant no_IQR \
  --csv Data/interim/seoul/seoul_rental_data.csv
```
