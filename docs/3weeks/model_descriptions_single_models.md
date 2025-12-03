
# 자전거 수요 예측 프로젝트 – 단일 모델 3종 설명서

이 문서는 현재 프로젝트에서 사용 중인 **단일 모델 3종**에 대해,
- 어떤 알고리즘인지
- 우리 코드에서 어떻게 설정/학습되는지
- 장단점과 활용 포인트

를 정리한 마크다운 문서입니다.

사용 중인 스크립트:
- `src/Models/train_single_models.py` (IQR 버전)
- `src/Models/train_single_models_no_iqr.py` (No-IQR 버전)

공통적으로 다음 세 가지 회귀 모델을 사용합니다.

- **RandomForestRegressor (RF)**
- **XGBRegressor (XGBoost)**
- **LGBMRegressor (LightGBM)**

---

## 1. 공통 설정 및 데이터 구조

### 1.1 입력 피처 구조

모든 모델은 공통 타깃 `rental_count`(시간대별 대여량)를 예측하며,  
다음 세 그룹의 피처를 조합해서 사용합니다 (`feature_mode = "all"` 기준):

1. **시간 기반(Time features)**
   - `month`, `weekend`, `quarter_flag`
   - 목적: 계절성, 평일/주말, 시간대(쿼터) 패턴 반영

2. **POI / 공간 기반(POI features)**
   - 가장 가까운 거리: `n_station_dis(m)`, `n_bus_dis(m)`, `n_school_dis(m)`, `n_park_dis(m)` 등
   - 반경 내 개수: `N_of_stations_within_100m`, `N_of_bus_within_500m` 등
   - 목적: 주변 인프라 밀도와 위치 특성이 수요에 미치는 영향 반영

3. **날씨 기반(Weather features)**
   - 서울: `temperature`, `Precipitation`, `windspeed`, `humidity`, `sunshine`, `cloudcover`, `weathersit` …
   - DC: `temperature`, `humidity`, `windspeed`, `cloud_cover`, `shortwave_radiation`, `precipitation`, `rain`, `snowfall`, `snow_depth`, `weathersit`
   - 목적: 날씨/기상 상태에 따른 수요 변동 반영

EDA + 모델링 단계에서 **문자열 컬럼은 제거하고, 숫자형 피처만 사용**하도록 구현되어 있습니다.

---

### 1.2 데이터 분할 & 튜닝 공통 구조

세 모델 모두 동일한 흐름으로 학습됩니다.

1. **시계열 정렬**
   ```python
   df = df.sort_values("date").reset_index(drop=True)
   ```

2. **시계열 기반 Split (예: 70 / 15 / 15)**
   - Train: 과거 70%
   - Validation: 중간 15%
   - Test: 가장 최근 15%

3. **튜닝용 샘플링 (최대 200,000행)**
   - 전체 Train 데이터를 그대로 쓰면 너무 크기 때문에
   - `make_tuning_sample()` 함수로 최대 20만 행까지 랜덤 샘플링 후 RandomizedSearchCV 적용

4. **RandomizedSearchCV로 하이퍼파라미터 튜닝**
   - 공통 설정
     ```python
     search = RandomizedSearchCV(
         model,
         param_distributions=param_dist,
         n_iter=8,          # 시도할 조합 수
         cv=3,              # 3-fold cross validation
         scoring="neg_root_mean_squared_error",
         n_jobs=-1,         # 가용 CPU 모두 사용
         verbose=1,
     )
     search.fit(X_tune, y_tune)
     best_params = search.best_params_
     ```

5. **Train+Val 전체로 재학습 → Test 평가 → .pkl 저장**
   - 찾은 `best_params`로 동일 모델 인스턴스를 새로 만들고,
   - `X_train_full = Train + Val` 전체로 다시 학습
   - Test 구간에서 RMSE, MAE, R² 계산
   - 최종 학습 모델을 `.pkl`로 저장
     ```text
     Seoul_RF_all_no_iqr.pkl
     Seoul_XGB_all_no_iqr.pkl
     Seoul_LGBM_all_no_iqr.pkl
     WashingtonDC_RF_all_no_iqr.pkl
     ...
     ```

---

## 2. RandomForestRegressor (랜덤포레스트)

### 2.1 알고리즘 개념

- **배깅(Bagging)** 기반의 앙상블 결정트리 모델
- 여러 개의 **Decision Tree**를 서로 다른 데이터 샘플/피처 서브셋으로 학습한 뒤,
  - 회귀 문제에서는 각 트리 예측값의 **평균**을 최종 예측으로 사용
- 개별 트리는 다소 과적합될 수 있지만,
  **여러 개를 평균내면서 분산을 줄이고 성능을 안정화**시키는 구조

장점:

- 비선형/복잡한 관계를 잘 모델링
- 스케일링 필요 없음 (표준화/정규화 필수 아님)
- 피처 중요도(feature importance) 해석 가능

단점:

- 트리 수가 많을수록 메모리와 예측 시간이 증가
- XGBoost/LightGBM에 비해 **정교한 경계**를 만드는 능력은 다소 떨어질 수 있음

---

### 2.2 코드에서의 설정/튜닝 방식

1. **기본 모델 정의**
   ```python
   rf = RandomForestRegressor(
       random_state=42,
       n_jobs=-1,  # CPU 병렬 처리
   )
   ```

2. **튜닝 파라미터 후보 예시**
   ```python
   rf_param = {
       "n_estimators": [100, 200],
       "max_depth": [8, 12],
       "min_samples_split": [5, 10],
       "min_samples_leaf": [2, 4],
       "max_features": ["sqrt", 0.5],
   }
   ```

3. **튜닝 & 재학습 플로우**
   - Train의 최대 200,000행에 대해 RandomizedSearchCV 실행
   - Best params 찾기 → Train+Val 전체로 다시 학습
   - Test에서 RMSE/MAE/R² 계산
   - 파일명 예:
     - `Seoul_RF_all_no_iqr.pkl`
     - `WashingtonDC_RF_all_no_iqr.pkl`

---

### 2.3 우리 프로젝트에서 RF의 포지션

- 매우 큰 데이터(서울 약 1,400만 행)에서는 런타임과 메모리 이슈가 있을 수 있지만,
  **샘플링 + 적절한 파라미터 제약**으로 안정적으로 학습 가능
- 트리 기반 특성상, 시간/POI/날씨의 **비선형 상호작용**도 어느 정도 포착 가능
- LightGBM, XGBoost와의 성능 비교에서 **베이스라인/참고 모델** 역할

---

## 3. XGBRegressor (XGBoost)

### 3.1 알고리즘 개념

- **Gradient Boosting Decision Tree(GBDT)** 계열의 대표적인 구현체
- 랜덤포레스트가 “여러 트리를 병렬로 학습하여 평균”하는 반면,  
  XGBoost는 **트리를 순차적으로 추가**하면서
  - 이전 모델의 오차(residual)를 줄이도록 다음 트리를 학습
- 손실 함수의 **기울기(gradient)**를 이용해 최적의 분할/노드를 찾는 방식

장점:

- 강력한 예측 성능 (특히 테이블형 데이터)
- 다양한 정규화(regularization) 옵션으로 과적합 제어에 강함
- missing 값 처리/가중치 등이 잘 정리된 라이브러리

단점:

- 하이퍼파라미터가 많아 튜닝 난이도가 다소 높음
- 매우 큰 데이터에서는 학습 시간/메모리 부담이 될 수 있음 (hist 모드로 완화)

---

### 3.2 코드에서의 설정/튜닝 방식

1. **기본 모델 정의**
   ```python
   xgb = XGBRegressor(
       objective="reg:squarederror",
       random_state=42,
       tree_method="hist",  # 대용량 데이터에 유리한 히스토그램 기반 분할
       n_jobs=-1,
   )
   ```

2. **튜닝 파라미터 후보 예시**
   ```python
   xgb_param = {
       "n_estimators": [300, 500],
       "max_depth": [3, 4, 5],
       "learning_rate": [0.03, 0.05, 0.1],
       "subsample": [0.7, 0.9],
       "colsample_bytree": [0.7, 0.9],
       "reg_lambda": [1.0, 3.0, 5.0],
   }
   ```

3. **튜닝 & 재학습 플로우**
   - RF와 동일하게 RandomizedSearchCV + 최대 20만 행 샘플링
   - 최적 하이퍼파라미터로 Train+Val 전체 재학습
   - Test 평가 및 `.pkl` 저장
     - 예: `Seoul_XGB_all_no_iqr.pkl`, `WashingtonDC_XGB_all_no_iqr.pkl`

---

### 3.3 우리 프로젝트에서 XGBoost의 포지션

- 시간·POI·날씨가 복잡하게 섞인 테이블 데이터에서 **강력한 기준 모델**
- RF보다 **일반적으로 높은 예측 성능**을 기대할 수 있으며,
  특히 **비선형/상호작용**이 강한 경우 유리
- 학습 시간/튜닝 복잡도는 RF 대비 크지만,
  LightGBM과 함께 “메인 부스트 계열”로 사용

---

## 4. LGBMRegressor (LightGBM)

### 4.1 알고리즘 개념

- 마이크로소프트에서 개발한 **Gradient Boosting 기반 GBDT 라이브러리**
- **Leaf-wise(리프 단위 성장)** 전략:
  - 트리를 레벨별로 넓게 성장시키는 대신,
  - **손실 감소가 가장 큰 리프 노드부터 계속 분할**해서 깊게 성장
- 효율적인 히스토그램 기반 학습과 다양한 최적화로
  - 매우 빠른 학습 속도
  - 대규모/고차원 데이터에서 우수한 성능

장점:

- XGBoost 대비 더 빠른 경우가 많음
- 고차원·대용량 데이터에 특화
- 카테고리 처리 기능(이번 프로젝트에서는 주로 숫자형 위주 사용)

단점:

- Leaf-wise 성장 특성상, 부적절한 파라미터 설정 시 과적합 위험
- 일부 설정이 직관적이지 않을 수 있음

---

### 4.2 코드에서의 설정/튜닝 방식

1. **기본 모델 정의**
   ```python
   lgbm = LGBMRegressor(
       objective="regression",
       random_state=42,
       n_estimators=500,
       n_jobs=-1,
   )
   ```

2. **튜닝 파라미터 후보 예시**
   ```python
   lgbm_param = {
       "num_leaves": [31, 63, 127],
       "max_depth": [-1, 8, 12],
       "learning_rate": [0.03, 0.05, 0.1],
       "subsample": [0.7, 0.9],
       "colsample_bytree": [0.7, 0.9],
       "reg_lambda": [0.0, 1.0, 3.0],
   }
   ```

3. **튜닝 & 재학습 플로우**
   - XGBoost와 동일하게 RandomizedSearchCV 적용
   - Train+Val 전체로 재학습 후 Test 평가
   - 최종 모델 저장:
     - `Seoul_LGBM_all_no_iqr.pkl`
     - `WashingtonDC_LGBM_all_no_iqr.pkl`
     - (IQR 버전은 `_no_iqr` 없이 저장)

---

### 4.3 우리 프로젝트에서 LightGBM의 포지션

- 실제 학습 결과에서 **대부분의 설정에서 최고 성능**을 보인 모델
- 시간 + POI + 날씨의 복잡한 조합을 잘 학습하면서도
  - 학습 속도가 빠르고
  - 메모리 효율이 좋아 **실무 배포까지 고려하면 가장 유력한 1등 모델**

따라서:

- 단일 모델 비교에서 **기준 SOTA 후보**
- 향후 **앙상블(예: RF + XGB + LGBM 가중 평균)** 구성 시
  - LightGBM을 핵심 축으로 두고,
  - RF, XGB를 보조 관점/불확실성 보정 용도로 활용 가능

---

## 5. 세 모델 비교 요약

| 항목 | RandomForest | XGBoost | LightGBM |
|------|--------------|---------|----------|
| 알고리즘 | Bagging Tree Ensemble | Gradient Boosting Tree | Gradient Boosting Tree (Leaf-wise) |
| 학습 방식 | 병렬 트리 평균 | 순차적 트리 추가 | 순차적 트리 + 히스토그램/Leaf-wise |
| 속도 | 중간 | 중간~느림 | 빠름 |
| 비선형/상호작용 | 잘 처리 | 매우 잘 처리 | 매우 잘 처리 |
| 튜닝 난이도 | 낮음~중간 | 중간~높음 | 중간 |
| 프로젝트 내 역할 | 베이스라인, 안정성 확인 | 강력한 부스트 모델 | 주력 1순위 모델 |

---

## 6. 결론 및 활용 방향

- 세 모델 모두 **시간·공간·날씨 기반 자전거 수요 예측**에 적합한 트리 기반 앙상블/부스트 계열 모델입니다.
- 현재 코드는
  - 동일한 데이터 분할/튜닝/평가 파이프라인 위에서
  - RF, XGB, LGBM을 공정하게 비교할 수 있도록 설계되어 있습니다.
- 이후 단계에서:
  - **앙상블(가중 평균, 스태킹 등)** 설계,
  - **SHAP 기반 피처 중요도 분석**
  - 도시별(서울 vs DC) 모델 차이 분석
  등을 진행하면, 논문/발표용 인사이트를 더 풍부하게 만들 수 있습니다.
