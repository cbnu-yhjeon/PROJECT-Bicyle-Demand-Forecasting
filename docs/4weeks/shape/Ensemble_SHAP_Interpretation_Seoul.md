# 📊 Ensemble SHAP Interpretation Report (Seoul)

## 1. Ensemble 관점 종합 해석

본 앙상블 모델은 LightGBM, XGBoost, RandomForest 세 모델의 예측값을 평균(가중/단순)하여 최종 예측을 산출한다.  
이 구조의 핵심적 특성은 다음과 같다.

- 세 모델이 **공통적으로 강하게 동의하는 패턴**은 앙상블에서 더욱 강화됨
- 특정 단일 모델에서만 두드러지는 패턴은 평균 과정에서 자연스럽게 약화됨
- 결과적으로 앙상블은 개별 모델의 우연적 규칙보다 **일관되고 안정적인 신호**에 의존

따라서 앙상블 예측 논리는 “시간·기상·공간 요인의 반복적 패턴”을 중심으로 형성된다.

---

## 2. 모델별 Top 중요 피처 및 영향 방향 비교

### 2.1 모델별 Top5 중요 피처 (노트북 기준)

- **LightGBM (LGBM)**
  - N_of_stations_within_2000m
  - Holding_quantity
  - n_station_dis(m)
  - temperature
  - quarter_flag

- **RandomForest (RF)**
  - month
  - n_station_dis(m)
  - Holding_quantity
  - temperature
  - quarter_flag

- **XGBoost (XGB)**
  - N_of_stations_within_2000m
  - Holding_quantity
  - n_station_dis(m)
  - temperature
  - quarter_flag  
  *(SHAP 플롯 및 중요도 분포가 LGBM과 매우 유사)*

---

### 2.2 주요 피처별 SHAP 해석

#### A. quarter_flag (시간대)
- 세 모델 모두에서 공통 상위
- 시간대별 SHAP 값이 명확히 분리됨
- 특정 시간대는 수요 증가(양의 SHAP), 특정 시간대는 수요 감소(음의 SHAP)

➡️ 자전거 수요는 **시간대 패턴에 매우 민감**

---

#### B. temperature (기온)
- 공통 상위 피처
- dependence plot에서 비선형 곡선 형태
- 너무 낮거나 너무 높을 경우 수요 감소
- 쾌적 온도 구간에서 수요 증가

➡️ 단순 선형 관계가 아닌 **적정 기온 효과**

---

#### C. n_station_dis(m) (접근성)
- 가까울수록 SHAP 양수, 멀수록 음수
- 접근성이 높을수록 대여 수요 증가

➡️ **공간적 접근성 효과**가 명확

---

#### D. N_of_stations_within_2000m (인프라 밀도)
- LGBM, XGB에서 특히 강함
- 중간 밀도에서 SHAP 최대
- 너무 적거나 너무 많으면 수요 감소

➡️ **적정 인프라 밀도**가 최적 수요 조건

---

#### E. Holding_quantity (자전거 보유량)
- 모든 모델에서 상위권
- 단독 효과보다는 다른 피처와 결합된 보조적 역할
- 공급 여건을 나타내는 컨텍스트 변수

---

#### F. month (계절성)
- RF에서만 Top5에 포함
- RF는 월 단위의 거친 계절 패턴을 직접 활용
- 부스팅 계열은 temperature 등 연속 변수로 계절성을 흡수

---

## 3. 앙상블이 공통적으로 의존하는 요인

### 3.1 앙상블 공통 코어 요인 (교집합)

- **시간대 요인**: quarter_flag
- **기상 요인**: temperature
- **접근성 요인**: n_station_dis(m)
- **공급 여건 요인**: Holding_quantity

### 3.2 부스팅 모델에서 강화되는 요인
- N_of_stations_within_2000m (공간 인프라 밀도)

### 3.3 RF에서 상대적으로 두드러지는 요인
- month (거친 계절성)

---

## 4. 한 문장 결론 (보고서용)

Seoul 앙상블 모델은 시간대(quarter_flag)와 기온(temperature)을 중심으로, 접근성(n_station_dis)과 주변 인프라 밀도(N_of_stations_within_2000m)를 결합해 수요를 예측하며, 세 모델이 공통적으로 동의하는 핵심 신호는 강화되고 단일 모델 특화 신호는 평균화되어 완화된다.
