
# 모델 학습 파이프라인 설명 (IQR & No-IQR 버전)

본 문서는 `train_single_models.py`와 `train_single_models_no_iqr.py` 두 스크립트의 학습 구조를 분석하여 정리한 문서입니다.

---

# 1. 전체 학습 흐름

```
1) 데이터 로드
2) (선택적) IQR 이상치 제거
3) 피처 자동 선택(time/poi/weather)
4) Train/Val/Test 시계열 분할
5) Hyperparameter Tuning(RandomizedSearchCV)
6) 모델 재학습 (RF / XGB / LGBM)
7) 평가 (Train/Val/Test)
8) 모델 저장(.pkl)
```

---

# 2. IQR 버전과 No-IQR 버전의 차이

| 항목 | IQR 버전 | No-IQR 버전 |
|------|----------|--------------|
| 이상치 제거 | 적용됨 | 없음 |
| 데이터 크기 | 감소 | 원본 유지 |
| 파일명 | *_all.pkl | *_all_no_iqr.pkl |
| 특징 | 노이즈 감소 | 자연 패턴 보존 |

---

# 3. 피처 구성 방식

두 스크립트 모두 다음 피처 그룹 기반으로 자동 선택:

### Time Features
- month  
- weekend  
- quarter_flag  

### POI Features
- n_station_dis(m)  
- n_bus_dis(m)  
- 거리 기반 주변 시설 개수 등  

### Weather Features
- temperature  
- humidity  
- windspeed  
- cloudcover  
- shortwave_radiation (DC)  

문자열 컬럼 제거 후 **숫자형 피처만 사용**하여 모델 안정성 확보.

---

# 4. 시계열 기반 데이터 분할 (Train/Val/Test)

두 스크립트 동일한 로직 적용:

```python
df = df.sort_values("date")
train : val : test = 70% : 15% : 15%
```

---

# 5. Hyperparameter Tuning

✓ RandomizedSearchCV 사용  
✓ 샘플링된 200,000행만 사용해 속도 최적화  
✓ 모델별 주요 탐색 범위:

### RandomForest
- n_estimators  
- max_depth  
- min_samples_split  
- min_samples_leaf  

### XGBoost
- max_depth  
- learning_rate  
- subsample  
- colsample_bytree  

### LightGBM
- num_leaves  
- learning_rate  
- max_depth  
- subsample  

튜닝 후 → Train+Val 전체로 **재학습** 수행.

---

# 6. 모델 저장 방식

예시:

```
Seoul_LGBM_all_no_iqr.pkl
Seoul_XGB_all_no_iqr.pkl
WashingtonDC_RF_all_no_iqr.pkl
```

구조:
```
{City}_{ModelShort}_{FeatureMode}_{Version}.pkl
```

---

# 7. 결론

- 두 스크립트는 동일한 ML 파이프라인을 공유  
- 차이는 오직 **IQR 이상치 제거 여부**  
- Hyperparameter Tuning + 시계열 분할을 통해 실제 예측 환경을 잘 모사  
- 파일 저장 규칙을 통해 도시/모델/전처리 버전을 명확히 구분

