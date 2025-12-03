# 🚲 자전거 수요 예측 모델 학습 결과 요약

## 📌 1. 전체 프로세스 개요
- Seoul + WashingtonDC 2개 도시 데이터 학습
- IQR 기반 이상치 제거 → Feature Selection → Train/Val/Test Split → 모델 학습
- 전체 수행 시간: 약 358초

---

## 📊 2. 데이터 처리 결과

### **Seoul**
- 원본: 14,581,612행  
- IQR 후: 3,930,710행  
→ 약 **73% 제거**

### **Washington DC**
- 원본: 1,402,340행  
- IQR 후: 225,484행  
→ 약 **84% 제거**

---

## 🤖 3. 모델 학습 결과 요약

---

# 🇰🇷 Seoul (Feature Mode: ALL)

### Train/Val/Test Size
- Train: 2,751,498  
- Val: 589,606  
- Test: 589,606  

### ⭐ Best Model = **LightGBM**

| 모델 | Train RMSE | Val RMSE | Test RMSE | Test R² |
|------|------------|-----------|------------|-----------|
| RandomForest | 9.381 | 11.525 | 8.124 | 0.564 |
| XGBoost | 9.159 | 11.152 | 7.821 | 0.596 |
| **LightGBM** | **6.555** | **7.499** | **6.007** | **0.762** |

### 학습 시간
- RF: 90.7초  
- XGB: 9.1초  
- LGBM: 12.0초  

---

# 🇺🇸 Washington DC (Feature Mode: ALL)

### Train/Val/Test Size
- Train: 157,840  
- Val: 33,822  
- Test: 33,822  

### ⭐ Best Model = **LightGBM**

| 모델 | Train RMSE | Val RMSE | Test RMSE | Test R² |
|------|------------|-----------|------------|-----------|
| RandomForest | 4.131 | 5.949 | 6.626 | 0.524 |
| XGBoost | 4.107 | 5.356 | 6.828 | 0.495 |
| **LightGBM** | **3.943** | **4.921** | **6.796** | **0.499** |

### 학습 시간
- RF: 1.1초  
- XGB: 2.7초  
- LGBM: 0.9초  

---

# 🏆 4. 최종 비교

| 도시 | Best Model | Test RMSE | Test R² |
|-------|-------------|------------|-----------|
| Seoul | **LightGBM** | **6.007** | **0.762** |
| WashingtonDC | **LightGBM** | **6.796** | **0.499** |

---

# 📌 5. 결론

- **Seoul 데이터의 품질과 패턴이 더 명확하여 정확도 우수**  
- **Washington DC 데이터는 IQR로 84%가 제거되어 성능 다소 저하**
- LightGBM이 **모든 도시에서 일관적으로 최적 성능**
- 향후 IQR 완화/제거 버전 비교 실험 필요

---

# 📁 마무리
본 문서는 모델링 결과를 요약한 리포트로, 추후 SHAP 분석·앙상블·시각화 단계의 기반 자료가 됩니다.
