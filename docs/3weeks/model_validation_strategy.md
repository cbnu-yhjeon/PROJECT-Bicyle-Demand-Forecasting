
# 모델 검증(Validation & Test) 방식 설명

이 문서는 자전거 수요 예측 프로젝트에서 사용하는 **모델 검증 방식(Time-series Validation)**을 정리한 문서입니다.

---

# 1. 시계열(Time-series) 기반 데이터 분할 전략
프로젝트는 수요 예측 문제이므로 데이터의 시간 순서를 보존하는 **시계열 기반 Hold-out 방식**을 사용합니다.

## 📌 분할 비율
Train : Validation : Test = **70% : 15% : 15%**

## 📌 처리 흐름
1) date 기준 정렬  
2) 앞 70% → Train  
3) 다음 15% → Validation  
4) 마지막 15% → Test  

---

# 2. 왜 시계열 분할을 사용하는가?

✓ 미래 데이터가 학습에 섞이지 않도록 **데이터 누수 방지**  
✓ 수요 예측은 시간 의존성이 크므로 **랜덤 분할은 부적절**  
✓ 실제 운영 환경에서 모델이 예측해야 하는 미래값을 가장 잘 모사

---

# 3. 평가 지표

- **RMSE** – 큰 오차에 강한 페널티, 수요 예측에서 핵심  
- **MAE** – 직관적인 평균 오차  
- **R²** – 모델 설명력  

---

# 4. 코드에서의 검증 방식 요약

```python
df = df.sort_values("date")
n = len(df)
n_test = int(n * 0.15)
n_val = int(n * 0.15)
n_train = n - n_test - n_val

train = df.iloc[:n_train]
val   = df.iloc[n_train:n_train+n_val]
test  = df.iloc[n_train+n_val:]
```

---

# 5. 결론

이 방식은 **실제 미래 수요 예측 환경에 가장 근접한 검증 방식**이며,  
예측 모델의 일반화 능력을 안정적으로 평가할 수 있도록 설계되었습니다.
