# src/Models/train_single_models_no_iqr.py
from pathlib import Path
import sys
import time
import traceback
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# -----------------------------------------
# 0. 경로 / 로거 설정
# -----------------------------------------
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# 모델 저장 유틸
from utils.model_utils.model_io import save_model

LOGGER = logging.getLogger("bike_demand")


def setup_logger(project_root: Path) -> None:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_single_models_no_iqr_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)

    if LOGGER.hasHandlers():
        LOGGER.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

    LOGGER.info(f"📌 로그 파일: {log_path}")


# -----------------------------
# 1. 데이터 로더
# -----------------------------
def load_seoul(csv_path: Path) -> pd.DataFrame:
    LOGGER.info(f"[LOAD] Seoul data: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_dc(csv_path: Path) -> pd.DataFrame:
    LOGGER.info(f"[LOAD] Washington DC data: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    if "quarter of day" in df.columns:
        df = df.rename(columns={"quarter of day": "quarter_flag"})

    df = df.sort_values("date").reset_index(drop=True)
    return df


# -----------------------------
# 2. 피처 그룹 정의
# -----------------------------
def get_feature_groups_seoul(df: pd.DataFrame):
    time_features = ["month", "weekend", "quarter_flag"]

    poi_features = [
        "Holding quantity",
        "n_station_dis(m)", "n_bus_dis(m)", "n_school_dis(m)", "n_park_dis(m)",
        "N_of_stations_within_100m", "N_of_stations_within_500m",
        "N_of_stations_within_1000m", "N_of_stations_within_1500m",
        "N_of_stations_within_2000m",
        "N_of_bus_within_100m", "N_of_bus_within_500m",
        "N_of_bus_within_1000m", "N_of_bus_within_1500m",
        "N_of_bus_within_2000m",
        "N_of_school_within_100m", "N_of_school_within_500m",
        "N_of_school_within_1000m", "N_of_school_within_1500m",
        "N_of_school_within_2000m",
        "N_of_park_within_100m", "N_of_park_within_500m",
        "N_of_park_within_1000m", "N_of_park_within_1500m",
        "N_of_park_within_2000m",
    ]

    weather_features = [
        "temperature", "Precipitation", "windspeed", "humidity",
        "sunshine", "snowcover", "cloudcover", "weathersit",
    ]

    return time_features, poi_features, weather_features


def get_feature_groups_dc(df: pd.DataFrame):
    time_features = ["month", "weekend", "quarter_flag"]

    # 🔴 문자열 피처(n_station, n_bus, n_park, n_school)는 제외
    poi_features = [
        "CAPACITY",
        # station
        "n_station_idx", "n_station_dis(m)",
        "N_of_station_within_100m", "N_of_station_within_500m",
        "N_of_station_within_1000m", "N_of_station_within_1500m",
        "N_of_station_within_2000m",
        # bus
        "n_bus_idx", "n_bus_dis(m)",
        "N_of_bus_within_100m", "N_of_bus_within_500m",
        "N_of_bus_within_1000m", "N_of_bus_within_1500m",
        "N_of_bus_within_2000m",
        # park
        "n_park_idx", "n_park_dis(m)",
        "N_of_park_within_100m", "N_of_park_within_500m",
        "N_of_park_within_1000m", "N_of_park_within_1500m",
        "N_of_park_within_2000m",
        # school
        "n_school_idx", "n_school_dis(m)",
        "N_of_school_within_100m", "N_of_school_within_500m",
        "N_of_school_within_1000m", "N_of_school_within_1500m",
        "N_of_school_within_2000m",
    ]

    weather_features = [
        "temperature", "humidity", "windspeed", "cloud_cover",
        "shortwave_radiation", "precipitation", "rain",
        "snowfall", "snow_depth", "weathersit",
    ]

    return time_features, poi_features, weather_features


    weather_features = [
        "temperature", "humidity", "windspeed", "cloud_cover",
        "shortwave_radiation", "precipitation", "rain",
        "snowfall", "snow_depth", "weathersit",
    ]

    return time_features, poi_features, weather_features


def select_features(df: pd.DataFrame,
                    time_features, poi_features, weather_features,
                    feature_mode: str = "all"):
    if feature_mode == "time":
        cols = time_features
    elif feature_mode == "poi":
        cols = poi_features
    elif feature_mode == "weather":
        cols = weather_features
    elif feature_mode == "all":
        cols = list(dict.fromkeys(time_features + poi_features + weather_features))
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    # 1차: 컬럼이 실제로 존재하는지 필터
    cols = [c for c in cols if c in df.columns]

    # 2차: 숫자 타입만 사용 (object/string 컬럼 제거)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols = [c for c in cols if c in numeric_cols]

    return cols



# -----------------------------
# 3. 시계열 기반 split
# -----------------------------
def time_series_split(df, date_col="date", target_col="rental_count",
                      val_size=0.15, test_size=0.15):

    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)

    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test

    df_train = df_sorted.iloc[:n_train]
    df_val = df_sorted.iloc[n_train:n_train + n_val]
    df_test = df_sorted.iloc[n_train + n_val:]

    X_train = df_train.drop(columns=[target_col, date_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col, date_col])
    y_val = df_val[target_col]

    X_test = df_test.drop(columns=[target_col, date_col])
    y_test = df_test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -----------------------------
# 4. 튜닝 + 학습
# -----------------------------
def make_tuning_sample(X, y, max_rows=200_000, random_state=42):
    n = len(X)
    if n <= max_rows:
        LOGGER.info(f"[TUNE] 전체 {n}행 사용")
        return X, y

    idx = np.random.RandomState(random_state).choice(n, max_rows, replace=False)
    LOGGER.info(f"[TUNE] {n}행 중 {max_rows}행 샘플링 사용")
    return X.iloc[idx], y.iloc[idx]


def tune_model(model, param_dist, X, y,
               n_iter=8, cv=3, max_rows_for_tuning=200_000):

    X_tune, y_tune = make_tuning_sample(X, y, max_rows_for_tuning)

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_tune, y_tune)

    LOGGER.info(f"  ▶ Best params: {search.best_params_}")
    LOGGER.info(f"  ▶ Best CV RMSE: {-search.best_score_:.4f}")

    return search.best_params_


# -----------------------------
# 5. 평가 함수
# -----------------------------
def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label=""):

    def compute(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    tr = compute(y_train, y_pred_train)
    va = compute(y_val, y_pred_val)
    te = compute(y_test, y_pred_test)

    LOGGER.info(f"\n[{label}] 성능 평가")
    LOGGER.info(f"  Train: RMSE={tr[0]:.3f}, MAE={tr[1]:.3f}, R2={tr[2]:.3f}")
    LOGGER.info(f"  Valid: RMSE={va[0]:.3f}, MAE={va[1]:.3f}, R2={va[2]:.3f}")
    LOGGER.info(f"  Test : RMSE={te[0]:.3f}, MAE={te[1]:.3f}, R2={te[2]:.3f}")


# -----------------------------
# 6. 도시별 전체 파이프라인
# -----------------------------
def run_for_city(city_name, df, get_feature_fn, feature_mode, project_root):

    LOGGER.info("\n==============================")
    LOGGER.info(f"  City: {city_name}, feature_mode={feature_mode}")
    LOGGER.info("==============================")

    start_city = time.time()

    time_feats, poi_feats, weather_feats = get_feature_fn(df)
    feature_cols = select_features(df, time_feats, poi_feats, weather_feats, feature_mode)

    LOGGER.info(f"[DEBUG] {city_name} 사용 피처 수 = {len(feature_cols)}")

    target = "rental_count"
    cols = feature_cols + ["date", target]
    cols = [c for c in cols if c in df.columns]

    df_model = df[cols]

    LOGGER.info(f"[DEBUG] {city_name} df_model.shape = {df_model.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(df_model)
    LOGGER.info(f"[DEBUG] split 결과:")
    LOGGER.info(f"  X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    max_tune_rows = 200_000

    # -------------------------------- RF --------------------------------
    LOGGER.info("\n[RandomForest] 튜닝 시작")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"],
    }
    best = tune_model(rf, rf_param, X_train, y_train, max_rows_for_tuning=max_tune_rows)
    rf_final = RandomForestRegressor(random_state=42, n_jobs=-1, **best)

    t0 = time.time()
    rf_final.fit(X_train_full, y_train_full)
    t1 = time.time()
    LOGGER.info(f"[Time] RF 학습 시간: {t1 - t0:.1f}초")

    evaluate(rf_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-RF-{feature_mode}")

    save_model(rf_final, models_dir / f"{city_name}_RF_{feature_mode}_no_iqr.pkl")

    # -------------------------------- XGB --------------------------------
    LOGGER.info("\n[XGBoost] 튜닝 시작")
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
    )
    xgb_param = {
        "n_estimators": [300, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [1.0, 5.0],
        "min_child_weight": [1, 3],
    }
    best = tune_model(xgb, xgb_param, X_train, y_train, max_rows_for_tuning=max_tune_rows)
    xgb_final = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        **best
    )

    t0 = time.time()
    xgb_final.fit(X_train_full, y_train_full)
    t1 = time.time()
    LOGGER.info(f"[Time] XGB 학습 시간: {t1 - t0:.1f}초")

    evaluate(xgb_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-XGB-{feature_mode}")

    save_model(xgb_final, models_dir / f"{city_name}_XGB_{feature_mode}_no_iqr.pkl")

    # -------------------------------- LGBM --------------------------------
    LOGGER.info("\n[LightGBM] 튜닝 시작")
    lgbm = LGBMRegressor(
        objective="regression",
        random_state=42,
    )
    lgbm_param = {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.05, 0.1],
        "num_leaves": [31, 63],
        "max_depth": [-1, 5, 7],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.7, 0.9],
        "reg_lambda": [0.0, 1.0, 5.0],
    }
    best = tune_model(lgbm, lgbm_param, X_train, y_train, max_rows_for_tuning=max_tune_rows)
    lgbm_final = LGBMRegressor(objective="regression",
                               random_state=42, **best)

    t0 = time.time()
    lgbm_final.fit(X_train_full, y_train_full)
    t1 = time.time()
    LOGGER.info(f"[Time] LGBM 학습 시간: {t1 - t0:.1f}초")

    evaluate(lgbm_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-LGBM-{feature_mode}")

    save_model(lgbm_final, models_dir / f"{city_name}_LGBM_{feature_mode}_no_iqr.pkl")

    end_city = time.time()
    LOGGER.info(f"[Time] {city_name} 전체 학습 시간: {end_city - start_city:.1f}초")


def main():
    setup_logger(PROJECT_ROOT)

    seoul_csv = PROJECT_ROOT / "Data/interim/seoul/seoul_rental_data.csv"
    dc_csv = PROJECT_ROOT / "Data/interim/washington/dc_rental_data.csv"

    LOGGER.info(f"Project root: {PROJECT_ROOT}")
    LOGGER.info(f"Seoul CSV: {seoul_csv}")
    LOGGER.info(f"DC CSV: {dc_csv}")

    df_seoul = load_seoul(seoul_csv)
    df_dc = load_dc(dc_csv)

    LOGGER.info(f"Seoul shape = {df_seoul.shape}")
    LOGGER.info(f"DC shape = {df_dc.shape}")

    feature_mode = "all"

    run_for_city("Seoul", df_seoul, get_feature_groups_seoul, feature_mode, PROJECT_ROOT)
    run_for_city("WashingtonDC", df_dc, get_feature_groups_dc, feature_mode, PROJECT_ROOT)


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    except Exception:
        LOGGER.exception("[ERROR] 프로그램 오류 발생")
    t1 = time.time()
    LOGGER.info(f"[Time] 전체 실행 시간: {t1 - t0:.1f}초")
