# src/Models/train_single_models.py

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
SRC_ROOT = CURRENT_FILE.parents[1]  # .../src
PROJECT_ROOT = CURRENT_FILE.parents[2]

if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# 유틸 import
from utils.model_utils.outlier import remove_outliers_iqr
from utils.model_utils.model_io import save_model

# 전역 로거
LOGGER = logging.getLogger("bike_demand")


def setup_logger(project_root: Path) -> None:
    """
    파일 + 콘솔로 동시에 로그를 남기는 기본 설정.
    """
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_single_models_{timestamp}.log"

    LOGGER.setLevel(logging.INFO)

    # 기존 핸들러 제거(중복 방지)
    if LOGGER.hasHandlers():
        LOGGER.handlers.clear()

    # 포맷
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 파일 핸들러
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # 콘솔 핸들러
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

    # "quarter of day" -> "quarter_flag" 이름 통일
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
# 3. 시계열 기반 train/val/test 분할
# -----------------------------
def time_series_split(df: pd.DataFrame,
                      date_col: str = "date",
                      target_col: str = "rental_count",
                      val_size: float = 0.15,
                      test_size: float = 0.15):
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
# 4. 튜닝용 샘플링 + 튜닝 함수
# -----------------------------
def make_tuning_sample(X, y, max_rows=200_000, random_state=42):
    n = len(X)
    if n <= max_rows:
        LOGGER.info(f"[TUNE] 전체 {n}행 사용 (샘플링 생략)")
        return X, y

    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)
    LOGGER.info(f"[TUNE] {n}행 중 {max_rows}행 샘플링 사용")
    return X.iloc[idx], y.iloc[idx]


def tune_model(model, param_distributions, X, y,
               n_iter=10, cv=3, random_state=42, max_rows_for_tuning=None):
    if max_rows_for_tuning is not None:
        X_tune, y_tune = make_tuning_sample(X, y, max_rows_for_tuning, random_state)
    else:
        X_tune, y_tune = X, y

    LOGGER.info(f"[DEBUG] RandomizedSearchCV 시작 (n_iter={n_iter}, cv={cv})")
    search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,   # 이 출력은 sklearn이 stdout으로 직접 찍음
        random_state=random_state,
    )
    search.fit(X_tune, y_tune)
    LOGGER.info(f"  ▶ Best params: {search.best_params_}")
    LOGGER.info(f"  ▶ Best CV RMSE: {-search.best_score_:.4f}")
    return search.best_params_


# -----------------------------
# 5. 평가 함수
# -----------------------------
def evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, label=""):
    def _metrics(y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, r2

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    tr_rmse, tr_mae, tr_r2 = _metrics(y_train, y_pred_train)
    v_rmse, v_mae, v_r2 = _metrics(y_val, y_pred_val)
    te_rmse, te_mae, te_r2 = _metrics(y_test, y_pred_test)

    LOGGER.info(f"\n[{label}] Performance")
    LOGGER.info(f"  Train: RMSE={tr_rmse:.3f}, MAE={tr_mae:.3f}, R2={tr_r2:.3f}")
    LOGGER.info(f"  Valid: RMSE={v_rmse:.3f}, MAE={v_mae:.3f}, R2={v_r2:.3f}")
    LOGGER.info(f"  Test : RMSE={te_rmse:.3f}, MAE={te_mae:.3f}, R2={te_r2:.3f}")


# -----------------------------
# 6. 메인 파이프라인 (도시별 실행)
# -----------------------------
def run_for_city(city_name: str,
                 df: pd.DataFrame,
                 get_feature_groups_fn,
                 feature_mode: str,
                 project_root: Path):
    LOGGER.info("\n==============================")
    LOGGER.info(f"  City: {city_name}, feature_mode={feature_mode}")
    LOGGER.info("==============================")

    city_start = time.time()

    time_feats, poi_feats, weather_feats = get_feature_groups_fn(df)
    LOGGER.info(f"[DEBUG] {city_name} feature groups 준비 완료")

    feature_cols = select_features(df, time_feats, poi_feats, weather_feats,
                                   feature_mode=feature_mode)
    LOGGER.info(f"[DEBUG] {city_name} 사용 피처 수: {len(feature_cols)}")
    LOGGER.info(f"[DEBUG] {city_name} 사용 피처 예시: {feature_cols[:10]}")

    target_col = "rental_count"
    cols_to_keep = feature_cols + [target_col, "date"]
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    df_model = df[cols_to_keep].copy()

    LOGGER.info(f"[DEBUG] {city_name} df_model.shape = {df_model.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = time_series_split(
        df_model, date_col="date", target_col=target_col
    )
    LOGGER.info(f"[DEBUG] {city_name} split:")
    LOGGER.info(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    LOGGER.info(f"[DEBUG] {city_name} X_train_full: {X_train_full.shape}")

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    max_rows_for_tuning = 200_000

    # ------------- RandomForest -------------
    LOGGER.info("\n[RandomForest] 하이퍼파라미터 튜닝 중...")
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_param = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
        "max_features": ["sqrt"],
    }
    rf_best_params = tune_model(
        rf, rf_param, X_train, y_train,
        n_iter=5,
        max_rows_for_tuning=max_rows_for_tuning,
    )

    rf_final = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **rf_best_params,
    )
    rf_start = time.time()
    rf_final.fit(X_train_full, y_train_full)
    rf_end = time.time()
    LOGGER.info(f"[Time] {city_name}-RF-{feature_mode} 최종 학습 시간: {rf_end - rf_start:.1f}초")

    evaluate(rf_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-RF-{feature_mode}")
    save_model(
        rf_final,
        models_dir / f"{city_name}_RF_{feature_mode}.pkl",
    )

    # ------------- XGBoost -------------
    LOGGER.info("\n[XGBoost] 하이퍼파라미터 튜닝 중...")
    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",  # GPU 사용 시 "gpu_hist"
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
    xgb_best_params = tune_model(
        xgb, xgb_param, X_train, y_train,
        n_iter=8,
        max_rows_for_tuning=max_rows_for_tuning,
    )

    xgb_final = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        **xgb_best_params,
    )
    xgb_start = time.time()
    xgb_final.fit(X_train_full, y_train_full)
    xgb_end = time.time()
    LOGGER.info(f"[Time] {city_name}-XGB-{feature_mode} 최종 학습 시간: {xgb_end - xgb_start:.1f}초")

    evaluate(xgb_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-XGB-{feature_mode}")
    save_model(
        xgb_final,
        models_dir / f"{city_name}_XGB_{feature_mode}.pkl",
    )

    # ------------- LightGBM -------------
    LOGGER.info("\n[LightGBM] 하이퍼파라미터 튜닝 중...")
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
    lgbm_best_params = tune_model(
        lgbm, lgbm_param, X_train, y_train,
        n_iter=8,
        max_rows_for_tuning=max_rows_for_tuning,
    )

    lgbm_final = LGBMRegressor(
        objective="regression",
        random_state=42,
        **lgbm_best_params,
    )
    lgbm_start = time.time()
    lgbm_final.fit(X_train_full, y_train_full)
    lgbm_end = time.time()
    LOGGER.info(f"[Time] {city_name}-LGBM-{feature_mode} 최종 학습 시간: {lgbm_end - lgbm_start:.1f}초")

    evaluate(lgbm_final, X_train, y_train, X_val, y_val, X_test, y_test,
             label=f"{city_name}-LGBM-{feature_mode}")
    save_model(
        lgbm_final,
        models_dir / f"{city_name}_LGBM_{feature_mode}.pkl",
    )

    city_end = time.time()
    LOGGER.info(f"\n[Time] {city_name} ({feature_mode}) 전체 학습 시간: "
                f"{city_end - city_start:.1f}초")


def main():
    setup_logger(PROJECT_ROOT)

    seoul_csv = PROJECT_ROOT / "Data" / "interim" / "seoul" / "seoul_rental_data.csv"
    dc_csv = PROJECT_ROOT / "Data" / "interim" / "washington" / "dc_rental_data.csv"

    LOGGER.info(f"Project root: {PROJECT_ROOT}")
    LOGGER.info(f"Seoul CSV   : {seoul_csv}")
    LOGGER.info(f"DC CSV      : {dc_csv}")

    LOGGER.info("[DEBUG] STEP 1: 서울 데이터 로드 시작")
    df_seoul = load_seoul(seoul_csv)
    LOGGER.info(f"[DEBUG] STEP 1 완료: df_seoul.shape = {df_seoul.shape}")

    LOGGER.info("[DEBUG] STEP 2: DC 데이터 로드 시작")
    df_dc = load_dc(dc_csv)
    LOGGER.info(f"[DEBUG] STEP 2 완료: df_dc.shape = {df_dc.shape}")

    LOGGER.info("\n=== [IQR CLEANING] Seoul 데이터 ===")
    numeric_cols_seoul = df_seoul.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "rental_count" in numeric_cols_seoul:
        numeric_cols_seoul.remove("rental_count")
    df_seoul = remove_outliers_iqr(df_seoul, numeric_cols_seoul, multiplier=1.5)
    LOGGER.info(f"[DEBUG] STEP 3 완료: IQR 이후 df_seoul.shape = {df_seoul.shape}")

    LOGGER.info("\n=== [IQR CLEANING] WashingtonDC 데이터 ===")
    numeric_cols_dc = df_dc.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "rental_count" in numeric_cols_dc:
        numeric_cols_dc.remove("rental_count")
    df_dc = remove_outliers_iqr(df_dc, numeric_cols_dc, multiplier=1.5)
    LOGGER.info(f"[DEBUG] STEP 4 완료: IQR 이후 df_dc.shape = {df_dc.shape}")

    feature_mode = "all"

    LOGGER.info("[DEBUG] STEP 5: Seoul 모델 학습 시작")
    run_for_city("Seoul", df_seoul, get_feature_groups_seoul, feature_mode, PROJECT_ROOT)
    LOGGER.info("[DEBUG] STEP 5 완료")

    LOGGER.info("[DEBUG] STEP 6: WashingtonDC 모델 학습 시작")
    run_for_city("WashingtonDC", df_dc, get_feature_groups_dc, feature_mode, PROJECT_ROOT)
    LOGGER.info("[DEBUG] STEP 6 완료")


if __name__ == "__main__":
    script_start = time.time()
    try:
        main()
    except Exception:
        LOGGER.exception("\n[ERROR] 스크립트 실행 중 예외 발생:")
    script_end = time.time()
    LOGGER.info(f"\n[Time] 전체 스크립트 실행 시간: {script_end - script_start:.1f}초")
