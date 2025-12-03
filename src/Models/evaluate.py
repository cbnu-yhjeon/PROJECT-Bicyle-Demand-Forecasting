#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
단일 모델 3종(RandomForest, XGBoost, LightGBM)에 대한
평가 및 시각화 스크립트.

- 별도의 외부 test 셋은 사용하지 않고,
  전체 데이터 중 뒤 20%를 '평가(eval) 세트'로 사용.
- 서울 / 워싱턴 DC 각각에 대해:
    1) 모델 로드
    2) 피처 선택 (train_single_models.py와 동일 구조)
    3) 시간 순 split (train / eval)
    4) RMSE / MAE / R^2 출력
    5) 예측 vs 실제 시각화 (타임 시리즈 + 산점도)
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ─────────────────────────────────────────────
# 1. 공통 유틸
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """프로젝트 루트 경로 추정 (train_single_models.py와 동일 방식 가정)."""
    return Path(__file__).resolve().parents[2]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2. 피처 그룹 정의 (train_single_models.py와 동일하게 맞춤)
# ─────────────────────────────────────────────

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
                    time_features,
                    poi_features,
                    weather_features,
                    feature_mode: str = "all"):
    """feature_mode에 따라 사용할 피처 목록 선택."""
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

    # 1차: 실제 컬럼 존재 여부 체크
    cols = [c for c in cols if c in df.columns]

    # 2차: 숫자형 컬럼만 사용 (문자열 등 제거)
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols = [c for c in cols if c in numeric_cols]

    return cols


# ─────────────────────────────────────────────
# 3. 평가 / 시각화 함수
# ─────────────────────────────────────────────

def time_based_split(df: pd.DataFrame,
                     feature_cols,
                     target_col="rental_count",
                     val_ratio=0.2):
    """시간 순으로 정렬 후 뒤쪽 val_ratio 만큼을 평가용으로 사용."""
    if "date" in df.columns:
        df_sorted = df.sort_values("date").reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    df_model = df_sorted.dropna(subset=feature_cols + [target_col]).copy()

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    n = len(df_model)
    split_idx = int(n * (1 - val_ratio))

    X_train, X_eval = X[:split_idx], X[split_idx:]
    y_train, y_eval = y[:split_idx], y[split_idx:]
    dates_eval = df_model["date"].iloc[split_idx:] if "date" in df_model.columns else None

    return X_train, X_eval, y_train, y_eval, dates_eval, feature_cols


def evaluate_models(city_name: str,
                    df: pd.DataFrame,
                    feature_mode: str,
                    model_dir: Path,
                    output_dir: Path,
                    use_iqr_version: bool = False):
    """
    한 도시(city)에 대해:
      - 피처 선택
      - Train/Eval split
      - 모델 로드
      - 성능 측정 & 시각화
    """

    print(f"\n==============================")
    print(f"  City: {city_name}, feature_mode={feature_mode}")
    print(f"==============================")

    # 도시별 피처 그룹 정의
    if city_name.lower().startswith("seoul"):
        time_features, poi_features, weather_features = get_feature_groups_seoul(df)
    else:
        time_features, poi_features, weather_features = get_feature_groups_dc(df)

    feature_cols = select_features(df, time_features, poi_features, weather_features, feature_mode)
    print(f"[INFO] 사용 피처 수: {len(feature_cols)}")
    print(f"[INFO] 사용 피처 목록: {feature_cols}")

    X_train, X_eval, y_train, y_eval, dates_eval, feature_cols = time_based_split(
        df, feature_cols, target_col="rental_count", val_ratio=0.2
    )

    print(f"[INFO] Train size: {X_train.shape}, Eval size: {X_eval.shape}")

    # ── 모델 로드 ──
    version_tag = "iqr" if use_iqr_version else "no_iqr"
    feature_mode_for_name = "all"  # 파일명은 all 기준으로 저장됨

    model_short_name = {
        "RandomForest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
    }

    model_paths = {
        model_name: model_dir / f"{city_name}_{model_short_name[model_name]}_{feature_mode_for_name}_{version_tag}.pkl"
        for model_name in model_short_name.keys()
    }

    # 피처 수가 안 맞는 모델은 로드 후 바로 제외하기 위해 dict 분리
    models = {}
    for name, path in model_paths.items():
        if not path.exists():
            print(f"[WARN] 모델 파일 없음, 스킵: {name} -> {path}")
            continue
        model = joblib.load(path)

        # 피처 수 확인
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and X_eval.shape[1] != expected_features:
            print(
                f"[ERROR] 피처 불일치: {name} 모델은 {expected_features} 피처를 기대하지만, "
                f"현재 평가용 X_eval은 {X_eval.shape[1]} 피처입니다."
            )
            print(f"[DEBUG] 현재 사용 중인 feature_cols ({len(feature_cols)}개): {feature_cols}")
            print(f"[DEBUG] 이 모델은 학습 시 {expected_features}개의 피처로 학습되었습니다.")
            print("       → get_feature_groups_* 또는 데이터 전처리(컬럼명/타입)를 학습 시점과 맞춰야 합니다.")
            # 이 모델은 스킵
            continue

        models[name] = model
        print(f"[LOAD] {name} 모델 로드 완료: {path}")

    if not models:
        print("[ERROR] 로드된 모델이 없음. model_paths 또는 feature_cols 설정을 확인하세요.")
        return

    # ── 성능 측정 및 시각화 ──
    ensure_dir(output_dir)
    metrics_rows = []

    # Eval 시각화를 위해, 너무 많으면 마지막 일부만 사용
    max_points_for_plot = 1000
    if len(y_eval) > max_points_for_plot:
        y_eval_plot = y_eval[-max_points_for_plot:]
        if dates_eval is not None:
            dates_eval_plot = dates_eval.iloc[-max_points_for_plot:]
        else:
            dates_eval_plot = np.arange(len(y_eval_plot))
        idx_plot_start = len(y_eval) - max_points_for_plot
    else:
        y_eval_plot = y_eval
        dates_eval_plot = dates_eval if dates_eval is not None else np.arange(len(y_eval_plot))
        idx_plot_start = 0

    # 타임시리즈 플롯 준비
    plt.figure(figsize=(14, 6))
    plt.plot(dates_eval_plot, y_eval_plot, label="Actual", linewidth=1)

    for model_name, model in models.items():
        # 예측
        y_pred_eval = model.predict(X_eval)

        # 전체 eval 성능
        _rmse = rmse(y_eval, y_pred_eval)
        _mae = mean_absolute_error(y_eval, y_pred_eval)
        _r2 = r2_score(y_eval, y_pred_eval)

        metrics_rows.append({
            "city": city_name,
            "feature_mode": feature_mode_for_name,
            "version": version_tag,
            "model": model_name,
            "RMSE": _rmse,
            "MAE": _mae,
            "R2": _r2,
        })

        print(f"\n[{city_name}] {model_name} (Eval)")
        print(f"  RMSE: { _rmse: .4f}")
        print(f"  MAE : { _mae: .4f}")
        print(f"  R^2 : { _r2: .4f}")

        # 타임 시리즈 일부 구간 예측 vs 실제
        y_pred_plot = y_pred_eval[idx_plot_start:]
        plt.plot(dates_eval_plot, y_pred_plot, label=f"Pred-{model_name}", alpha=0.8, linewidth=1)

        # 산점도 플롯 (실제 vs 예측)
        plt_scatter = plt.figure(figsize=(6, 6))
        plt_scatter_ax = plt_scatter.add_subplot(111)
        plt_scatter_ax.scatter(y_eval, y_pred_eval, s=2, alpha=0.5)
        min_v = min(y_eval.min(), y_pred_eval.min())
        max_v = max(y_eval.max(), y_pred_eval.max())
        plt_scatter_ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
        plt_scatter_ax.set_xlabel("Actual rental_count")
        plt_scatter_ax.set_ylabel("Predicted rental_count")
        plt_scatter_ax.set_title(f"{city_name} - {model_name} ({feature_mode_for_name}, {version_tag})")

        scatter_path = output_dir / f"{city_name}_{feature_mode_for_name}_{model_name}_{version_tag}_scatter.png"
        plt_scatter.tight_layout()
        plt_scatter.savefig(scatter_path, dpi=150)
        plt.close(plt_scatter)
        print(f"[SAVE] 산점도 플롯 저장: {scatter_path}")

    # 타임시리즈 플롯 저장
    plt.title(f"{city_name} Eval - Actual vs Pred (feature_mode={feature_mode_for_name}, version={version_tag})")
    plt.xlabel("Time")
    plt.ylabel("rental_count")
    plt.legend()
    plt.tight_layout()

    line_path = output_dir / f"{city_name}_{feature_mode_for_name}_{version_tag}_timeseries.png"
    plt.savefig(line_path, dpi=150)
    plt.close()
    print(f"[SAVE] 타임 시리즈 플롯 저장: {line_path}")

    # 메트릭 CSV 저장
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = output_dir / f"{city_name}_{feature_mode_for_name}_{version_tag}_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[SAVE] 메트릭 CSV 저장: {metrics_csv_path}")


# ─────────────────────────────────────────────
# 4. main 함수
# ─────────────────────────────────────────────

def main():
    project_root = get_project_root()
    print(f"Project root: {project_root}")

    # 데이터 경로
    seoul_csv = project_root / "Data" / "interim" / "seoul" / "seoul_rental_data.csv"
    dc_csv = project_root / "Data" / "interim" / "washington" / "dc_rental_data.csv"

    print(f"Seoul CSV: {seoul_csv}")
    print(f"DC CSV   : {dc_csv}")

    # ── 데이터 로드 ──
    df_seoul = pd.read_csv(seoul_csv)
    df_dc = pd.read_csv(dc_csv)

    # ✅ DC 쪽도 학습 코드와 동일하게 quarter 컬럼 rename
    if "quarter of day" in df_dc.columns and "quarter_flag" not in df_dc.columns:
        print("[INFO] Renaming 'quarter of day' -> 'quarter_flag' (WashingtonDC)")
        df_dc = df_dc.rename(columns={"quarter of day": "quarter_flag"})

    # date 컬럼 datetime 변환 (정렬용)
    for df in (df_seoul, df_dc):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

    # 모델 저장 경로 (실제 train_single_models.py와 맞춰줘야 함)
    model_dir = project_root / "models" / "no_IQR"
    output_dir = project_root / "docs" / "3weeks" / "evaluation"

    ensure_dir(model_dir)
    ensure_dir(output_dir)

    feature_mode = "all"

    # No-IQR 버전 평가
    evaluate_models("Seoul", df_seoul, feature_mode, model_dir, output_dir, use_iqr_version=False)
    evaluate_models("WashingtonDC", df_dc, feature_mode, model_dir, output_dir, use_iqr_version=False)


if __name__ == "__main__":
    main()
