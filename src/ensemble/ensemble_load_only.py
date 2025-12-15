# -*- coding: utf-8 -*-
"""
ensemble_load_only_v4.py

Changes vs v3
-------------
✅ Added output shuffling option so saved CSV isn't visually concentrated on a single date.

- By default:
  - Prediction/evaluation uses time-sorted order (safe for time-series).
  - Saved output is also time-sorted.

- If you pass `--shuffle_output`:
  - Only the *saved CSV rows* are shuffled (predictions are identical, just row order changes).
  - Evaluation still uses time-sorted order (so metrics remain meaningful).

Usage
-----
python src/ensemble/ensemble_load_only_v4.py --city seoul --variant IQR --save_preds --shuffle_output
python src/ensemble/ensemble_load_only_v4.py --city wdc --variant no_IQR --save_preds --shuffle_output --shuffle_seed 42
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


LOGGER = logging.getLogger("ensemble_load_only_v4")


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
def setup_logger(project_root: Path, name_prefix: str) -> Path:
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{name_prefix}_{ts}.log"

    LOGGER.setLevel(logging.INFO)
    if LOGGER.hasHandlers():
        LOGGER.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    LOGGER.addHandler(fh)
    LOGGER.addHandler(ch)

    LOGGER.info(f"📌 Log file: {log_path}")
    return log_path


# ─────────────────────────────────────────────────────────────
# Model IO (prefer your project util if available)
# ─────────────────────────────────────────────────────────────
def load_model(path: Path):
    try:
        from utils.model_utils.model_io import load_model as util_load  # type: ignore
        return util_load(path)
    except Exception:
        pass

    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# Feature groups (mirrors your training scripts)
# ─────────────────────────────────────────────────────────────
def get_feature_groups_seoul():
    time_features = ["month", "weekend", "quarter_flag"]

    poi_features = [
        "Holding quantity",
        "n_station_dis(m)", "n_bus_dis(m)", "n_school_dis(m)", "n_park_dis(m)",
        "N_of_stations_within_100m", "N_of_bus_within_100m", "N_of_school_within_100m", "N_of_park_within_100m",
        "N_of_stations_within_500m", "N_of_bus_within_500m", "N_of_school_within_500m", "N_of_park_within_500m",
        "N_of_stations_within_1000m", "N_of_bus_within_1000m", "N_of_school_within_1000m", "N_of_park_within_1000m",
        "N_of_stations_within_1500m", "N_of_bus_within_1500m", "N_of_school_within_1500m", "N_of_park_within_1500m",
        "N_of_stations_within_2000m", "N_of_bus_within_2000m", "N_of_school_within_2000m", "N_of_park_within_2000m",
        "used_time(avg)", "used_dis(avg)",  # optional usage features
    ]

    weather_features = [
        "temperature", "Precipitation", "windspeed", "humidity",
        "sunshine", "snowcover", "cloudcover", "weathersit",
    ]
    return time_features, poi_features, weather_features


def get_feature_groups_wdc():
    time_features = ["month", "weekend", "quarter_flag"]

    poi_features = [
        "CAPACITY",
        "n_station_idx", "n_station_dis(m)",
        "N_of_station_within_100m", "N_of_station_within_500m",
        "N_of_station_within_1000m", "N_of_station_within_1500m", "N_of_station_within_2000m",
        "n_bus_idx", "n_bus_dis(m)",
        "N_of_bus_within_100m", "N_of_bus_within_500m",
        "N_of_bus_within_1000m", "N_of_bus_within_1500m", "N_of_bus_within_2000m",
        "n_park_idx", "n_park_dis(m)",
        "N_of_park_within_100m", "N_of_park_within_500m",
        "N_of_park_within_1000m", "N_of_park_within_1500m", "N_of_park_within_2000m",
        "n_school_idx", "n_school_dis(m)",
        "N_of_school_within_100m", "N_of_school_within_500m",
        "N_of_school_within_1000m", "N_of_school_within_1500m", "N_of_school_within_2000m",
        "used_time(avg)", "used_dis(avg)",  # optional usage features
    ]

    weather_features = [
        "temperature", "humidity", "windspeed", "cloud_cover",
        "shortwave_radiation", "precipitation", "rain",
        "snowfall", "snow_depth", "weathersit",
    ]
    return time_features, poi_features, weather_features


def select_features(df: pd.DataFrame,
                    time_features: List[str],
                    poi_features: List[str],
                    weather_features: List[str],
                    feature_mode: str = "all") -> List[str]:
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

    cols = [c for c in cols if c in df.columns]
    numeric_cols = df.select_dtypes(include=["number"]).columns
    cols = [c for c in cols if c in numeric_cols]
    return cols


# ─────────────────────────────────────────────────────────────
# Extract training feature list from model artifacts
# ─────────────────────────────────────────────────────────────
def feature_cols_from_model(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        try:
            return list(model.feature_names_in_)
        except Exception:
            pass

    if hasattr(model, "feature_name_"):
        try:
            return list(model.feature_name_)
        except Exception:
            pass

    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass

    return None


def expected_n_features(model) -> Optional[int]:
    if hasattr(model, "n_features_in_"):
        try:
            return int(model.n_features_in_)
        except Exception:
            return None
    if hasattr(model, "_n_features"):
        try:
            return int(model._n_features)
        except Exception:
            return None
    return None


def is_f_index_style(names: List[str]) -> bool:
    return bool(names) and all(isinstance(c, str) and c.startswith("f") and c[1:].isdigit() for c in names)


def build_aligned_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    missing = [c for c in feature_cols if c not in df.columns]
    for c in missing:
        df[c] = 0.0

    X = df.reindex(columns=feature_cols)
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return X, missing


# ─────────────────────────────────────────────────────────────
# Time-series split indices (simple holdout)
# ─────────────────────────────────────────────────────────────
def time_series_split_indices(n: int, val_size: float = 0.15, test_size: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_val - n_test
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Tuple[float, float, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    LOGGER.info(f"[{label}] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")
    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────
# Slot timestamp builder
# ─────────────────────────────────────────────────────────────
def ensure_quarter_flag(df: pd.DataFrame, city: str) -> pd.DataFrame:
    if city == "wdc" and "quarter of day" in df.columns and "quarter_flag" not in df.columns:
        df = df.rename(columns={"quarter of day": "quarter_flag"})
    if "quarter_flag" in df.columns:
        df["quarter_flag"] = pd.to_numeric(df["quarter_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["quarter_flag"] = 0
    return df


def add_slot_ts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["slot_ts"] = df[date_col] + pd.to_timedelta(df["quarter_flag"] * 6, unit="h")
    return df


# ─────────────────────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────────────────────
@dataclass
class WeightedAveragingEnsemble:
    weights: Dict[str, float]
    model_paths: Dict[str, Path]
    feature_cols_per_model: Dict[str, List[str]]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(df), dtype=float)
        for key in ["lgbm", "xgb", "rf"]:
            m = load_model(self.model_paths[key])
            feat = self.feature_cols_per_model[key]
            X, missing = build_aligned_matrix(df, feat)
            if missing:
                LOGGER.warning(f"[FEATURES] {key.upper()} missing columns filled with 0.0: {missing}")
            preds += float(self.weights[key]) * m.predict(X)
        return preds


# ─────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────
def resolve_model_paths(project_root: Path, city_name: str, variant_dir: str) -> Dict[str, Path]:
    models_root = project_root / "models" / variant_dir

    if variant_dir == "IQR":
        file_map = {
            "lgbm": f"{city_name}_LGBM_all.pkl",
            "xgb":  f"{city_name}_XGB_all.pkl",
            "rf":   f"{city_name}_RF_all.pkl",
        }
    else:
        file_map = {
            "lgbm": f"{city_name}_LGBM_all_no_iqr.pkl",
            "xgb":  f"{city_name}_XGB_all_no_iqr.pkl",
            "rf":   f"{city_name}_RF_all_no_iqr.pkl",
        }

    paths = {k: (models_root / v) for k, v in file_map.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))
    return paths


def resolve_data_path(project_root: Path, city: str) -> Path:
    if city == "wdc":
        return project_root / "Data" / "interim" / "washington" / "dc_rental_data.csv"
    return project_root / "Data" / "interim" / "seoul" / "seoul_rental_data.csv"


def pick_station_id_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in ["STATION_ID", "NAME", "name"]:
        if c in df.columns:
            cols.append(c)
    return cols


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, choices=["wdc", "seoul"])
    parser.add_argument("--variant", type=str, default="no_IQR", choices=["IQR", "no_IQR"])
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--date_col", type=str, default="date")
    parser.add_argument("--target_col", type=str, default="rental_count")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.4, 0.35, 0.25], help="lgbm xgb rf")
    parser.add_argument("--save_preds", action="store_true", help="Save predictions to <PROJECT_ROOT>/ensemble_result/")
    parser.add_argument("--shuffle_output", action="store_true", help="Shuffle only the saved CSV row order")
    parser.add_argument("--shuffle_seed", type=int, default=42, help="Seed for output shuffling")
    args = parser.parse_args()

    setup_logger(project_root, f"ensemble_load_only_v4_{args.city}_{args.variant}")

    # Normalize weights
    w_lgbm, w_xgb, w_rf = [float(x) for x in args.weights]
    weights = {"lgbm": w_lgbm, "xgb": w_xgb, "rf": w_rf}
    s = sum(weights.values())
    if s <= 0:
        raise ValueError("Sum of weights must be > 0")
    weights = {k: v / s for k, v in weights.items()}
    LOGGER.info(f"[WEIGHTS] normalized = {weights}")

    # Load CSV
    csv_path = Path(args.csv) if args.csv else resolve_data_path(project_root, args.city)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.date_col not in df.columns:
        raise KeyError(f"date_col '{args.date_col}' not found in CSV columns")

    # Time normalization (for evaluation safety)
    df = ensure_quarter_flag(df, args.city)
    df = add_slot_ts(df, args.date_col)
    df_sorted = df.sort_values(["slot_ts"]).reset_index(drop=True)

    # Resolve models
    city_name = "WashingtonDC" if args.city == "wdc" else "Seoul"
    variant_dir = "IQR" if args.variant == "IQR" else "no_IQR"
    model_paths = resolve_model_paths(project_root, city_name, variant_dir)
    LOGGER.info(f"[MODELS] {model_paths}")

    # Load models to extract feature lists
    models = {k: load_model(p) for k, p in model_paths.items()}

    feature_cols_per_model: Dict[str, List[str]] = {}
    for k, m in models.items():
        names = feature_cols_from_model(m)
        n_exp = expected_n_features(m)

        if names and not is_f_index_style(names):
            feature_cols_per_model[k] = list(names)
            LOGGER.info(f"[FEATURES] {k.upper()} uses {len(names)} features from model artifact.")
        else:
            if args.city == "seoul":
                tf, pf, wf = get_feature_groups_seoul()
            else:
                tf, pf, wf = get_feature_groups_wdc()
            fallback = select_features(df_sorted, tf, pf, wf, feature_mode="all")
            feature_cols_per_model[k] = fallback
            LOGGER.warning(f"[FEATURES] {k.upper()} has no reliable feature names -> using training-style list ({len(fallback)}).")

        if n_exp is not None and len(feature_cols_per_model[k]) != n_exp:
            LOGGER.warning(
                f"[FEATURES] {k.upper()} expects {n_exp} features, but selected {len(feature_cols_per_model[k])}. "
                f"Missing columns will be created and filled with 0.0 where possible."
            )

    ens = WeightedAveragingEnsemble(weights=weights, model_paths=model_paths, feature_cols_per_model=feature_cols_per_model)

    # Predict on time-sorted df (safe and reproducible)
    y_pred_all = ens.predict(df_sorted)
    LOGGER.info(f"[PRED] produced {len(y_pred_all)} predictions")

    # Optional evaluation (keep time order!)
    if args.target_col in df_sorted.columns:
        y_all = pd.to_numeric(df_sorted[args.target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        tr, va, te = time_series_split_indices(len(df_sorted))
        evaluate(y_all[tr], y_pred_all[tr], f"{city_name}-{variant_dir}-ENSEMBLE-TRAIN")
        evaluate(y_all[va], y_pred_all[va], f"{city_name}-{variant_dir}-ENSEMBLE-VAL")
        evaluate(y_all[te], y_pred_all[te], f"{city_name}-{variant_dir}-ENSEMBLE-TEST")

    # Save predictions (optionally shuffled)
    if args.save_preds:
        out_dir = project_root / "ensemble_result"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{city_name}_{variant_dir}_ensemble_preds_{ts}.csv"

        out_cols: List[str] = []
        if args.city == "wdc":
            out_cols += pick_station_id_cols(df_sorted)
        out_cols += [args.date_col, "quarter_flag", "slot_ts"]

        out_df = df_sorted[out_cols].copy()
        out_df["y_pred_ensemble"] = y_pred_all
        if args.target_col in df_sorted.columns:
            out_df["y_true"] = df_sorted[args.target_col].values

        if args.shuffle_output:
            out_df = out_df.sample(frac=1.0, random_state=int(args.shuffle_seed)).reset_index(drop=True)
            LOGGER.info(f"[SAVE] output rows shuffled (seed={args.shuffle_seed})")

        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        LOGGER.info(f"[SAVE] predictions saved: {out_path}")

    LOGGER.info("✅ Done.")


if __name__ == "__main__":
    main()
