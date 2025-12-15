"""
notebook_ensemble_visualization.py (Python 3.8+ compatible)

⚠️ Fix:
- Removed PEP604 union type hints (e.g., `str | Path`) which break on Python 3.8.
- Replaced with typing.Union / Optional.

Jupyter Notebook에서 바로 복붙해서 실행할 수 있는 "결과 시각화 코드"입니다.

✅ 지원 입력
- ensemble 결과 CSV (예: ensemble_result/*.csv)
  - 최소 컬럼: date, y_pred_ensemble
  - 있으면 더 좋음: y_true, quarter_flag, slot_ts, (WDC의 경우 STATION_ID/NAME/name)

✅ 시각화
1) (가능하면) slot_ts 기준 시계열: y_true vs y_pred_ensemble
2) (slot_ts가 없으면) date + quarter_flag로 slot_ts 생성해서 시계열
3) (둘 다 없으면) date 단위로 집계(daily mean) 시계열
4) Scatter: y_true vs y_pred_ensemble
5) Residual histogram
6) 날짜별/슬롯별 샘플 수(n) 확인
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _make_slot_ts_if_possible(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    slot_ts 우선순위:
    1) slot_ts가 있으면 그대로 사용
    2) date + quarter_flag 있으면 slot_ts 생성 (quarter_flag*6시간)
    3) 아니면 slot_ts 생성 불가 (date만 유지)
    """
    if "slot_ts" in df.columns:
        df = _ensure_datetime(df, "slot_ts")
        return df

    if date_col in df.columns:
        df = _ensure_datetime(df, date_col)

    if date_col in df.columns and "quarter_flag" in df.columns:
        q = pd.to_numeric(df["quarter_flag"], errors="coerce").fillna(0).astype(int)
        df["quarter_flag"] = q
        df["slot_ts"] = df[date_col] + pd.to_timedelta(q * 6, unit="h")
        return df

    return df


def visualize_ensemble_result(
    csv_path: Union[str, Path],
    date_col: str = "date",
    pred_col: str = "y_pred_ensemble",
    true_col: str = "y_true",
    title_prefix: Optional[str] = None,
    aggregate_if_needed: bool = True,
) -> None:
    """
    Parameters
    ----------
    csv_path : str | Path
        ensemble_result/*.csv 경로
    date_col : str
        날짜 컬럼명
    pred_col : str
        앙상블 예측 컬럼명 (기본 y_pred_ensemble)
    true_col : str
        실제값 컬럼명 (있을 때만 y_true)
    title_prefix : str | None
        그래프 제목 prefix
    aggregate_if_needed : bool
        slot_ts가 없으면 date 단위로 평균 집계해서 시계열을 그릴지 여부
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if pred_col not in df.columns:
        raise KeyError("'{}' column not found. Columns: {}".format(pred_col, list(df.columns)))

    # 가능한 경우 slot_ts 생성
    df = _make_slot_ts_if_possible(df, date_col=date_col)

    # title
    if title_prefix is None:
        title_prefix = csv_path.stem

    # slot_ts 존재/타입 확인
    has_slot = ("slot_ts" in df.columns) and pd.api.types.is_datetime64_any_dtype(df["slot_ts"])

    # ─────────────────────────────────────────────
    # 1) 시계열 라인 플롯
    # ─────────────────────────────────────────────
    if has_slot:
        plot_df = df.dropna(subset=["slot_ts"]).sort_values("slot_ts").reset_index(drop=True)

        plt.figure(figsize=(12, 4))
        if true_col in plot_df.columns:
            plt.plot(plot_df["slot_ts"], plot_df[true_col], label=true_col)
        plt.plot(plot_df["slot_ts"], plot_df[pred_col], label=pred_col)
        plt.title("{} | Time Series (slot_ts)".format(title_prefix))
        plt.xlabel("slot_ts")
        plt.ylabel("rental_count")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 샘플 수 확인 (slot 기준)
        cnt = plot_df.groupby(plot_df["slot_ts"].dt.date).size()
        plt.figure(figsize=(12, 2.5))
        plt.plot(cnt.index, cnt.values)
        plt.title("{} | #Samples per day (from slot_ts)".format(title_prefix))
        plt.xlabel("date")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    else:
        # slot_ts가 없으면 date 중심
        if date_col not in df.columns:
            print("⚠️ date/slot_ts가 없어 시계열 플롯을 생략합니다.")
        else:
            df = _ensure_datetime(df, date_col)
            df2 = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

            if aggregate_if_needed:
                agg = {pred_col: "mean"}
                if true_col in df2.columns:
                    agg[true_col] = "mean"
                daily = df2.groupby(date_col, as_index=False).agg(agg)

                plt.figure(figsize=(12, 4))
                if true_col in daily.columns:
                    plt.plot(daily[date_col], daily[true_col], label="{} (daily mean)".format(true_col))
                plt.plot(daily[date_col], daily[pred_col], label="{} (daily mean)".format(pred_col))
                plt.title("{} | Time Series (daily mean)".format(title_prefix))
                plt.xlabel("date")
                plt.ylabel("rental_count")
                plt.legend()
                plt.tight_layout()
                plt.show()

                # 샘플 수
                cnt = df2.groupby(date_col).size()
                plt.figure(figsize=(12, 2.5))
                plt.plot(cnt.index, cnt.values)
                plt.title("{} | #Samples per date".format(title_prefix))
                plt.xlabel("date")
                plt.ylabel("count")
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(12, 4))
                if true_col in df2.columns:
                    plt.plot(df2[date_col], df2[true_col], label=true_col)
                plt.plot(df2[date_col], df2[pred_col], label=pred_col)
                plt.title("{} | Time Series (raw rows)".format(title_prefix))
                plt.xlabel("date")
                plt.ylabel("rental_count")
                plt.legend()
                plt.tight_layout()
                plt.show()

    # ─────────────────────────────────────────────
    # 2) Scatter + Residuals (y_true 있을 때)
    # ─────────────────────────────────────────────
    if true_col in df.columns:
        y_true = pd.to_numeric(df[true_col], errors="coerce")
        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        mask = (~y_true.isna()) & (~y_pred.isna())
        y_true = y_true[mask].to_numpy(dtype=float)
        y_pred = y_pred[mask].to_numpy(dtype=float)

        # Scatter
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, s=8, alpha=0.5)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        plt.plot([mn, mx], [mn, mx], linestyle="--", label="Ideal y=x")
        plt.title("{} | Scatter: {} vs {}".format(title_prefix, true_col, pred_col))
        plt.xlabel(true_col)
        plt.ylabel(pred_col)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Residuals
        resid = y_true - y_pred
        plt.figure(figsize=(7, 4))
        plt.hist(resid, bins=40)
        plt.title("{} | Residuals: ({} - {})".format(title_prefix, true_col, pred_col))
        plt.xlabel("residual")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

        # Metrics quick print
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
        print("✅ Metrics (row-level) | RMSE={:.4f}  MAE={:.4f}  R2={:.4f}".format(rmse, mae, r2))
    else:
        print("ℹ️ '{}' column not found → scatter/residual/metrics skipped.".format(true_col))


# Example
# visualize_ensemble_result("ensemble_result/Seoul_IQR_ensemble_preds_YYYYMMDD_HHMMSS.csv")
