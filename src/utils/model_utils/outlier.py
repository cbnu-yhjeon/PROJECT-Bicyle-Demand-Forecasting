# src/utils/model_utils/outlier.py

from __future__ import annotations
from typing import List
import pandas as pd
import logging

# train_single_models.py 에서 설정한 로거를 같이 사용
LOGGER = logging.getLogger("bike_demand")


def remove_outliers_iqr(
    df: pd.DataFrame,
    cols: List[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    IQR 방식으로 이상치를 제거하는 함수.
    각 컬럼마다 제거 개수와 남은 행 수를 로그로 남긴다.
    """
    df_clean = df.copy()

    for col in cols:
        if col not in df_clean.columns:
            LOGGER.info(f"[IQR] 컬럼 없음: {col} (스킵)")
            continue

        series = df_clean[col]

        # 전부 NaN 이면 스킵
        if series.isna().all():
            LOGGER.info(f"[IQR] {col}: 전체 NaN (스킵)")
            continue

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR

        before = len(df_clean)
        mask = (series >= lower) & (series <= upper)
        df_clean = df_clean[mask]
        removed = before - len(df_clean)

        LOGGER.info(f"[IQR] {col}: 제거 {removed}개 (남은 행: {len(df_clean)})")

    return df_clean
