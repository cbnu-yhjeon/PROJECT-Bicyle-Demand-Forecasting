#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
자전거 이용정보(시간별) CSV 품질 리포트 생성 스크립트
- argument 사용 X
- 스크립트 내부에서 경로 직접 지정
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# ★ 1. 파일 경로 설정
# =========================
INPUT_CSV = Path("/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/raw/seoul/Time-series Data/Rental_Bike_usage/20~25/merged_20.csv")
OUTPUT_MD = Path("/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/raw/seoul/Time-series Data/Rental_Bike_usage/20~25/bike_usage_quality_report.md")

# =========================
# ★ 2. 기대 Column 스키마
# =========================
EXPECTED_COLUMNS = [
    "rental_date",
    "rental_hour",
    "station_id",
    "station_name",
    "rental_type",
    "gender",
    "age_group",
    "count",
    "calories",
    "carbon_reduction",
    "distance_m",
    "usage_time_min",
]


# =========================
# ★ 3. CSV 로드 함수
# =========================
def load_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "cp949","EUC-KR"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[INFO] CSV Loaded ({enc}) → {path}")
            return df
        except Exception:
            pass
    raise RuntimeError(f"❌ CSV 파일을 읽지 못했습니다: {path}")


# =========================
# ★ 4. 스키마 점검
# =========================
def check_schema(df: pd.DataFrame) -> str:
    md = []
    md.append("## 1. 스키마 점검\n")

    schema = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "missing": df.isna().sum(),
    })
    schema["missing_rate(%)"] = (schema["missing"] / len(df) * 100).round(2)
    schema["unique"] = df.nunique()
    schema["example"] = [
        str(df[c].dropna().iloc[0]) if df[c].notna().any() else ""
        for c in df.columns
    ]

    md.append(schema.to_markdown(index=False))
    md.append("")

    # 스키마 비교
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)

    missing_cols = sorted(list(expected - actual))
    extra_cols = sorted(list(actual - expected))

    md.append("### 1-2. 기대 스키마 비교\n")
    if missing_cols:
        md.append(f"- ⚠ 누락된 컬럼: `{', '.join(missing_cols)}`")
    else:
        md.append("- ✔ 누락 없음")

    if extra_cols:
        md.append(f"- ℹ 추가 컬럼: `{', '.join(extra_cols)}`")
    else:
        md.append("- ✔ 불필요한 추가 컬럼 없음")

    md.append("")
    return "\n".join(md)


# =========================
# ★ 5. 결측 및 중복 점검
# =========================
def check_missing_duplicates(df: pd.DataFrame) -> str:
    md = []
    md.append("## 2. 결측치 및 중복 점검\n")

    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum(),
        "missing_rate(%)": (df.isna().sum() / len(df) * 100).round(2)
    })

    md.append("### 2-1. 결측치\n")
    md.append(missing.sort_values("missing_rate(%)", ascending=False).to_markdown(index=False))

    md.append("\n### 2-2. 중복 행")
    dup = df.duplicated().sum()
    md.append(f"- 중복 행 수: **{dup}**")
    if dup == 0:
        md.append("- ✔ 중복 없음")
    else:
        md.append("- ⚠ 중복 제거 필요")

    md.append("")
    return "\n".join(md)


# =========================
# ★ 6. 수치형 통계 및 범위
# =========================
def check_numeric(df: pd.DataFrame) -> str:
    md = []
    md.append("## 3. 수치형 통계 및 범위\n")

    num_cols = df.select_dtypes(include=[np.number]).columns
    desc = df[num_cols].describe().transpose()

    md.append("### 3-1. 기본 통계\n")
    md.append(desc.to_markdown())
    md.append("")

    md.append("### 3-2. 값 범위\n")
    ranges = pd.DataFrame({
        "column": num_cols,
        "min": [df[c].min() for c in num_cols],
        "max": [df[c].max() for c in num_cols],
    })

    md.append(ranges.to_markdown(index=False))
    md.append("")
    return "\n".join(md)


# =========================
# ★ 7. 도메인 규칙 기반 검사
# =========================
def check_domain(df: pd.DataFrame) -> str:
    md = []
    md.append("## 4. 도메인 규칙 기반 검사\n")

    # 시간 범위
    if "rental_hour" in df.columns:
        invalid = df[(df["rental_hour"] < 0) | (df["rental_hour"] > 23)]
        md.append(f"- rental_hour 범위 외 값: **{len(invalid)}**")

    # 음수 불가 항목
    non_negative_cols = ["count", "calories", "carbon_reduction", "distance_m", "usage_time_min"]
    for col in non_negative_cols:
        if col in df.columns:
            invalid = df[df[col] < 0]
            md.append(f"- `{col}` 음수값 개수: **{len(invalid)}**")

    # 범주형 분포
    if "gender" in df.columns:
        md.append("\n### gender 분포\n")
        md.append(df["gender"].value_counts(dropna=False).to_markdown())

    if "age_group" in df.columns:
        md.append("\n### age_group 분포\n")
        md.append(df["age_group"].value_counts(dropna=False).to_markdown())

    if "rental_type" in df.columns:
        md.append("\n### rental_type 분포\n")
        md.append(df["rental_type"].value_counts(dropna=False).to_markdown())

    md.append("")
    return "\n".join(md)


# =========================
# ★ 8. 전체 리포트 생성
# =========================
def generate_report():
    df = load_csv(INPUT_CSV)

    md = []
    md.append("# 🚲 자전거 이용정보 품질 리포트\n")
    md.append(f"- 입력 파일: `{INPUT_CSV}`")
    md.append(f"- 총 행 수: **{len(df):,}**")
    md.append(f"- 총 컬럼 수: **{len(df.columns)}**\n")

    md.append(check_schema(df))
    md.append(check_missing_duplicates(df))
    md.append(check_numeric(df))
    md.append(check_domain(df))

    OUTPUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[INFO] 품질 리포트 생성 완료 → {OUTPUT_MD.resolve()}")


# =========================
# ★ 실행
# =========================
if __name__ == "__main__":
    generate_report()
