#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
따릉이 대여소별 거치수량 품질 검사 + 마크다운 리포트 자동 생성 스크립트
(argparse 제거 버전 — main()에서 바로 CSV 경로 지정)
"""

import pandas as pd
import numpy as np
from datetime import datetime



# 🔥 CSV 절대 경로 (WSL 기준)
# -------------------------------------------------------------------
CSV_PATH = "/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/raw/seoul/Operational Event Data/Daily holding quantity by Rentor(day)/2021.01~2021.05.csv"
# -------------------------------------------------------------------


def load_data(path: str) -> pd.DataFrame:
    """CSV 파일 로드"""
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp949")
    return df


def write_md(lines, path="quality_report.md"):
    """마크다운 파일 저장"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n📄 마크다운 리포트 생성 완료 → {path}")


def generate_report(df):
    """품질 검사 결과를 마크다운 리스트로 생성"""
    md = []
    md.append(f"# 📊 따릉이 대여소별 거치수량 데이터 품질 검사 리포트")
    md.append(f"생성일: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n")
    md.append("---\n")

    # 1. 기본 정보
    md.append("## 1. 기본 정보")
    md.append(f"- 총 행 수: **{df.shape[0]}**")
    md.append(f"- 총 열 수: **{df.shape[1]}**")
    md.append(f"- 컬럼: `{', '.join(df.columns)}`\n")

    # 2. datetime
    df["일시_dt"] = pd.to_datetime(df["일시"], errors="coerce")
    invalid_dt = df["일시_dt"].isna().sum()
    md.append("## 2. 일시(datetime) 변환 검사")
    md.append(f"- 변환 실패(NaT): **{invalid_dt}건**\n")

    # 3. 결측치
    md.append("## 3. 결측치 검사")
    nulls = df.isna().sum()
    md.append("| 컬럼 | 결측치 개수 |")
    md.append("|------|------------|")
    for col, cnt in nulls.items():
        md.append(f"| {col} | {cnt} |")
    md.append("")

    # 4. 중복 검사
    md.append("## 4. 중복 레코드 검사")
    dup_cnt = df.duplicated(subset=["일시_dt", "대여소번호"]).sum()
    md.append(f"- (일시_dt, 대여소번호) 기준 중복 행: **{dup_cnt}건**\n")

    # 5. 거치수량 숫자 변환
    df["거치수량_num"] = pd.to_numeric(df["거치수량"], errors="coerce")
    num_invalid = df["거치수량_num"].isna().sum()

    md.append("## 5. 거치수량 값 품질 검사")
    md.append(f"- 숫자변환 실패/결측: **{num_invalid}건**")

    neg = df[df["거치수량_num"] < 0].shape[0]
    md.append(f"- 음수 값: **{neg}건**")

    q99 = df["거치수량_num"].quantile(0.99)
    threshold = q99 * 2
    extreme = df[df["거치수량_num"] > threshold].shape[0]
    md.append(f"- 극단값(> 2 × 99th percentile): **{extreme}건**\n")

    md.append("\n---\n")
    md.append("### ✔ 품질 검사 자동화 완료\n이 리포트는 데이터 정제 및 수요 예측 모델링 준비 단계에서 활용 가능합니다.")

    return md


def main():
    print(f"📂 CSV 로드 중: {CSV_PATH}")
    df = load_data(CSV_PATH)

    md = generate_report(df)
    write_md(md)


if __name__ == "__main__":
    main()
