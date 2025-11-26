import pandas as pd
from pathlib import Path

def add_time_features(input_csv, output_csv, datetime_col="datatime"):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    # 1) CSV 읽기
    df = pd.read_csv(input_csv)

    # 2) datatime 컬럼 → datetime 타입으로 변환
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # 3) 날짜/시간 구성요소 추가
    df["year"] = df[datetime_col].dt.year
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["hour"] = df[datetime_col].dt.hour

    # 4) 6시간 단위 쿼터 플래그 생성
    #    00~05 → 0
    #    06~11 → 1
    #    12~17 → 2
    #    18~23 → 3
    slot = df["hour"] // 6
    mapping = {0: 0, 1: 1, 2: 2, 3: 3}
    df["quarter_flag"] = slot.map(mapping)

    # 5) 저장 (기존 데이터 + 새 컬럼 모두 포함)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 변환 완료: {output_csv}")


if __name__ == "__main__":
    add_time_features(
        input_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged.csv",
        output_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features.csv",
        datetime_col="dt"
    )
