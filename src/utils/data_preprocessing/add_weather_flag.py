import pandas as pd
import numpy as np
from pathlib import Path


def add_weather_flag(input_csv, output_csv):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    # 1) CSV 읽기
    df = pd.read_csv(input_csv)

    # 2) 강수/적설 컬럼에 남아 있을 수 있는 NaN 방어적으로 0.0으로 처리
    for col in ["precipitation", "snowcover"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
        else:
            print(f"⚠ 컬럼 없음: {col} (weather_flag 계산에서 제외됨)")

    # 3) 조건 정의
    is_rain = df["precipitation"] > 0
    is_snow = df["snowcover"] > 0

    # 4) weather_flag 생성
    #    0: 건조 (비X, 눈X)
    #    1: 비만
    #    2: 눈만
    #    3: 비+눈 (혼합)
    df["weather_flag"] = np.select(
        [
            (~is_rain & ~is_snow),  # 건조
            (is_rain & ~is_snow),   # 비
            (~is_rain & is_snow),   # 눈
            (is_rain & is_snow),    # 비+눈
        ],
        [0, 1, 2, 3],
        default=0,  # 혹시 이상 케이스가 있더라도 일단 0(건조)로
    )

    # 5) 저장 (기존 컬럼 + weather_flag 모두 포함)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ weather_flag 추가 완료: {output_csv}")


if __name__ == "__main__":
    add_weather_flag(
        input_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features_filled.csv",
        output_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/seoul/Contextual Data/weather/merged_with_time_features_filled_flagged.csv",
    )
