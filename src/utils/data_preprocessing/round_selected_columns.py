import pandas as pd
from pathlib import Path


def round_selected_columns(input_csv, output_csv, columns, digits=2):
    """
    지정된 컬럼들만 소수점 digits 자리로 반올림하여 저장하는 함수.

    Parameters:
        input_csv (str): 입력 CSV 경로
        output_csv (str): 출력 CSV 경로
        columns (list): 소수점 반올림 적용할 컬럼명 리스트
        digits (int): 반올림 자릿수 (기본값 2)
    """

    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    print(f"📂 CSV 로드 중: {input_csv}")
    df = pd.read_csv(input_csv)

    # 존재하는 컬럼만 골라서 처리
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        print("⚠ 반올림 처리할 수 있는 대상 컬럼이 없습니다.")
    else:
        print(f"🔧 반올림 적용 컬럼: {available_cols}")
        df[available_cols] = df[available_cols].round(digits)

    # 저장
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 저장 완료: {output_csv}")


if __name__ == "__main__":
    round_selected_columns(
        input_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/join/merged_with_weather.csv",
        output_csv=r"/mnt/c/projects/PROJECT-Bicyle-Demand-Forecasting/Data/processed/join/merged_with_weather_rounded.csv",

        # 🔥 소수점 두자리 유지할 컬럼 목록
        columns=[
            "used_time(avg)",
            "used_dis(avg)",
        ],


    digits=2
    )
