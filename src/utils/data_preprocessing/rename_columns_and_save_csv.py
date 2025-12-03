import pandas as pd
from pathlib import Path

# CSV 파일 읽기 (Linux 경로 형식 사용)
input_file = '/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/interim/seoul/bicycle_final_data.csv'
output_file = '/home/avg/PROJECT-Bicyle-Demand-Forecasting/Data/interim/seoul/seoul_rental_data.csv'

# Path 객체로 변환하여 운영체제에 관계없이 안전하게 처리
input_path = Path(input_file)
output_path = Path(output_file)

# 파일 존재 여부 확인
if not input_path.exists():
    print(f"❌ 파일을 찾을 수 없습니다: {input_path}")
    exit(1)

# CSV 파일을 DataFrame으로 로드
df = pd.read_csv(input_path, encoding='utf-8-sig')

# 현재 칼럼명 확인
print("현재 칼럼명:")
print(df.columns.tolist())

# 칼럼명 변경
# 방법 1: 딕셔너리로 특정 칼럼만 변경
column_mapping = {
    '거치대수': 'Holding quantity',
}

df.rename(columns=column_mapping, inplace=True)

# 변경된 칼럼명 확인
print("\n변경된 칼럼명:")
print(df.columns.tolist())

# 변경된 DataFrame을 새 CSV 파일로 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 파일이 저장되었습니다: {output_path}")
