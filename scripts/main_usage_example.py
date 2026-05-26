import sys
from pathlib import Path
import pandas as pd

# utils 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_dataset_dod_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.denoise import denoise_innovation, plot_denoising_result

# 데이터셋 경로
DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0/swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST/CC/ISC_BD_0.5CC_0.5CD_1000ohm.csv"

def main():
    # 1. 정상 CC 데이터 로드
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc])
    capacity_coulomb = normal_cc["capacity_ah"].max() * 3600.0

    # 2. EKF 적용 → residual 생성
    target_data = add_dataset_dod_soc(normal_cc)
    result_data = run_ekf(
        target_data,
        soc_table,
        ocv_table,
        capacity_coulomb,
        initialize_vrc=True
    )

    # 3. denoise 호출 → filtered_residual 생성
    result_data = denoise_innovation(result_data, source_col="residual")

    # 4. denoise 결과 그래프 저장
    plot_denoising_result(
        result_data,
        output_path=PROJECT_ROOT / "images" / "denoise_result.png"
    )

    # 확인
    print(result_data[["time", "residual", "filtered_residual"]].head())
    print("그래프 저장 완료:", PROJECT_ROOT / "images" / "denoise_result.png")

if __name__ == "__main__":
    main()