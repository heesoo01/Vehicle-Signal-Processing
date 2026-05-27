import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 상위 폴더(utils)를 인식하기 위한 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.fft_processor import process_residual_to_psd

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"

NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
ISC_DST_DIR = DATASET_ROOT / "NCM811_ISC_TEST" / "DST"

PROCESSED_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DIR.mkdir(exist_ok=True)


def main():
    print("=== 1단계: OCV-SOC 기준표 생성 중 ===")
    normal_cc_data = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc_data)
    capacity_coulomb = normal_cc_data["capacity_ah"].max() * 3600.0

    normal_psd_list = []
    isc_psd_list = []

    # --- 정상 데이터(Normal) 전체 변환 ---
    normal_files = list(NORMAL_DST_DIR.glob("*.csv"))
    print(f"\n=== 2단계: 정상 주행 데이터 변환 시작 (총 {len(normal_files)}개) ===")

    for i, file_path in enumerate(normal_files):
        print(f"  -> Normal 변환 중 ({i + 1}/{len(normal_files)}): {file_path.name}")
        target_data = add_coulomb_counted_soc(load_right_block(file_path), capacity_coulomb)
        result_data = run_ekf(target_data, soc_table, ocv_table, capacity_coulomb, initialize_vrc=True)

        freqs, psd = process_residual_to_psd(result_data, crop_seconds=500)
        normal_psd_list.append(psd)

    # --- 단락 데이터(ISC) 전체 변환 ---
    isc_files = list(ISC_DST_DIR.glob("*.csv"))
    print(f"\n=== 3단계: 단락(ISC) 데이터 변환 시작 (총 {len(isc_files)}개) ===")

    for i, file_path in enumerate(isc_files):
        print(f"  -> ISC 변환 중 ({i + 1}/{len(isc_files)}): {file_path.name}")
        target_data = add_coulomb_counted_soc(load_right_block(file_path), capacity_coulomb)
        result_data = run_ekf(target_data, soc_table, ocv_table, capacity_coulomb, initialize_vrc=True)

        freqs, psd = process_residual_to_psd(result_data, crop_seconds=500)
        isc_psd_list.append(psd)

    # --- 최종 저장 ---
    print("\n=== 4단계: .npy 파일 영구 저장 중 ===")
    normal_array = np.array(normal_psd_list)
    isc_array = np.array(isc_psd_list)

    np.save(PROCESSED_DIR / "normal_psd.npy", normal_array)
    np.save(PROCESSED_DIR / "isc_psd.npy", isc_array)

    print(f"\n🎉 전체 데이터 변환 완료!")
    print(f"저장 경로: {PROCESSED_DIR}")
    print(f"Normal PSD 형태: {normal_array.shape}")
    print(f"ISC PSD 형태: {isc_array.shape}")


if __name__ == "__main__":
    main()