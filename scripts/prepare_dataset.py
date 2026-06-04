import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.denoise import denoise_innovation
from utils.fft_processor import process_residual_to_psd

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
ISC_DST_DIR = DATASET_ROOT / "NCM811_ISC_TEST" / "DST"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "processed_data"
PROCESSED_DIR.mkdir(exist_ok=True)

ECM_FIXED = {"R0": 0.015, "R1": 0.01, "C1": 2400.0, "P0": np.diag([1e-4, 1e-3])}


def load_best_ekf_params():
    """cc_ekf_design.py가 저장한 최적 Q/R을 읽어 EKF 파라미터로 변환한다."""
    best_path = RESULTS_DIR / "cc_best_ekf_params.csv"
    grid_path = RESULTS_DIR / "cc_qr_grid_search.csv"

    if best_path.exists():
        best = pd.read_csv(best_path).iloc[0]
    elif grid_path.exists():
        best = pd.read_csv(grid_path).sort_values("score").iloc[0]
    else:
        raise FileNotFoundError(
            "최적 Q/R 결과 파일이 없습니다. 먼저 `python scripts/cc_ekf_design.py`를 실행하세요."
        )

    params = {
        **ECM_FIXED,
        "Q": np.diag([float(best["q_soc"]), float(best["q_vrc"])]),
        "R": np.array([[float(best["r"])]]),
    }
    print(
        f"최적 EKF Q/R 사용: q_soc={float(best['q_soc']):.3e}, "
        f"q_vrc={float(best['q_vrc']):.3e}, r={float(best['r']):.3e}"
    )
    return params


def ekf_denoise_psd(file_path: Path, soc_table, ocv_table, capacity_coulomb: float, ekf_params):
    """CSV -> EKF residual -> denoise -> filtered_residual PSD."""
    raw_data = load_right_block(file_path)
    target_data = add_coulomb_counted_soc(raw_data, capacity_coulomb)

    result_data = run_ekf(
        target_data,
        soc_table,
        ocv_table,
        capacity_coulomb,
        ekf_params=ekf_params,
        initialize_vrc=True,
    )

    result_data = denoise_innovation(
        result_data,
        source_col="residual",
        output_col="filtered_residual",
        cutoff_hz=0.05,
    )

    freqs, psd = process_residual_to_psd(
        result_data,
        crop_seconds=500,
        source_col="filtered_residual",
        nperseg=256,
    )
    return freqs, psd


def main():
    print("=== 1단계: OCV-SOC 기준표 생성 중 ===")
    normal_cc_data = load_right_block(NORMAL_CC_PATH)

    # build_ocv_table_from_cc()는 DataFrame 리스트를 기대하므로 [normal_cc_data]로 전달한다.
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc_data])
    capacity_coulomb = normal_cc_data["capacity_ah"].max() * 3600.0
    ekf_params = load_best_ekf_params()

    normal_psd_list = []
    isc_psd_list = []

    normal_files = sorted(NORMAL_DST_DIR.glob("*.csv"))
    print(f"\n=== 2단계: 정상 주행 데이터 변환 시작 (총 {len(normal_files)}개) ===")
    for i, file_path in enumerate(normal_files):
        print(f" -> Normal 변환 중 ({i + 1}/{len(normal_files)}): {file_path.name}")
        _, psd = ekf_denoise_psd(file_path, soc_table, ocv_table, capacity_coulomb, ekf_params)
        normal_psd_list.append(psd)

    isc_files = sorted(ISC_DST_DIR.glob("*.csv"))
    print(f"\n=== 3단계: 단락(ISC) 데이터 변환 시작 (총 {len(isc_files)}개) ===")
    for i, file_path in enumerate(isc_files):
        print(f" -> ISC 변환 중 ({i + 1}/{len(isc_files)}): {file_path.name}")
        _, psd = ekf_denoise_psd(file_path, soc_table, ocv_table, capacity_coulomb, ekf_params)
        isc_psd_list.append(psd)

    print("\n=== 4단계: .npy 파일 저장 중 ===")
    normal_array = np.array(normal_psd_list)
    isc_array = np.array(isc_psd_list)

    # 기존 파일명도 유지하되, denoise 적용 여부가 드러나는 파일명도 함께 저장한다.
    np.save(PROCESSED_DIR / "normal_psd.npy", normal_array)
    np.save(PROCESSED_DIR / "isc_psd.npy", isc_array)
    np.save(PROCESSED_DIR / "normal_psd_filtered.npy", normal_array)
    np.save(PROCESSED_DIR / "isc_psd_filtered.npy", isc_array)

    print("\n전체 데이터 변환 완료")
    print(f"저장 경로: {PROCESSED_DIR}")
    print(f"Normal filtered PSD 형태: {normal_array.shape}")
    print(f"ISC filtered PSD 형태: {isc_array.shape}")


if __name__ == "__main__":
    main()
