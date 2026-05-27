import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
# 타겟 데이터를 1.2CC 단락 데이터(10ohm)로 변경
TARGET_PATH = DATASET_ROOT / "NCM811_ISC_TEST" / "DST" / "ISC_CS_1.2CC_DST_10ohm.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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


def main():
    normal_cc = load_right_block(NORMAL_CC_PATH)

    # build_ocv_table_from_cc()는 DataFrame 리스트를 기대한다.
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc])
    cap = normal_cc["capacity_ah"].max() * 3600.0
    ekf_params = load_best_ekf_params()

    target_data = add_coulomb_counted_soc(load_right_block(TARGET_PATH), cap)

    # 1. EKF 실행: residual 생성
    result_data = run_ekf(
        target_data,
        soc_table,
        ocv_table,
        cap,
        ekf_params=ekf_params,
        initialize_vrc=True,
    )

    # 2. denoise 적용: filtered_residual 생성
    result_data = denoise_innovation(
        result_data,
        source_col="residual",
        output_col="filtered_residual",
        cutoff_hz=0.05,
    )

   # 3. Raw residual PSD와 filtered_residual PSD 둘 다 생성
    freqs_raw, psd_raw = process_residual_to_psd(
        result_data,
        crop_seconds=500,
        source_col="residual",
        nperseg=256,
    )

    freqs_filt, psd_filt = process_residual_to_psd(
        result_data,
         crop_seconds=500,
         source_col="filtered_residual",
        nperseg=256,
    )

    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f"Pipeline Validation: {TARGET_PATH.name}", fontsize=16, fontweight="bold")

    stable_data = result_data.iloc[500:]
    stable_time = stable_data["time"].to_numpy()

    axes[0].plot(stable_time, stable_data["voltage"], label="Measured Voltage")
    axes[0].plot(stable_time, stable_data["voltage_hat"], label="EKF Estimated")
    axes[0].set_title("Step 1: EKF Voltage")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(stable_time, stable_data["residual"], label="Raw residual")
    axes[1].axhline(0, linestyle="--")
    axes[1].set_title("Step 2: Raw EKF Innovation")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(stable_time, stable_data["filtered_residual"], label="Filtered residual")
    axes[2].axhline(0, linestyle="--")
    axes[2].set_title("Step 3: Denoised Innovation")
    axes[2].legend()
    axes[2].grid(True)

    axes[3].semilogy(freqs_raw, psd_raw + 1e-18, label="Raw residual PSD")
    axes[3].semilogy(freqs_filt, psd_filt + 1e-18, label="Filtered residual PSD")
    axes[3].set_xlim(0, 0.1)
    axes[3].set_title("Step 4: Welch PSD Comparison")
    axes[3].set_xlabel("Frequency [Hz]")
    axes[3].set_ylabel("PSD [V²/Hz]")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    save_path = RESULTS_DIR / "pipeline_test_1.2cc_raw_vs_filtered_psd.png"
    plt.savefig(save_path, dpi=300)
    print(f"저장 완료: {save_path}")


if __name__ == "__main__":
    main()
