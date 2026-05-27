import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.fft_processor import process_residual_to_psd

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"

# 타겟 데이터를 1.2CC 단락 데이터(10ohm)로 변경
TARGET_PATH = DATASET_ROOT / "NCM811_ISC_TEST" / "DST" / "ISC_CS_1.2CC_DST_10ohm.csv"

def main():
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc)
    cap = normal_cc["capacity_ah"].max() * 3600.0

    target_data = add_coulomb_counted_soc(load_right_block(TARGET_PATH), cap)
    result_data = run_ekf(target_data, soc_table, ocv_table, cap, initialize_vrc=True)

    freqs, psd = process_residual_to_psd(result_data, crop_seconds=500)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"Pipeline Validation: {TARGET_PATH.name}", fontsize=16, fontweight='bold')

    stable_data = result_data.iloc[500:]
    stable_time = stable_data["time"].to_numpy()

    axes[0].plot(stable_time, stable_data["voltage"], label="Measured Voltage")
    axes[0].plot(stable_time, stable_data["voltage_hat"], label="EKF Estimated")
    axes[0].set_title("Step 1: EKF Voltage (1.2CC)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(stable_time, stable_data["residual"], color='orange', label="Residual")
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title("Step 2: Residual")
    axes[1].grid(True)

    axes[2].plot(freqs, psd, color='purple', label="Welch PSD")
    axes[2].set_xlim(0, 0.1)
    axes[2].set_title("Step 3: Frequency Domain (Model Input)")
    axes[2].grid(True)

    plt.tight_layout()
    save_path = PROJECT_ROOT / "pipeline_test_1.2cc.png"
    plt.savefig(save_path, dpi=300)

if __name__ == "__main__":
    main()