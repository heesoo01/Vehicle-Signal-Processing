import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.denoise import (
    denoise_innovation,
    estimate_sampling_rate,
    crop_by_seconds,
)

# ==============================
# 경로 설정
# ==============================
DATASET_ROOT = (
    PROJECT_ROOT
    / "voltage_prediction_and_ISC_detection-V1.0"
    / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
)

NORMAL_CC_PATH = (
    DATASET_ROOT
    / "NCM811_NORMAL_TEST"
    / "CC"
    / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
)

# 비교할 대상 파일
# 필요하면 10ohm / 100ohm / 1000ohm 바꿔가며 실행하면 됨
TARGET_PATH = (
    DATASET_ROOT
    / "NCM811_ISC_TEST"
    / "DST"
    / "ISC_CS_1.2CC_DST_1000ohm.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ECM_FIXED = {
    "R0": 0.015,
    "R1": 0.01,
    "C1": 2400.0,
    "P0": np.diag([1e-4, 1e-3]),
}


def load_best_ekf_params():
    """
    cc_ekf_design.py가 저장한 최적 Q/R을 읽어 EKF 파라미터로 변환.
    먼저 scripts/cc_ekf_design.py를 실행해 results에 csv가 있어야 함.
    """
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


def compute_psd(result_data, source_col, crop_seconds=500, nperseg=256):
    """
    result_data에서 특정 컬럼의 Welch PSD 계산.
    denoise.py의 fs 추정/crop 방식과 동일하게 맞춤.
    """
    _, fs = estimate_sampling_rate(result_data, time_col="time")
    stable = crop_by_seconds(result_data, crop_seconds=crop_seconds, time_col="time")

    if stable.empty:
        raise ValueError("crop 이후 남은 데이터가 없습니다. crop_seconds를 줄이세요.")

    signal = stable[source_col].to_numpy(dtype=float)
    signal = signal[np.isfinite(signal)]

    seg = min(nperseg, len(signal))
    if seg < 2:
        raise ValueError("PSD 계산에 필요한 데이터가 부족합니다.")

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=seg,
        scaling="density",
        detrend=False,
    )

    return freqs, psd


def main():
    # ==============================
    # 1. OCV(SOC) 기준 생성
    # ==============================
    print("1. OCV(SOC) 기준 생성 중...")
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc])
    cap = normal_cc["capacity_ah"].max() * 3600.0
    ekf_params = load_best_ekf_params()

    # ==============================
    # 2. 대상 DST 데이터에 EKF 적용
    # ==============================
    print(f"\n2. 대상 파일 로드: {TARGET_PATH.name}")
    raw_data = load_right_block(TARGET_PATH)
    target_data = add_coulomb_counted_soc(raw_data, cap)

    print("3. EKF 실행 → residual 생성")
    result_data = run_ekf(
        target_data,
        soc_table,
        ocv_table,
        cap,
        ekf_params=ekf_params,
        initialize_vrc=True,
    )

    # ==============================
    # 3. Raw PSD 계산
    # ==============================
    print("4. Raw residual PSD 계산")
    freqs_raw, psd_raw = compute_psd(
        result_data,
        source_col="residual",
        crop_seconds=500,
        nperseg=256,
    )

    # ==============================
    # 4. cutoff sweep
    # ==============================
    cutoffs = [0.05, 0.10, 0.20, 0.30]

    psd_results = {}

    for cutoff in cutoffs:
        print(f"5. cutoff={cutoff:.2f}Hz denoise 적용 및 PSD 계산")

        filtered_data = denoise_innovation(
            result_data,
            source_col="residual",
            output_col=f"filtered_residual_{cutoff}",
            cutoff_hz=cutoff,
            butter_order=4,
            savgol_window=0,
        )

        freqs_filt, psd_filt = compute_psd(
            filtered_data,
            source_col=f"filtered_residual_{cutoff}",
            crop_seconds=500,
            nperseg=256,
        )

        psd_results[cutoff] = (freqs_filt, psd_filt)

    # ==============================
    # 5. PSD 비교 그래프 저장
    # ==============================
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogy(
        freqs_raw,
        psd_raw + 1e-18,
        label="Raw residual PSD",
        linewidth=1.5,
        alpha=0.9,
    )

    for cutoff, (freqs_filt, psd_filt) in psd_results.items():
        ax.semilogy(
            freqs_filt,
            psd_filt + 1e-18,
            label=f"Filtered PSD cutoff={cutoff:.2f}Hz",
            linewidth=1.2,
        )

    ax.set_xlim(0.0, 0.30)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [V²/Hz]")
    ax.set_title(f"Cutoff Sweep PSD Comparison: {TARGET_PATH.name}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    save_path = RESULTS_DIR / f"denoise_cutoff_sweep_{TARGET_PATH.stem}.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"\nPSD 비교 그래프 저장 완료: {save_path}")

    # ==============================
    # 6. 특정 대역 에너지 요약 저장
    # ==============================
    bands = {
        "low_0.00_0.05Hz": (0.00, 0.05),
        "isc_candidate_0.10_0.20Hz": (0.10, 0.20),
        "high_0.20_0.30Hz": (0.20, 0.30),
    }

    rows = []

    def band_energy(freqs, psd, fmin, fmax):
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return np.nan
        return float(np.trapezoid(psd[mask], freqs[mask]))

    for band_name, (fmin, fmax) in bands.items():
        rows.append({
            "case": "raw",
            "cutoff_hz": "none",
            "band": band_name,
            "energy": band_energy(freqs_raw, psd_raw, fmin, fmax),
        })

        for cutoff, (freqs_filt, psd_filt) in psd_results.items():
            rows.append({
                "case": "filtered",
                "cutoff_hz": cutoff,
                "band": band_name,
                "energy": band_energy(freqs_filt, psd_filt, fmin, fmax),
            })

    summary_df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / f"denoise_cutoff_sweep_{TARGET_PATH.stem}.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"대역별 PSD 에너지 요약 저장 완료: {csv_path}")


if __name__ == "__main__":
    main()