import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.denoise import denoise_innovation
from utils.fft_processor import process_residual_to_psd
from utils.autoencoder import PSDAutoencoder

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
ISC_DST_DIR = DATASET_ROOT / "NCM811_ISC_TEST" / "DST"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"

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


def get_cropped_psd(file_path: Path, soc_table, ocv_table, cap: float, ekf_params):
    """
    CSV -> EKF -> denoise -> cropped filtered_residual PSD.

    이전 merge 문제는 여기서 raw residual을 바로 PSD로 넘긴 점이었다.
    이제 Autoencoder 입력은 항상 filtered_residual의 Welch PSD가 된다.
    """
    raw_data = load_right_block(file_path)
    target_data = add_coulomb_counted_soc(raw_data, cap)

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

    # 3. 가장 먼저 죽는 10ohm 배터리 기준에 맞춰 8500초/행까지만 사용
    cropped_result = result_data.iloc[:8500]

    # 4. filtered_residual 기준 PSD 생성
    freqs, psd = process_residual_to_psd(
        cropped_result,
        crop_seconds=500,
        source_col="filtered_residual",
        nperseg=256,
    )
    return psd


def minmax_preprocess(data, train_min, train_max):
    scaled = (10 * np.log10(data + 1e-12) - train_min) / (train_max - train_min + 1e-12)
    return torch.tensor(scaled, dtype=torch.float32)


def evaluate(model, criterion, tensor):
    with torch.no_grad():
        recon = model(tensor)
        mse = criterion(recon, tensor).mean().item()
    return recon[0].numpy(), mse


def main():
    seed = 41
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("1. 기본 OCV Table 생성 중...")
    normal_cc = load_right_block(NORMAL_CC_PATH)

    # build_ocv_table_from_cc()는 DataFrame 리스트를 기대한다.
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc])
    cap = normal_cc["capacity_ah"].max() * 3600.0
    ekf_params = load_best_ekf_params()

    print("\n2. 1.2CC 파일 리스트 수집 중...")
    normal_files = sorted(NORMAL_DST_DIR.glob("*1.2CC*.csv"))
    isc_files = sorted(ISC_DST_DIR.glob("*1.2CC*.csv"))

    train_psd_list = []
    test_psd_dict = {}

    print(f"3. 정상 데이터(BD) {len(normal_files)}개 변환 중: EKF -> denoise -> PSD")
    for p in normal_files:
        if "ISC_BD" not in p.name:
            continue

        psd = get_cropped_psd(p, soc_table, ocv_table, cap, ekf_params)
        if "1000ohm" in p.name:
            test_psd_dict["Normal"] = psd
        else:
            train_psd_list.append(psd)

    print("4. 고장 데이터(CS) 변환 중: EKF -> denoise -> PSD")
    for p in isc_files:
        if "ISC_CS" not in p.name:
            continue

        psd = get_cropped_psd(p, soc_table, ocv_table, cap, ekf_params)
        if "10ohm" in p.name:
            test_psd_dict["ISC_10"] = psd
        elif "100ohm" in p.name:
            test_psd_dict["ISC_100"] = psd
        elif "1000ohm" in p.name:
            test_psd_dict["ISC_1000"] = psd

    required_keys = ["Normal", "ISC_10", "ISC_100", "ISC_1000"]
    missing = [k for k in required_keys if k not in test_psd_dict]
    if missing:
        raise FileNotFoundError(f"테스트 PSD를 만들지 못했습니다. 누락: {missing}")
    if len(train_psd_list) == 0:
        raise ValueError("학습용 정상 PSD가 없습니다. NORMAL_DST_DIR의 파일명을 확인하세요.")

    print("\n5. 모델 학습 준비 (Data Leakage 방지: train 기준 min/max만 사용)")
    train_data = np.array(train_psd_list)
    train_log = 10 * np.log10(train_data + 1e-12)
    train_min, train_max = train_log.min(axis=0), train_log.max(axis=0)

    X_train = minmax_preprocess(train_data, train_min, train_max)
    X_test_normal = minmax_preprocess(test_psd_dict["Normal"], train_min, train_max).unsqueeze(0)
    X_test_isc10 = minmax_preprocess(test_psd_dict["ISC_10"], train_min, train_max).unsqueeze(0)
    X_test_isc100 = minmax_preprocess(test_psd_dict["ISC_100"], train_min, train_max).unsqueeze(0)
    X_test_isc1000 = minmax_preprocess(test_psd_dict["ISC_1000"], train_min, train_max).unsqueeze(0)

    input_dim = X_train.shape[1]
    model = PSDAutoencoder(input_dim=input_dim, latent_dim=4)
    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("6. Autoencoder 모델 학습 중 (Epoch: 150)")
    model.train()
    for _ in range(150):
        optimizer.zero_grad()
        loss = criterion(model(X_train), X_train).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_mses = criterion(model(X_train), X_train).mean(dim=1)
        train_mean = train_mses.mean().item()
        train_std = train_mses.std().item()
        threshold = train_mean + (3 * train_std)

    print("\n--- 최종 오차 및 탐지 결과 ---")
    print(f"Train Mean: {train_mean:.6f} | Train Std: {train_std:.6f}")
    print(f"설정된 Threshold: {threshold:.6f}\n")

    recon_norm, mse_norm = evaluate(model, criterion, X_test_normal)
    recon_isc10, mse_isc10 = evaluate(model, criterion, X_test_isc10)
    recon_isc100, mse_isc100 = evaluate(model, criterion, X_test_isc100)
    recon_isc1000, mse_isc1000 = evaluate(model, criterion, X_test_isc1000)

    result_rows = [
        {"case": "Normal_1000ohm_BD", "mse": mse_norm, "threshold": threshold, "anomaly": mse_norm > threshold},
        {"case": "ISC_10ohm_CS", "mse": mse_isc10, "threshold": threshold, "anomaly": mse_isc10 > threshold},
        {"case": "ISC_100ohm_CS", "mse": mse_isc100, "threshold": threshold, "anomaly": mse_isc100 > threshold},
        {"case": "ISC_1000ohm_CS", "mse": mse_isc1000, "threshold": threshold, "anomaly": mse_isc1000 > threshold},
    ]
    pd.DataFrame(result_rows).to_csv(RESULTS_DIR / "autoencoder_1.2cc_denoised_metrics.csv", index=False)

    print(f"[Normal 1000ohm BD] MSE: {mse_norm:.6f} | Anomaly: {mse_norm > threshold}")
    print(f"[ISC 10ohm CS] MSE: {mse_isc10:.6f} | Anomaly: {mse_isc10 > threshold}")
    print(f"[ISC 100ohm CS] MSE: {mse_isc100:.6f} | Anomaly: {mse_isc100 > threshold}")
    print(f"[ISC 1000ohm CS] MSE: {mse_isc1000:.6f} | Anomaly: {mse_isc1000 > threshold}")
    print(f"결과 CSV 저장 완료: {RESULTS_DIR / 'autoencoder_1.2cc_denoised_metrics.csv'}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_axis = np.arange(input_dim)
    cases = [
        (axes[0, 0], X_test_normal[0].numpy(), recon_norm, mse_norm, "Normal (1000ohm BD)"),
        (axes[0, 1], X_test_isc10[0].numpy(), recon_isc10, mse_isc10, "ISC (10ohm CS)"),
        (axes[1, 0], X_test_isc100[0].numpy(), recon_isc100, mse_isc100, "ISC (100ohm CS)"),
        (axes[1, 1], X_test_isc1000[0].numpy(), recon_isc1000, mse_isc1000, "ISC (1000ohm CS)"),
    ]

    for ax, orig, recon, mse, title in cases:
        ax.plot(x_axis, orig, label="Input PSD")
        ax.plot(x_axis, recon, linestyle="--", label="Reconstruction")
        is_anomaly = mse > threshold
        ax.set_title(f"{title} | MSE: {mse:.6f} | {'ANOMALY' if is_anomaly else 'NORMAL'}", fontweight="bold")
        ax.set_ylim(-0.2, 1.2)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    save_path = RESULTS_DIR / "inference_1.2cc_denoised.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n그래프 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
