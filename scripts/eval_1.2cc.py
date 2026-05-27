import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 유틸리티 임포트 (다이렉트 처리를 위함)
from utils.data import load_right_block, add_coulomb_counted_soc
from utils.ocv import build_ocv_table_from_cc
from utils.pipeline import run_ekf
from utils.fft_processor import process_residual_to_psd
from utils.autoencoder import PSDAutoencoder

DATASET_ROOT = PROJECT_ROOT / "voltage_prediction_and_ISC_detection-V1.0" / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
ISC_DST_DIR = DATASET_ROOT / "NCM811_ISC_TEST" / "DST"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"


def get_cropped_psd(file_path, soc_table, ocv_table, cap):
    """EKF를 돌리고 데이터를 8500초까지만 강제로 잘라서 PSD를 구하는 함수"""
    raw_data = load_right_block(file_path)
    target_data = add_coulomb_counted_soc(raw_data, cap)

    # EKF 실행
    result_data = run_ekf(target_data, soc_table, ocv_table, cap, initialize_vrc=True)

    # ★ 핵심 물리적 수정: 가장 먼저 죽는 10ohm 배터리(8900초) 기준에 맞춰,
    # 모든 배터리의 데이터를 8500초(행)까지만 자름 (Apples-to-Apples 비교)
    cropped_result = result_data.iloc[:8500]

    # 500초(초기화 구간) ~ 8500초 구간에 대해서만 PSD 계산 (총 8000초 데이터)
    freqs, psd = process_residual_to_psd(cropped_result, crop_seconds=500)
    return psd


def main():
    seed = 41
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("1. 기본 OCV Table 생성 중...")
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc(normal_cc)
    cap = normal_cc["capacity_ah"].max() * 3600.0

    print("\n2. 1.2CC 파일 리스트 수집 중...")
    normal_files = sorted(list(NORMAL_DST_DIR.glob("*1.2CC*.csv")))
    isc_files = sorted(list(ISC_DST_DIR.glob("*1.2CC*.csv")))

    train_psd_list = []
    test_psd_dict = {}

    # 학습용 정상 파일 추출 및 다이렉트 PSD 변환
    print(f"3. 정상 데이터(BD) {len(normal_files) - 1}개 타임 윈도우(8000초) 강제 적용 및 변환 중...")
    for p in normal_files:
        if "ISC_BD" in p.name:
            psd = get_cropped_psd(p, soc_table, ocv_table, cap)
            if "1000ohm" in p.name:
                test_psd_dict['Normal'] = psd
            else:
                train_psd_list.append(psd)

    # 테스트용 고장 파일 추출 및 다이렉트 PSD 변환
    print("4. 고장 데이터(CS) 3개 타임 윈도우 강제 적용 및 변환 중...")
    for p in isc_files:
        if "ISC_CS" in p.name:
            psd = get_cropped_psd(p, soc_table, ocv_table, cap)
            if "10ohm" in p.name:
                test_psd_dict['ISC_10'] = psd
            elif "100ohm" in p.name:
                test_psd_dict['ISC_100'] = psd
            elif "1000ohm" in p.name:
                test_psd_dict['ISC_1000'] = psd

    print("\n5. 모델 학습 준비 (Data Leakage 방지)...")
    train_data = np.array(train_psd_list)
    train_log = 10 * np.log10(train_data + 1e-12)
    train_min, train_max = train_log.min(axis=0), train_log.max(axis=0)

    def preprocess(data):
        scaled = (10 * np.log10(data + 1e-12) - train_min) / (train_max - train_min + 1e-12)
        return torch.tensor(scaled, dtype=torch.float32)

    X_train = preprocess(train_data)
    X_test_normal = preprocess(test_psd_dict['Normal']).unsqueeze(0)
    X_test_isc10 = preprocess(test_psd_dict['ISC_10']).unsqueeze(0)
    X_test_isc100 = preprocess(test_psd_dict['ISC_100']).unsqueeze(0)
    X_test_isc1000 = preprocess(test_psd_dict['ISC_1000']).unsqueeze(0)

    model = PSDAutoencoder(input_dim=129, latent_dim=4)
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("6. Autoencoder 모델 학습 중 (Epoch: 150)...")
    for epoch in range(150):
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

    print(f"\n--- 최종 오차 및 탐지 결과 ---")
    print(f"Train Mean: {train_mean:.6f} | Train Std: {train_std:.6f}")
    print(f"👉 설정된 Threshold: {threshold:.6f}\n")

    def evaluate(tensor):
        with torch.no_grad():
            recon = model(tensor)
            mse = criterion(recon, tensor).mean().item()
        return recon[0].numpy(), mse

    recon_norm, mse_norm = evaluate(X_test_normal)
    recon_isc10, mse_isc10 = evaluate(X_test_isc10)
    recon_isc100, mse_isc100 = evaluate(X_test_isc100)
    recon_isc1000, mse_isc1000 = evaluate(X_test_isc1000)

    print(f"[Normal 1000ohm BD] MSE: {mse_norm:.6f} | Anomaly: {mse_norm > threshold}")
    print(f"[ISC 10ohm CS]      MSE: {mse_isc10:.6f} | Anomaly: {mse_isc10 > threshold}")
    print(f"[ISC 100ohm CS]     MSE: {mse_isc100:.6f} | Anomaly: {mse_isc100 > threshold}")
    print(f"[ISC 1000ohm CS]    MSE: {mse_isc1000:.6f} | Anomaly: {mse_isc1000 > threshold}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_axis = np.arange(129)
    cases = [
        (axes[0, 0], X_test_normal[0].numpy(), recon_norm, mse_norm, "Normal (1000ohm BD)"),
        (axes[0, 1], X_test_isc10[0].numpy(), recon_isc10, mse_isc10, "ISC (10ohm CS)"),
        (axes[1, 0], X_test_isc100[0].numpy(), recon_isc100, mse_isc100, "ISC (100ohm CS)"),
        (axes[1, 1], X_test_isc1000[0].numpy(), recon_isc1000, mse_isc1000, "ISC (1000ohm CS)")
    ]

    for ax, orig, recon, mse, title in cases:
        ax.plot(x_axis, orig, color='blue', label='Input (Target)')
        ax.plot(x_axis, recon, color='orange', linestyle='--', label='Reconstruction')
        is_anomaly = mse > threshold
        ax.set_title(f"{title} | MSE: {mse:.6f} | {'ANOMALY' if is_anomaly else 'NORMAL'}",
                     color='red' if is_anomaly else 'green', fontweight='bold')
        ax.set_ylim(-0.2, 1.2)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "inference_1.2cc.png", dpi=300)


if __name__ == "__main__":
    main()