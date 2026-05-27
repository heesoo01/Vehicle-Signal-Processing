import numpy as np
from scipy.signal import welch


def process_residual_to_psd(result_data, crop_seconds=500, dt=1.0):
    """
    [잔차 처리 + FFT 변환 유틸]
    EKF 결과 데이터프레임을 받아 초반 불안정 구간을 잘라내고 PSD로 변환합니다.
    """
    # 1. 잔차 처리: 초반 노이즈 구간(crop_seconds) 잘라내기
    stable_residual = result_data["residual"].iloc[crop_seconds:].to_numpy()

    # 결측치(NaN) 안전 제거
    stable_residual = stable_residual[~np.isnan(stable_residual)]

    # 2. FFT (Welch 메서드)
    fs = 1.0 / dt
    window_length = min(len(stable_residual), 256)

    freqs, psd = welch(
        stable_residual,
        fs=fs,
        nperseg=window_length,
        scaling='density',
        detrend=False
    )

    return freqs, psd