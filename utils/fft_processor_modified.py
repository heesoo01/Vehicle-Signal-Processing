import numpy as np
from scipy.signal import welch


def process_residual_to_psd(result_data, crop_seconds=500, dt=1.0, source_col="residual"):
    """
    EKF 결과 DataFrame을 받아 innovation을 Welch PSD로 변환한다.

    source_col 기본값은 기존 코드와 호환되도록 "residual"이며,
    2단계 denoising 이후에는 source_col="filtered_residual"로 호출하면 된다.
    """
    if source_col not in result_data.columns:
        raise KeyError(f"'{source_col}' 컬럼이 result_data에 없습니다.")

    stable_signal = result_data[source_col].iloc[crop_seconds:].to_numpy(dtype=float)
    stable_signal = stable_signal[np.isfinite(stable_signal)]

    fs = 1.0 / dt
    window_length = min(len(stable_signal), 256)

    freqs, psd = welch(
        stable_signal,
        fs=fs,
        nperseg=window_length,
        scaling="density",
        detrend=False,
    )

    return freqs, psd
