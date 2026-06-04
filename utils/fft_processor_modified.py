<<<<<<< HEAD
"""
Backward-compatible wrapper.

기존 코드가 utils.fft_processor_modified를 import해도 동일하게 동작하도록
utils.fft_processor.process_residual_to_psd를 다시 export한다.
"""
from .fft_processor import process_residual_to_psd

__all__ = ["process_residual_to_psd"]
=======
import numpy as np
from scipy.signal import welch

from .denoise import crop_by_seconds, estimate_sampling_rate


def process_residual_to_psd(
    result_data,
    crop_seconds: int = 500,
    source_col: str = "residual",
    time_col: str = "time",
    nperseg: int = 256,
):
    """
    EKF 결과 DataFrame을 받아 innovation을 Welch PSD로 변환한다.

    - crop_seconds: 초기 EKF 수렴 구간을 '초 단위'로 제거한다.
    - source_col 기본값은 "residual"이며, denoise 이후에는 "filtered_residual"로 호출하면 된다.
    - 샘플링 주파수(fs)는 time 컬럼에서 자동 추정한다.
    """
    if source_col not in result_data.columns:
        raise KeyError(f"'{source_col}' 컬럼이 result_data에 없습니다.")

    _, fs = estimate_sampling_rate(result_data, time_col=time_col)
    stable = crop_by_seconds(result_data, crop_seconds=crop_seconds, time_col=time_col)

    signal = stable[source_col].to_numpy(dtype=float)
    signal = signal[np.isfinite(signal)]

    window_length = min(len(signal), nperseg)

    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=window_length,
        scaling="density",
        detrend=False,
    )

    return freqs, psd
>>>>>>> 9ea84ddfa1aff10910ed890d159c97eb1db187df
