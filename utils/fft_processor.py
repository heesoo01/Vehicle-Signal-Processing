"""Welch PSD conversion utilities for EKF innovation signals."""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .denoise import crop_by_seconds, estimate_sampling_rate


def process_residual_to_psd(
    result_data,
    crop_seconds: int = 500,
    source_col: str = "residual",
    time_col: str = "time",
    nperseg: int = 256,
    dt: float | None = None,
):
    """
    Convert an EKF innovation signal to Welch PSD.

    Parameters
    ----------
    result_data:
        EKF result DataFrame.
    crop_seconds:
        Initial EKF convergence section to remove.
    source_col:
        Column used for PSD. Use "filtered_residual" after denoising.
    time_col:
        Time column used for estimating sampling frequency.
    nperseg:
        Welch segment length. 256 gives 129 PSD bins.
    dt:
        Optional fixed sampling interval kept for backward compatibility.
    """
    if source_col not in result_data.columns:
        raise KeyError(
            f"'{source_col}' 컬럼이 result_data에 없습니다. "
            "denoise 이후 PSD를 만들려면 denoise_innovation()을 먼저 실행하세요."
        )

    if dt is None:
        _, fs = estimate_sampling_rate(result_data, time_col=time_col)
    else:
        fs = 1.0 / float(dt)

    stable = crop_by_seconds(result_data, crop_seconds=crop_seconds, time_col=time_col)
    signal = stable[source_col].to_numpy(dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 2:
        raise ValueError("PSD 계산에 필요한 residual 데이터가 부족합니다.")

    window_length = min(len(signal), nperseg)
    freqs, psd = welch(
        signal,
        fs=fs,
        nperseg=window_length,
        scaling="density",
        detrend=False,
    )
    return freqs, psd
