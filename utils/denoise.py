"""
EKF innovation(residual) denoising utilities.

Purpose
-------
This module does not classify ISC by itself. It only creates a cleaner
`filtered_residual` signal that can be used by the PSD/Autoencoder pipeline.

Pipeline:
    residual -> NaN/Inf interpolation -> Hampel despike
             -> zero-phase Butterworth low-pass -> optional Savitzky-Golay
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfiltfilt, welch


def estimate_sampling_rate(result_data: pd.DataFrame, time_col: str = "time") -> Tuple[float, float]:
    """Estimate representative dt and sampling frequency from the time column."""
    if time_col not in result_data.columns:
        return 1.0, 1.0

    t = result_data[time_col].to_numpy(dtype=float)
    dt_values = np.diff(t)
    dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0)]

    if len(dt_values) == 0:
        return 1.0, 1.0

    dt = float(np.median(dt_values))
    fs = 1.0 / dt
    return dt, fs


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values by linear interpolation."""
    x = np.asarray(x, dtype=float)
    s = pd.Series(x).replace([np.inf, -np.inf], np.nan)
    s = s.interpolate(method="linear", limit_direction="both")
    return s.to_numpy(dtype=float)


def hampel_filter(x: np.ndarray, window_size: int = 9, n_sigmas: float = 3.0) -> np.ndarray:
    """Hampel filter for spike/outlier mitigation."""
    x = _interpolate_nans(x)

    if window_size < 3:
        return x.copy()
    if window_size % 2 == 0:
        window_size += 1

    s = pd.Series(x)
    rolling_median = s.rolling(window_size, center=True, min_periods=1).median()
    diff = np.abs(s - rolling_median)
    mad = diff.rolling(window_size, center=True, min_periods=1).median()

    threshold = n_sigmas * 1.4826 * mad.replace(0, np.nan)
    outlier = diff > threshold

    y = s.copy()
    y[outlier.fillna(False)] = rolling_median[outlier.fillna(False)]
    return y.to_numpy(dtype=float)


def lowpass_butterworth(
    x: np.ndarray,
    fs: float,
    cutoff_hz: Optional[float] = 0.05,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    x = _interpolate_nans(x)

    if fs <= 0 or len(x) < 8:
        return x.copy()

    nyquist = 0.5 * fs
    if cutoff_hz is None:
        cutoff_hz = min(0.05, nyquist * 0.2)

    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0:
        return x.copy()
    if cutoff_hz >= nyquist:
        cutoff_hz = nyquist * 0.8

    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")

    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        # If the sequence is too short for filtfilt padding, keep original signal.
        return x.copy()


def optional_savgol_smoothing(x: np.ndarray, window_length: int = 0, polyorder: int = 2) -> np.ndarray:
    """Optional weak smoothing. window_length=0 disables this step."""
    x = _interpolate_nans(x)

    if window_length is None or window_length <= 0:
        return x.copy()
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder + 2 or len(x) < window_length:
        return x.copy()

    return savgol_filter(x, window_length=window_length, polyorder=polyorder)


def denoise_innovation(
    result_data: pd.DataFrame,
    source_col: str = "residual",
    output_col: str = "filtered_residual",
    time_col: str = "time",
    hampel_window: int = 9,
    hampel_sigmas: float = 3.0,
    cutoff_hz: Optional[float] = 0.05,
    butter_order: int = 4,
    savgol_window: int = 0,
    savgol_polyorder: int = 2,
) -> pd.DataFrame:
    """Create `filtered_residual` from EKF innovation/residual."""
    if source_col not in result_data.columns:
        raise KeyError(f"'{source_col}' 컬럼이 result_data에 없습니다.")

    out = result_data.copy()
    _, fs = estimate_sampling_rate(out, time_col=time_col)

    raw = out[source_col].to_numpy(dtype=float)
    clean = _interpolate_nans(raw)
    despiked = hampel_filter(clean, window_size=hampel_window, n_sigmas=hampel_sigmas)
    filtered = lowpass_butterworth(despiked, fs=fs, cutoff_hz=cutoff_hz, order=butter_order)
    filtered = optional_savgol_smoothing(
        filtered,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )

    out[output_col] = filtered
    return out


def crop_by_seconds(df: pd.DataFrame, crop_seconds: int, time_col: str = "time") -> pd.DataFrame:
    """Remove the initial EKF convergence period by seconds, falling back to rows."""
    if crop_seconds <= 0:
        return df

    if time_col in df.columns and len(df) > 0:
        t0 = float(df[time_col].iloc[0])
        return df[df[time_col] >= t0 + crop_seconds]

    return df.iloc[crop_seconds:]


def plot_denoising_result(
    result_data: pd.DataFrame,
    output_path: str | Path,
    raw_col: str = "residual",
    filtered_col: str = "filtered_residual",
    time_col: str = "time",
    crop_seconds: int = 500,
    nperseg: int = 256,
    freq_xlim: Tuple[float, float] = (0.0, 0.1),
) -> None:
    """Save a time-domain and PSD comparison plot for raw vs filtered residual."""
    if raw_col not in result_data.columns:
        raise KeyError(f"'{raw_col}' 컬럼이 없습니다.")
    if filtered_col not in result_data.columns:
        raise KeyError(f"'{filtered_col}' 컬럼이 없습니다. denoise_innovation()을 먼저 실행하세요.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dt, fs = estimate_sampling_rate(result_data, time_col=time_col)
    stable = crop_by_seconds(result_data, crop_seconds=crop_seconds, time_col=time_col)

    if stable.empty:
        raise ValueError("crop 이후 남은 데이터가 없습니다. crop_seconds를 줄이세요.")

    t = stable[time_col].to_numpy(dtype=float) if time_col in stable.columns else np.arange(len(stable)) * dt
    raw = _interpolate_nans(stable[raw_col].to_numpy(dtype=float))
    filt = _interpolate_nans(stable[filtered_col].to_numpy(dtype=float))

    seg = min(nperseg, len(raw), len(filt))
    if seg < 2:
        raise ValueError("PSD 계산에 필요한 데이터가 부족합니다.")

    freqs_raw, psd_raw = welch(raw, fs=fs, nperseg=seg, scaling="density", detrend=False)
    freqs_flt, psd_flt = welch(filt, fs=fs, nperseg=seg, scaling="density", detrend=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(t, raw * 1e3, label="Raw innovation", linewidth=0.8, alpha=0.75)
    axes[0].plot(t, filt * 1e3, label="Filtered innovation", linewidth=1.2)
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_title("Time Domain: Raw vs Filtered Innovation")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Innovation [mV]")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(freqs_raw, psd_raw + 1e-18, label="Raw PSD", linewidth=1.0, alpha=0.75)
    axes[1].semilogy(freqs_flt, psd_flt + 1e-18, label="Filtered PSD", linewidth=1.2)
    axes[1].set_xlim(*freq_xlim)
    axes[1].set_title("Frequency Domain: Raw PSD vs Filtered PSD")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("PSD [V²/Hz]")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
