"""
Backward-compatible wrapper.

기존 코드가 utils.fft_processor_modified를 import해도 동일하게 동작하도록
utils.fft_processor.process_residual_to_psd를 다시 export한다.
"""
from .fft_processor import process_residual_to_psd

__all__ = ["process_residual_to_psd"]
