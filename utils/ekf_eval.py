import numpy as np

NIS_THRESHOLD_95 = 3.84
NIS_THRESHOLD_99 = 6.63


def compute_metrics(result_df):
    """Innovation MAE, RMSE, bias."""
    r = result_df["residual"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return {
        "mae":  float(np.mean(np.abs(r))),
        "rmse": float(np.sqrt(np.mean(r ** 2))),
        "bias": float(np.mean(r)),
    }


def compute_nis_metrics(result_df):
    """NIS 통계 및 chi-square 초과 비율 (S 컬럼 필요)."""
    r = result_df["residual"].to_numpy(dtype=float)
    S = result_df["S"].to_numpy(dtype=float)
    mask = np.isfinite(r) & np.isfinite(S) & (S > 0)
    nis = (r[mask] ** 2) / S[mask]
    return {
        "nis_mean":         float(np.mean(nis)),
        "nis_median":       float(np.median(nis)),
        "nis_exceed_3_84":  float(np.mean(nis > NIS_THRESHOLD_95)),
        "nis_exceed_6_63":  float(np.mean(nis > NIS_THRESHOLD_99)),
        "n_samples":        int(len(nis)),
    }


def nis_score(nis_metrics):
    """NIS 일관성 스코어 (낮을수록 좋음). 설계 기준."""
    return (
        abs(nis_metrics["nis_mean"] - 1.0)
        + 0.5 * nis_metrics["nis_exceed_3_84"]
        + 1.0 * nis_metrics["nis_exceed_6_63"]
    )


def print_report(metrics, nis_metrics, title="EKF"):
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  Innovation MAE  : {metrics['mae']*1e3:7.3f} mV")
    print(f"  Innovation RMSE : {metrics['rmse']*1e3:7.3f} mV")
    print(f"  Innovation Bias : {metrics['bias']*1e3:7.3f} mV")
    print(f"  {'-'*46}")
    print(f"  NIS Mean        : {nis_metrics['nis_mean']:7.4f}  (target ~= 1)")
    print(f"  NIS Median      : {nis_metrics['nis_median']:7.4f}")
    print(f"  NIS > 3.84      : {nis_metrics['nis_exceed_3_84']:7.4f}  (ideal ~= 0.05)")
    print(f"  NIS > 6.63      : {nis_metrics['nis_exceed_6_63']:7.4f}  (ideal ~= 0.01)")
    print(f"  N samples       : {nis_metrics['n_samples']:7d}")
    print(f"{sep}\n")
