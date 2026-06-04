"""
CC 데이터 기반 EKF Q/R 설계
  - NIS 그리드서치로 Q/R 결정 (설계 기준)
  - MAE / RMSE로 최종 성능 평가 (정량 지표)
"""

import itertools
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    add_dataset_dod_soc,
    build_ocv_table_from_cc,
    load_right_block,
    run_ekf,
)
from utils.ekf_eval import (
    compute_metrics,
    compute_nis_metrics,
    nis_score,
    print_report,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "voltage_prediction_and_ISC_detection-V1.0"
    / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
)
NORMAL_CC_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC"
IMAGE_DIR  = PROJECT_ROOT / "images"
RESULT_DIR = PROJECT_ROOT / "results"

# ECM 고정 파라미터 (R0, R1, C1 은 NCM811 1-RC 모델 대표값)
ECM_FIXED = {"R0": 0.015, "R1": 0.01, "C1": 2400.0, "P0": np.diag([1e-4, 1e-3])}

# NIS 그리드서치 후보
Q_SOC_GRID = [1e-9, 1e-8, 1e-7, 5e-7, 1e-6]
Q_VRC_GRID = [1e-7, 1e-6, 3e-6, 1e-5, 3e-5]
R_GRID     = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]


def run_all_cc(cc_datasets, soc_table, ocv_table, capacity_coulomb, params):
    """모든 CC 파일에 EKF 적용 후 결과 합치기."""
    results = []
    for df in cc_datasets:
        res = run_ekf(
            df, soc_table, ocv_table, capacity_coulomb,
            ekf_params=params, initialize_vrc=True,
        )
        results.append(res)
    return pd.concat(results, ignore_index=True)


def main():
    IMAGE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # ── 1. CC 데이터 로드 ────────────────────────────────────────────────
    cc_paths = sorted(NORMAL_CC_DIR.glob("ISC_BD_0.5CC_0.5CD_*ohm.csv"))
    cc_datasets = []
    for p in cc_paths:
        df = load_right_block(p)
        df = add_dataset_dod_soc(df)
        df["label"] = p.stem
        cc_datasets.append(df)

    capacity_coulomb = max(df["capacity_ah"].max() for df in cc_datasets) * 3600.0
    print(f"CC 파일 {len(cc_datasets)}개  |  용량 {capacity_coulomb/3600:.4f} Ah")

    # ── 2. OCV 테이블 (전체 CC + CV 앵커) ───────────────────────────────
    from utils import extract_cv_ocv_anchor, load_left_block
    cv_anchors = [extract_cv_ocv_anchor(load_left_block(p)) for p in cc_paths]
    soc_table, ocv_table = build_ocv_table_from_cc(cc_datasets, cv_anchors=cv_anchors)
    print(f"OCV 테이블  |  SOC {soc_table[0]:.3f} ~ {soc_table[-1]:.3f}")

    # ── 3. NIS 그리드서치 ────────────────────────────────────────────────
    combos = list(itertools.product(Q_SOC_GRID, Q_VRC_GRID, R_GRID))
    print(f"\nNIS 그리드서치: {len(combos)}개 조합...")

    rows = []
    best_score = float("inf")
    log_step = max(1, len(combos) // 10)

    for i, (q_soc, q_vrc, r) in enumerate(combos):
        params = {
            **ECM_FIXED,
            "Q": np.diag([q_soc, q_vrc]),
            "R": np.array([[r]]),
        }
        combined = run_all_cc(cc_datasets, soc_table, ocv_table, capacity_coulomb, params)
        m  = compute_metrics(combined)
        nm = compute_nis_metrics(combined)
        sc = 100.0 * m["rmse"] + nis_score(nm)

        rows.append({"q_soc": q_soc, "q_vrc": q_vrc, "r": r, **m, **nm, "score": sc})

        if sc < best_score:
            best_score = sc

        if (i + 1) % log_step == 0 or (i + 1) == len(combos):
            best_row = min(rows, key=lambda x: x["score"])
            print(
                f"  [{i+1:3d}/{len(combos)}]  best score={best_row['score']:.5f}"
                f"  (MAE={best_row['mae']*1e3:.2f}mV, NIS={best_row['nis_mean']:.3f})"
            )

    grid_df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    grid_df.to_csv(RESULT_DIR / "cc_qr_grid_search.csv", index=False)

    best = grid_df.iloc[0]
    print(f"\n최적 Q/R:")
    print(f"  q_soc = {best['q_soc']:.3e}")
    print(f"  q_vrc = {best['q_vrc']:.3e}")
    print(f"  r     = {best['r']:.3e}")

    # ── 4. 최적 파라미터로 최종 평가 ────────────────────────────────────
    best_params = {
        **ECM_FIXED,
        "Q": np.diag([best["q_soc"], best["q_vrc"]]),
        "R": np.array([[best["r"]]]),
    }
<<<<<<< HEAD

    # 뒤 단계(eval/prepare/visualize)에서 읽기 쉽게 최적 파라미터만 별도 저장
    pd.DataFrame([{
        "q_soc": best["q_soc"],
        "q_vrc": best["q_vrc"],
        "r": best["r"],
        "score": best["score"],
        "mae": best["mae"],
        "rmse": best["rmse"],
        "nis_mean": best["nis_mean"],
    }]).to_csv(RESULT_DIR / "cc_best_ekf_params.csv", index=False)
=======
>>>>>>> 9ea84ddfa1aff10910ed890d159c97eb1db187df
    combined = run_all_cc(cc_datasets, soc_table, ocv_table, capacity_coulomb, best_params)
    m  = compute_metrics(combined)
    nm = compute_nis_metrics(combined)
    print_report(m, nm, title="CC EKF - 최적 Q/R 적용 결과")

    # ── 5. 플롯: 대표 파일 innovation 시계열 + 히스토그램 ────────────────
    rep = run_ekf(
        cc_datasets[0], soc_table, ocv_table, capacity_coulomb,
        ekf_params=best_params, initialize_vrc=True,
    )
    _plot_innovation(rep, IMAGE_DIR / "cc_ekf_innovation.png")
    print(f"플롯 저장 -> {IMAGE_DIR / 'cc_ekf_innovation.png'}")
    print("완료.")


def _plot_innovation(result_df, save_path):
    time_min = result_df["time"].to_numpy() / 60.0
    innov_mv = result_df["residual"].to_numpy() * 1e3
    nis      = (result_df["residual"].to_numpy() ** 2) / result_df["S"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # 전압 추적
    axes[0].plot(time_min, result_df["voltage"].to_numpy() * 1e3,
                 label="측정 전압", linewidth=1.2)
    axes[0].plot(time_min, result_df["voltage_hat"].to_numpy() * 1e3,
                 label="EKF 예측 전압", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("전압 [mV]")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[0].set_title("CC 데이터 EKF 검증 (최적 Q/R)")

    # Innovation 시계열
    axes[1].plot(time_min, innov_mv, linewidth=0.8, color="#4c78a8")
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_ylabel("Innovation [mV]")
    axes[1].grid(alpha=0.25)

    # NIS 시계열
    axes[2].plot(time_min, nis, linewidth=0.7, color="#4c78a8", alpha=0.8)
    axes[2].axhline(3.84, color="#d62728", linestyle="--", linewidth=1.0,
                    label="chi2(0.95)=3.84")
    axes[2].axhline(1.0, color="green", linestyle="-", linewidth=0.8,
                    alpha=0.6, label="NIS=1 (이상적)")
    axes[2].set_ylim(0, min(float(np.percentile(nis[np.isfinite(nis)], 99)) * 1.5, 15))
    axes[2].set_ylabel("NIS")
    axes[2].set_xlabel("시간 [min]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
