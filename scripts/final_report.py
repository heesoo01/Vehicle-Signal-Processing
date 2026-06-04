"""
기말 프로젝트 최종 보고서 시각화
  - 목표 1: 적정 노이즈 선정 및 강건성 검증 (std=0.01 vs 0.05/0.10)
  - 목표 2: Offline(Zero-phase Butterworth) / Online(Welch 직접) 파이프라인 비교
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
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

# ── 경로 ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = (
    PROJECT_ROOT
    / "voltage_prediction_and_ISC_detection-V1.0"
    / "swhlqu-voltage_prediction_and_ISC_detection-dd56682"
)
NORMAL_DST_DIR = DATASET_ROOT / "NCM811_NORMAL_TEST" / "DST"
ISC_DST_DIR    = DATASET_ROOT / "NCM811_ISC_TEST"    / "DST"
NORMAL_CC_PATH = DATASET_ROOT / "NCM811_NORMAL_TEST" / "CC" / "ISC_BD_0.5CC_0.5CD_1000ohm.csv"
RESULTS_DIR    = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ECM_FIXED = {"R0": 0.015, "R1": 0.01, "C1": 2400.0, "P0": np.diag([1e-4, 1e-3])}

# 비교할 노이즈 수준 (단위: V)
NOISE_STDS  = [0.0, 0.01, 0.05, 0.10]
NOISE_NAMES_SHORT = ["No Noise", "std=0.01V (10mV, BMS)", "std=0.05V (50mV)", "std=0.10V (100mV)"]
SEED = 42


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def load_best_ekf_params():
    best_path = RESULTS_DIR / "cc_best_ekf_params.csv"
    grid_path = RESULTS_DIR / "cc_qr_grid_search.csv"
    if best_path.exists():
        best = pd.read_csv(best_path).iloc[0]
    elif grid_path.exists():
        best = pd.read_csv(grid_path).sort_values("score").iloc[0]
    else:
        raise FileNotFoundError("최적 Q/R 파일 없음. cc_ekf_design.py를 먼저 실행하세요.")
    return {
        **ECM_FIXED,
        "Q": np.diag([float(best["q_soc"]), float(best["q_vrc"])]),
        "R": np.array([[float(best["r"])]]),
    }


def inject_noise(data: pd.DataFrame, std: float) -> pd.DataFrame:
    """전압 컬럼에 AWGN 주입 (재현성을 위해 seed 고정)."""
    if std <= 0:
        return data.copy()
    rng = np.random.default_rng(SEED)
    d = data.copy()
    d["voltage"] = d["voltage"] + rng.normal(0, std, len(d))
    return d


def run_full_pipeline(file_path, soc_table, ocv_table, cap, ekf_params,
                      noise_std=0.0, offline=True, crop_rows=8500, crop_seconds=500):
    """
    파일 -> 노이즈 주입 -> EKF -> (offline=True이면 Butterworth LPF) -> Welch PSD.

    Returns: (freqs, psd, result_df)
    """
    raw    = load_right_block(file_path)
    data   = add_coulomb_counted_soc(raw, cap)
    data   = inject_noise(data, noise_std)
    result = run_ekf(data, soc_table, ocv_table, cap,
                     ekf_params=ekf_params, initialize_vrc=True)

    if offline:
        result = denoise_innovation(result, source_col="residual",
                                    output_col="filtered_residual", cutoff_hz=0.05)
        src = "filtered_residual"
    else:
        src = "residual"

    cropped = result.iloc[:crop_rows]
    freqs, psd = process_residual_to_psd(
        cropped, crop_seconds=crop_seconds, source_col=src, nperseg=256
    )
    return freqs, psd, result


def minmax_preprocess(data_arr, tmin, tmax):
    s = (10 * np.log10(np.asarray(data_arr) + 1e-12) - tmin) / (tmax - tmin + 1e-12)
    return torch.tensor(s, dtype=torch.float32)


def train_and_eval(train_psds, test_psd_dict):
    """Autoencoder 학습 후 테스트 케이스별 MSE, threshold, 탐지 결과 반환."""
    torch.manual_seed(SEED)
    arr  = np.array(train_psds)
    logv = 10 * np.log10(arr + 1e-12)
    tmin, tmax = logv.min(axis=0), logv.max(axis=0)

    X_train = minmax_preprocess(arr, tmin, tmax)
    model = PSDAutoencoder(input_dim=X_train.shape[1], latent_dim=4)
    crit  = nn.MSELoss(reduction="none")
    opt   = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in range(150):
        opt.zero_grad()
        crit(model(X_train), X_train).mean().backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        ms = crit(model(X_train), X_train).mean(dim=1)
        threshold = ms.mean().item() + 3 * ms.std().item()

    results = {}
    for key, psd in test_psd_dict.items():
        x = minmax_preprocess(psd, tmin, tmax).unsqueeze(0)
        with torch.no_grad():
            mse = crit(model(x), x).mean().item()
        results[key] = {"mse": mse, "threshold": threshold, "detected": mse > threshold}

    return results, threshold


# ── 표 이미지 저장 함수 ───────────────────────────────────────────────────────
def _save_noise_table_image(df: pd.DataFrame, out_path):
    """
    Table 1: 노이즈 수준별 탐지 결과를 피벗해서 이미지로 저장.
    행=케이스, 열=노이즈 수준, 셀=MSE/임계값 비율 + 탐지여부
    """
    cases  = df["case"].unique().tolist()
    levels = df["noise_level"].unique().tolist()

    col_labels = levels
    row_labels = cases

    cell_text   = []
    cell_colors = []

    for case in cases:
        row_text   = []
        row_colors = []
        for level in levels:
            sub = df[(df["case"] == case) & (df["noise_level"] == level)]
            if sub.empty:
                row_text.append("-")
                row_colors.append("#FFFFFF")
                continue
            mse   = sub["mse"].values[0]
            ratio = sub["mse_vs_thr"].values[0]
            det   = sub["detected"].values[0]
            is_isc = "ISC" in case

            row_text.append(f"{mse:.5f}\n비율: {ratio:.3f}")

            if not is_isc:
                # Normal 케이스: 미탐지(False)가 정상
                color = "#C8E6C9" if not det else "#FFCDD2"
            else:
                # ISC 케이스: 탐지(True)가 정상
                color = "#C8E6C9" if det else "#FFCDD2"

            row_colors.append(color)
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    fig_h  = 1.0 + n_rows * 0.85
    fig_w  = 2.5 + n_cols * 3.2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#E3F2FD"] * n_rows,
        colColours=["#1565C0"] * n_cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.2)

    # 헤더 글씨 흰색
    for j in range(n_cols):
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Table 1. 노이즈 수준별 ISC 탐지 결과\n"
        "셀값: Autoencoder MSE  /  비율 = MSE÷임계값  (초록=정상판정, 빨강=오판정)",
        fontsize=11, fontweight="bold", pad=14
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pipeline_table_image(df: pd.DataFrame, thr_on: float, thr_off: float, out_path):
    """
    Table 2: Online vs Offline 파이프라인 비교 표 이미지 저장.
    """
    col_labels = [
        "Online\nMSE", "Online\n임계값", "Online\n비율", "Online\n결과",
        "Offline\nMSE", "Offline\n임계값", "Offline\n비율", "Offline\n결과",
    ]
    row_labels = df["case"].tolist()

    cell_text   = []
    cell_colors = []

    for _, row in df.iterrows():
        is_isc = "ISC" in row["case"]

        def verdict(detected, is_fault):
            if is_fault:
                return ("탐지 O", "#C8E6C9") if detected else ("미탐지 X", "#FFCDD2")
            else:
                return ("정상 O", "#C8E6C9") if not detected else ("오탐지 X", "#FFCDD2")

        on_label,  on_color  = verdict(row["online_detected"],  is_isc)
        off_label, off_color = verdict(row["offline_detected"], is_isc)

        neutral = "#FFFFFF"
        cell_text.append([
            f"{row['online_mse']:.6f}",
            f"{thr_on:.6f}",
            f"{row['online_ratio']:.3f}",
            on_label,
            f"{row['offline_mse']:.6f}",
            f"{thr_off:.6f}",
            f"{row['offline_ratio']:.3f}",
            off_label,
        ])
        cell_colors.append([
            neutral, neutral,
            "#FFF9C4" if row["online_ratio"] < 1.0 else "#E8F5E9",
            on_color,
            neutral, neutral,
            "#FFF9C4" if row["offline_ratio"] < 1.0 else "#E8F5E9",
            off_color,
        ])

    n_rows = len(row_labels)
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(14, 1.2 + n_rows * 0.9))
    ax.axis("off")

    # 헤더를 Online / Offline 두 색으로 구분
    col_colors = (["#1565C0"] * 4) + (["#BF360C"] * 4)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellColours=cell_colors,
        rowColours=["#E3F2FD"] * n_rows,
        colColours=col_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 2.4)

    for j in range(n_cols):
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Table 2. Online vs Offline 파이프라인 ISC 탐지 성능 비교 (AWGN std=0.01V)\n"
        "비율 = MSE÷임계값  (>1.0 이면 이상 탐지)  |  초록=정상판정, 빨강=오판정",
        fontsize=11, fontweight="bold", pad=14
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(SEED)
    print("=" * 60)
    print("  기말 프로젝트 최종 보고서 시각화 시작")
    print("=" * 60)

    # 공통 OCV 테이블 + EKF 파라미터
    print("\n[초기화] OCV 테이블 및 EKF 파라미터 로드...")
    normal_cc = load_right_block(NORMAL_CC_PATH)
    soc_table, ocv_table = build_ocv_table_from_cc([normal_cc])
    cap = normal_cc["capacity_ah"].max() * 3600.0
    ekf_params = load_best_ekf_params()

    # 1.2CC 파일 분류
    normal_files = sorted(NORMAL_DST_DIR.glob("*1.2CC*.csv"))
    isc_files    = sorted(ISC_DST_DIR.glob("*1.2CC*.csv"))
    normal_bd    = [p for p in normal_files if "ISC_BD" in p.name]
    isc_cs       = [p for p in isc_files    if "ISC_CS" in p.name]

    isc_paths = {
        "10ohm":   next((p for p in isc_cs if "10ohm"   in p.name and "100ohm" not in p.name and "1000ohm" not in p.name), None),
        "100ohm":  next((p for p in isc_cs if "100ohm"  in p.name and "1000ohm" not in p.name), None),
        "1000ohm": next((p for p in isc_cs if "1000ohm" in p.name), None),
    }
    normal_test  = next((p for p in normal_bd if "1000ohm" in p.name), None)
    train_bd     = [p for p in normal_bd if "1000ohm" not in p.name] or normal_bd

    missing = [k for k, v in isc_paths.items() if v is None] + (["normal_test"] if normal_test is None else [])
    if missing:
        raise FileNotFoundError(f"필요한 파일 없음: {missing}")

    # 케이스 라벨 (표시용)
    CASE_LABELS = {
        "Normal (1000ohm BD)": normal_test,
        "ISC 10ohm":           isc_paths["10ohm"],
        "ISC 100ohm":          isc_paths["100ohm"],
        "ISC 1000ohm":         isc_paths["1000ohm"],
    }

    # ──────────────────────────────────────────────────────────────────────────
    # PART 1: 노이즈 수준별 강건성 검증
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Part 1] 노이즈 수준별 EKF Residual + Autoencoder 비교")
    print("─" * 50)

    noise_results    = {}  # noise_std -> {case_label: {mse, threshold, detected}}
    residuals_1000   = {}  # noise_std -> result_df  (1000ohm ISC, for time-domain plot)

    for noise_std, short_name in zip(NOISE_STDS, NOISE_NAMES_SHORT):
        print(f"  처리 중: {short_name}")

        train_psds = []
        for p in train_bd:
            _, psd, _ = run_full_pipeline(p, soc_table, ocv_table, cap, ekf_params,
                                          noise_std=noise_std, offline=True)
            train_psds.append(psd)

        test_psd_dict = {}
        for label, path in CASE_LABELS.items():
            _, psd, result = run_full_pipeline(path, soc_table, ocv_table, cap, ekf_params,
                                               noise_std=noise_std, offline=True)
            test_psd_dict[label] = psd
            if "1000ohm" in label.lower() and "ISC" in label:
                residuals_1000[noise_std] = result

        results, threshold = train_and_eval(train_psds, test_psd_dict)
        noise_results[noise_std] = results

    # ── Figure 1: 1000ohm ISC Residual 시계열 × 노이즈 수준 비교 ───────────────
    print("\n  [Figure 1] 노이즈별 EKF Residual 시계열 비교 그래프 생성...")
    fig, axes = plt.subplots(len(NOISE_STDS), 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        "Figure 1. EKF Residual on 1000ohm ISC — AWGN 노이즈 수준별 비교\n"
        "(실제 BMS 기준 10mV vs 임의 노이즈 50~100mV)",
        fontsize=13, fontweight="bold"
    )
    COLORS_NOISE = ["#1565C0", "#2E7D32", "#E65100", "#B71C1C"]

    for idx, (noise_std, short_name, color) in enumerate(
        zip(NOISE_STDS, NOISE_NAMES_SHORT, COLORS_NOISE)
    ):
        ax = axes[idx]
        if noise_std not in residuals_1000:
            ax.set_visible(False)
            continue

        result = residuals_1000[noise_std]
        t0 = result["time"].iloc[0]
        stable = result[result["time"] >= t0 + 500]
        t = stable["time"].to_numpy()
        r = stable["residual"].to_numpy() * 1e3

        ax.plot(t, r, linewidth=0.7, alpha=0.85, color=color)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_ylabel("Innovation [mV]", fontsize=9)
        ax.grid(True, alpha=0.25)

        noise_mV = noise_std * 1000
        entry = noise_results.get(noise_std, {}).get("ISC 1000ohm", {})
        status = "탐지 성공" if entry.get("detected") else "탐지 실패"
        status_color = "darkgreen" if entry.get("detected") else "crimson"
        title_str = f"{short_name}  →  1000ohm ISC 탐지 결과: {status}"
        ax.set_title(title_str, fontsize=10, fontweight="bold", color=status_color)

        info = f"σ={noise_mV:.0f}mV  MSE={entry.get('mse', 0):.5f}  Thr={entry.get('threshold', 0):.5f}"
        ax.annotate(info, xy=(0.02, 0.86), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

    axes[-1].set_xlabel("Time [s]", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = RESULTS_DIR / "final_fig1_noise_residual_comparison.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    print(f"    저장: {out1.name}")

    # ── Figure 2: Autoencoder MSE 바 차트 (노이즈별 탐지 성능) ──────────────────
    print("  [Figure 2] Autoencoder MSE 바 차트 생성...")
    case_labels  = list(CASE_LABELS.keys())
    n_cases = len(case_labels)
    n_noise = len(NOISE_STDS)
    x       = np.arange(n_cases)
    bar_w   = 0.19
    COLORS_NOISE2 = ["#1565C0", "#2E7D32", "#E65100", "#B71C1C"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (noise_std, short_name, color) in enumerate(
        zip(NOISE_STDS, NOISE_NAMES_SHORT, COLORS_NOISE2)
    ):
        if noise_std not in noise_results:
            continue
        res = noise_results[noise_std]
        mses = [res.get(k, {}).get("mse", 0) for k in case_labels]
        threshold = res.get(case_labels[0], {}).get("threshold", 0)
        offset = (i - n_noise / 2 + 0.5) * bar_w

        ax.bar(x + offset, mses, bar_w, label=short_name,
               color=color, alpha=0.82, edgecolor="white", zorder=3)
        ax.hlines(threshold, x[0] + offset - bar_w * 0.55,
                  x[-1] + offset + bar_w * 0.55,
                  colors=color, linestyles="dashed", linewidths=1.6, alpha=0.9, zorder=4)

        # 탐지 성공/실패 마크
        for j, key in enumerate(case_labels):
            entry = res.get(key, {})
            is_isc = "ISC" in key
            detected = entry.get("detected", False)
            mark  = "O" if (not is_isc and not detected) or (is_isc and detected) else "X"
            mcolor = "darkgreen" if mark == "O" else "crimson"
            mse_val = entry.get("mse", 0)
            ax.text(x[j] + offset, mse_val + threshold * 0.03, mark,
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=mcolor, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=10)
    ax.set_ylabel("Autoencoder Reconstruction MSE", fontsize=11)
    ax.set_title(
        "Figure 2. ISC 탐지 성능: 노이즈 수준별 Autoencoder MSE 비교\n"
        "(점선=임계값, O=정상탐지/정상정상, X=오탐지/미탐지)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Noise Level (AWGN std)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    out2 = RESULTS_DIR / "final_fig2_noise_mse_comparison.png"
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"    저장: {out2.name}")

    # ── Table 1: 노이즈 수준별 탐지 결과 CSV ─────────────────────────────────────
    rows1 = []
    for noise_std, short_name in zip(NOISE_STDS, NOISE_NAMES_SHORT):
        if noise_std not in noise_results:
            continue
        for key in case_labels:
            entry = noise_results[noise_std].get(key, {})
            rows1.append({
                "noise_std_V":  noise_std,
                "noise_std_mV": noise_std * 1000,
                "noise_level":  short_name,
                "case":         key,
                "mse":          round(entry.get("mse", float("nan")), 7),
                "threshold":    round(entry.get("threshold", float("nan")), 7),
                "mse_vs_thr":   round(entry.get("mse", 0) / max(entry.get("threshold", 1e-9), 1e-9), 4),
                "detected":     entry.get("detected", False),
            })
    df1 = pd.DataFrame(rows1)
    csv1 = RESULTS_DIR / "final_table1_noise_comparison.csv"
    df1.to_csv(csv1, index=False)
    print(f"  [Table 1] 저장: {csv1.name}")

    # ──────────────────────────────────────────────────────────────────────────
    # PART 2: Offline / Online 파이프라인 비교 (BMS 기준 노이즈 std=0.01 사용)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("[Part 2] Offline vs Online 파이프라인 비교 (std=0.01V 노이즈)")
    print("─" * 50)
    BEST_NOISE = 0.01

    # 1000ohm ISC로 시계열/PSD 비교 (가장 약한 단락 → 핵심 검증 대상)
    isc1000_path = isc_paths["1000ohm"]

    print("  Online 파이프라인 실행 (Raw Welch PSD, 실시간 가능)...")
    freqs_on, psd_on, result_on = run_full_pipeline(
        isc1000_path, soc_table, ocv_table, cap, ekf_params,
        noise_std=BEST_NOISE, offline=False
    )

    print("  Offline 파이프라인 실행 (Zero-phase Butterworth + Welch PSD)...")
    freqs_off, psd_off, result_off = run_full_pipeline(
        isc1000_path, soc_table, ocv_table, cap, ekf_params,
        noise_std=BEST_NOISE, offline=True
    )

    # 각 파이프라인별 학습/테스트 PSD 생성
    print("  각 파이프라인별 Autoencoder 학습/평가 데이터 생성...")
    train_on  = []
    train_off = []
    for p in train_bd:
        _, psd, _ = run_full_pipeline(p, soc_table, ocv_table, cap, ekf_params,
                                      noise_std=BEST_NOISE, offline=False)
        train_on.append(psd)
        _, psd, _ = run_full_pipeline(p, soc_table, ocv_table, cap, ekf_params,
                                      noise_std=BEST_NOISE, offline=True)
        train_off.append(psd)

    test_on  = {}
    test_off = {}
    for label, path in CASE_LABELS.items():
        _, psd, _ = run_full_pipeline(path, soc_table, ocv_table, cap, ekf_params,
                                      noise_std=BEST_NOISE, offline=False)
        test_on[label] = psd
        _, psd, _ = run_full_pipeline(path, soc_table, ocv_table, cap, ekf_params,
                                      noise_std=BEST_NOISE, offline=True)
        test_off[label] = psd

    results_on,  thr_on  = train_and_eval(train_on,  test_on)
    results_off, thr_off = train_and_eval(train_off, test_off)

    # ── Figure 3: 시계열 + PSD 비교 (1000ohm ISC) ──────────────────────────────
    print("\n  [Figure 3] Online vs Offline 시계열/PSD 비교 그래프 생성...")

    t0_on  = result_on["time"].iloc[0]
    t0_off = result_off["time"].iloc[0]
    stab_on  = result_on[result_on["time"]   >= t0_on  + 500]
    stab_off = result_off[result_off["time"] >= t0_off + 500]

    t_on  = stab_on["time"].to_numpy()
    t_off = stab_off["time"].to_numpy()
    r_on  = stab_on["residual"].to_numpy() * 1e3
    r_off = stab_off["residual"].to_numpy() * 1e3
    f_off = stab_off["filtered_residual"].to_numpy() * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        "Figure 3. Online vs Offline 파이프라인 비교 — 1000ohm ISC (std=0.01V 노이즈)\n"
        "Online: Welch PSD 직접 적용  |  Offline: Zero-phase Butterworth + Welch PSD",
        fontsize=12, fontweight="bold"
    )

    # 서브플롯 1: Online 입력 (raw residual)
    axes[0].plot(t_on, r_on, linewidth=0.7, alpha=0.85, color="#1565C0",
                 label="Raw EKF Residual (Online 입력)")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    axes[0].set_title("① Online 파이프라인 — Raw Residual (Causal, 실시간 스트리밍 가능)",
                       fontweight="bold", fontsize=10, color="#1565C0")
    axes[0].set_ylabel("Innovation [mV]", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.25)
    axes[0].annotate(
        "미래 데이터 불필요 → 1초 단위 실시간 배포 가능\n단점: 고주파 노이즈 포함",
        xy=(0.01, 0.82), xycoords="axes fraction", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.9)
    )

    # 서브플롯 2: Offline 입력 (zero-phase filtered residual)
    axes[1].plot(t_off, r_off, linewidth=0.6, alpha=0.45, color="gray",
                 label="Raw Residual")
    axes[1].plot(t_off, f_off, linewidth=1.3, color="#E65100",
                 label="Zero-phase Butterworth LPF (Offline)")
    axes[1].axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    axes[1].set_title("② Offline 파이프라인 — Zero-phase Butterworth (Non-causal, 사후 정밀 분석)",
                       fontweight="bold", fontsize=10, color="#E65100")
    axes[1].set_ylabel("Innovation [mV]", fontsize=10)
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(True, alpha=0.25)
    axes[1].annotate(
        "전체 신호에 양방향 필터 적용 → 위상 지연 없음, 고순도 신호 복원\n단점: 미래 데이터 필요 → 실시간 불가",
        xy=(0.01, 0.78), xycoords="axes fraction", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", alpha=0.9)
    )

    # 서브플롯 3: PSD 비교
    on_entry  = results_on.get("ISC 1000ohm", {})
    off_entry = results_off.get("ISC 1000ohm", {})
    on_status  = "탐지 O" if on_entry.get("detected")  else "미탐지 X"
    off_status = "탐지 O" if off_entry.get("detected") else "미탐지 X"

    axes[2].semilogy(freqs_on,  psd_on  + 1e-18, linewidth=1.2, color="#1565C0",
                     label=f"Online PSD (Raw Welch) → MSE={on_entry.get('mse', 0):.5f} [{on_status}]")
    axes[2].semilogy(freqs_off, psd_off + 1e-18, linewidth=1.5, color="#E65100",
                     label=f"Offline PSD (Filtered Welch) → MSE={off_entry.get('mse', 0):.5f} [{off_status}]")
    axes[2].axvline(0.05, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
                    label="Butterworth cutoff (0.05 Hz)")
    axes[2].set_xlim(0.0, 0.15)
    axes[2].set_title("③ Welch PSD 비교: Online vs Offline — ISC 1000ohm",
                       fontweight="bold", fontsize=10)
    axes[2].set_xlabel("Frequency [Hz]", fontsize=10)
    axes[2].set_ylabel("PSD [V²/Hz]", fontsize=10)
    axes[2].legend(loc="upper right", fontsize=8.5)
    axes[2].grid(True, alpha=0.25, which="both")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out3 = RESULTS_DIR / "final_fig3_pipeline_comparison.png"
    fig.savefig(out3, dpi=200)
    plt.close(fig)
    print(f"    저장: {out3.name}")

    # ── Figure 4: Online vs Offline MSE 바 차트 ────────────────────────────────
    print("  [Figure 4] Online vs Offline MSE 바 차트 생성...")
    x    = np.arange(len(case_labels))
    bw   = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    mses_on_list  = [results_on.get(k,  {}).get("mse", 0) for k in case_labels]
    mses_off_list = [results_off.get(k, {}).get("mse", 0) for k in case_labels]

    b1 = ax.bar(x - bw / 2, mses_on_list,  bw, label="Online (Raw Welch PSD)",
                color="#1565C0", alpha=0.85, edgecolor="white", zorder=3)
    b2 = ax.bar(x + bw / 2, mses_off_list, bw, label="Offline (Butterworth+Welch PSD)",
                color="#E65100", alpha=0.85, edgecolor="white", zorder=3)

    ax.axhline(thr_on,  color="#1565C0", linestyle="--", linewidth=2.0,
               label=f"Online 임계값 ({thr_on:.5f})", zorder=4)
    ax.axhline(thr_off, color="#E65100", linestyle="--", linewidth=2.0,
               label=f"Offline 임계값 ({thr_off:.5f})", zorder=4)

    for j, key in enumerate(case_labels):
        is_isc = "ISC" in key
        on_e   = results_on.get(key, {})
        off_e  = results_off.get(key, {})

        def mark_bar(xpos, mse_val, detected, is_fault):
            correct = (is_fault and detected) or (not is_fault and not detected)
            symbol = "O" if correct else "X"
            color  = "darkgreen" if correct else "crimson"
            ax.text(xpos, mse_val + max(thr_on, thr_off) * 0.02, symbol,
                    ha="center", va="bottom", fontsize=11, fontweight="bold",
                    color=color, zorder=5)

        mark_bar(x[j] - bw / 2, mses_on_list[j],  on_e.get("detected",  False), is_isc)
        mark_bar(x[j] + bw / 2, mses_off_list[j], off_e.get("detected", False), is_isc)

    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, fontsize=10)
    ax.set_ylabel("Autoencoder Reconstruction MSE", fontsize=11)
    ax.set_title(
        "Figure 4. Online vs Offline 파이프라인 ISC 탐지 성능 비교 (std=0.01V 노이즈)\n"
        "(점선=임계값, O=올바른판정, X=오판정)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    out4 = RESULTS_DIR / "final_fig4_pipeline_mse_comparison.png"
    fig.savefig(out4, dpi=200)
    plt.close(fig)
    print(f"    저장: {out4.name}")

    # ── Table 2: Online vs Offline 탐지 결과 CSV ─────────────────────────────
    rows2 = []
    for key in case_labels:
        on_e  = results_on.get(key, {})
        off_e = results_off.get(key, {})
        rows2.append({
            "case":              key,
            "online_mse":        round(on_e.get("mse", float("nan")), 7),
            "online_threshold":  round(thr_on, 7),
            "online_ratio":      round(on_e.get("mse", 0) / max(thr_on, 1e-9), 4),
            "online_detected":   on_e.get("detected", False),
            "offline_mse":       round(off_e.get("mse", float("nan")), 7),
            "offline_threshold": round(thr_off, 7),
            "offline_ratio":     round(off_e.get("mse", 0) / max(thr_off, 1e-9), 4),
            "offline_detected":  off_e.get("detected", False),
        })
    df2 = pd.DataFrame(rows2)
    csv2 = RESULTS_DIR / "final_table2_pipeline_comparison.csv"
    df2.to_csv(csv2, index=False)
    print(f"  [Table 2] 저장: {csv2.name}")

    # ── Table 이미지 저장 ──────────────────────────────────────────────────────
    print("\n  [Table 이미지] 결과 표 이미지 생성...")
    _save_noise_table_image(df1, RESULTS_DIR / "final_table1_noise_image.png")
    _save_pipeline_table_image(df2, thr_on, thr_off, RESULTS_DIR / "final_table2_pipeline_image.png")
    print("    저장: final_table1_noise_image.png")
    print("    저장: final_table2_pipeline_image.png")

    # ── 최종 콘솔 요약 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Part 1 - 1000ohm ISC 탐지 결과 (가장 약한 단락)")
    print("=" * 60)
    for noise_std, name in zip(NOISE_STDS, NOISE_NAMES_SHORT):
        e = noise_results.get(noise_std, {}).get("ISC 1000ohm", {})
        mse = e.get("mse", 0)
        thr = e.get("threshold", 1e-9)
        status = "[탐지 O]" if e.get("detected") else "[미탐지 X]"
        print(f"  {name[:22]}  MSE={mse:.6f}  Thr={thr:.6f}  MSE/Thr={mse/max(thr,1e-9):.3f}  {status}")

    print("\n" + "=" * 60)
    print("  Part 2 - Online vs Offline 전체 케이스 (std=0.01V)")
    print("=" * 60)
    for key in case_labels:
        on_e  = results_on.get(key, {})
        off_e = results_off.get(key, {})
        on_r  = "O" if on_e.get("detected")  else "X"
        off_r = "O" if off_e.get("detected") else "X"
        print(f"  {key[:20]}  Online={on_e.get('mse',0):.6f}[{on_r}]"
              f"  Offline={off_e.get('mse',0):.6f}[{off_r}]")

    print("\n" + "=" * 60)
    print("  저장된 결과 파일")
    print("=" * 60)
    for f in sorted(RESULTS_DIR.glob("final_*")):
        print(f"  {f.name}")
    print()


if __name__ == "__main__":
    main()
