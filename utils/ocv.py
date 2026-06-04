import numpy as np


def extract_cv_ocv_anchor(left_df, R0=0.015, R1=0.01, C1=2400.0,
                           soc_thresh=99.0, current_thresh=1.0):
    """CV 충전 말기(SOC≥99%, 전류<1A)에서 OCV(SOC=1.0) 앵커 포인트 추출."""
    tau = R1 * C1
    mask = (left_df["soc_percent"] >= soc_thresh) & (left_df["current_raw"] < current_thresh)
    cv = left_df[mask]
    if len(cv) == 0:
        return None, None
    t = cv["time"].to_numpy()
    I = cv["current_raw"].to_numpy()
    V = cv["voltage"].to_numpy()
    ocv = V - R0 * I - R1 * (1.0 - np.exp(-t / tau)) * I
    return cv["soc_percent"].to_numpy() / 100.0, ocv


def build_ocv_table_from_cc(cc_datasets, points=300, degree=7,
                             R0=0.015, R1=0.01, C1=2400.0,
                             cv_anchors=None):
    """
    CC 방전 데이터(파일별 리스트)로 OCV(SOC) 룩업테이블 생성.

    IR 보정: OCV(t) = V_terminal + R0*I + R1*(1-exp(-t/τ))*I
      - t=0에서 Vrc=0 가정 (완전충전 후 방전 시작)
      - 시간에 따른 Vrc 수렴을 물리 모델로 직접 보정 → 초반 데이터도 활용 가능

    cv_anchors: load_left_block + extract_cv_ocv_anchor로 얻은 충전 말기 앵커.
      - SOC=1.0 근방 OCV를 보강해 극초반 EKF 수렴 오차를 줄임.
    """
    tau = R1 * C1
    soc_all, ocv_all = [], []

    for df in cc_datasets:
        soc = 1.0 - df["soc_dod_percent"].to_numpy() / 100.0
        t   = df["time"].to_numpy()
        I   = df["current"].to_numpy()
        V   = df["voltage"].to_numpy()
        ocv_all.append(V + R0 * I + R1 * (1.0 - np.exp(-t / tau)) * I)
        soc_all.append(soc)

    if cv_anchors:
        for soc_anc, ocv_anc in cv_anchors:
            if soc_anc is not None:
                soc_all.append(soc_anc)
                ocv_all.append(ocv_anc)

    soc = np.concatenate(soc_all)
    ocv = np.concatenate(ocv_all)

    coeffs    = np.polyfit(soc, ocv, deg=degree)
    soc_table = np.linspace(soc.min(), soc.max(), points)
    ocv_table = np.polyval(coeffs, soc_table)

    return soc_table, ocv_table
