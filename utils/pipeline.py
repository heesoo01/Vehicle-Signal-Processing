import numpy as np

from .ekf import EKF


def run_ekf(
    data,
    soc_table,
    ocv_table,
    capacity_coulomb,
    ekf_params=None,
    initial_soc=None,
    initialize_vrc=True,
):
    params = {
        "P0": np.diag([1e-4, 1e-3]),
        "Q": np.diag([1e-6, 1e-7]),   # CC NIS 그리드서치 결과
        "R": np.array([[3e-5]]),        # CC NIS 그리드서치 결과
        "R0": 0.015,
        "R1": 0.01,
        "C1": 2400.0,
    }
    if ekf_params is not None:
        params.update(ekf_params)

    if initial_soc is None:
        if "soc_ref" in data.columns:
            initial_soc = data["soc_ref"].iloc[0]
        else:
            initial_soc = 1.0

    initial_current = data["current"].iloc[0]
    initial_voltage = data["voltage"].iloc[0]

    if initialize_vrc:
        initial_ocv = np.interp(initial_soc, soc_table, ocv_table)
        initial_vrc = initial_ocv - initial_current * params["R0"] - initial_voltage
    else:
        initial_vrc = 0.0

    ekf = EKF(
        x0=[initial_soc, initial_vrc],
        P0=params["P0"],
        Q=params["Q"],
        R=params["R"],
        R0=params["R0"],
        R1=params["R1"],
        C1=params["C1"],
        capacity=capacity_coulomb,
        soc_table=soc_table,
        ocv_table=ocv_table,
    )

    soc_est = []
    voltage_hat = []
    residual = []
    S_vals = []

    times = data["time"].to_numpy()
    currents = data["current"].to_numpy()
    voltages = data["voltage"].to_numpy()

    for i in range(1, len(data)):
        dt = times[i] - times[i - 1]
        result = ekf.update(currents[i], voltages[i], dt)

        soc_est.append(result["soc"])
        voltage_hat.append(result["voltage_hat"])
        residual.append(result["residual"])
        S_vals.append(result["S"])

    result_data = data.iloc[1:].copy()
    result_data["soc_est"] = soc_est
    result_data["voltage_hat"] = voltage_hat
    result_data["residual"] = residual
    result_data["S"] = S_vals

    return result_data
