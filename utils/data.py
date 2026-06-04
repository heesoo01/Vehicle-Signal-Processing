import numpy as np
import pandas as pd


def load_left_block(path):
    """충전 블록(왼쪽)을 읽어옴. CV 말기 데이터로 OCV(SOC=1.0) 앵커 추출에 사용."""
    df = pd.read_csv(path)
    data = pd.DataFrame({
        "time":        df.iloc[:, 0],
        "current_raw": df.iloc[:, 1],
        "soc_percent": df.iloc[:, 3],
        "voltage":     df.iloc[:, 4],
    }).dropna()
    data["time"] = data["time"] - data["time"].iloc[0]
    return data.reset_index(drop=True)


## 데이터셋에서 충전부가 아닌, 방전부를 읽어옴
def load_right_block(path):
    df = pd.read_csv(path)

    data = pd.DataFrame(
        {
            "time": df.iloc[:, 6],
            "current_raw": df.iloc[:, 7],
            "capacity_ah": df.iloc[:, 8],
            "soc_dod_percent": df.iloc[:, 9],
            "voltage": df.iloc[:, 10],
        }
    ).dropna()

    data["time"] = data["time"] - data["time"].iloc[0]

    # 데이터셋과 EKF에 들어가는 전류 방향이 달라서 -부호를 붙임
    data["current"] = -data["current_raw"]

    return data.reset_index(drop=True)


def add_coulomb_counted_soc(data, capacity_coulomb):
    soc = [1.0]
    times = data["time"].to_numpy()
    currents = data["current"].to_numpy()

    for i in range(1, len(data)):
        dt = times[i] - times[i - 1]
        next_soc = soc[-1] - currents[i] * dt / capacity_coulomb
        soc.append(np.clip(next_soc, 0.0, 1.0))

    data = data.copy()
    data["soc_ref"] = soc
    return data


def add_dataset_dod_soc(data):
    data = data.copy()
    data["soc_ref"] = 1.0 - data["soc_dod_percent"] / 100.0
    data["soc_ref"] = data["soc_ref"].clip(0.0, 1.0)
    return data
