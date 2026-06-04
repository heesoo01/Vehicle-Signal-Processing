import numpy as np


class EKF:
    def __init__(
        self,
        x0,
        P0,
        Q,
        R,
        R0,
        R1,
        C1,
        capacity,
        soc_table,
        ocv_table,
        efficiency=1.0,
    ):
        self.x = np.array(x0, dtype=float)  # [SOC, Vrc]
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.R0 = R0
        self.R1 = R1
        self.C1 = C1
        self.capacity = capacity
        self.efficiency = efficiency

        self.soc_table = np.array(soc_table, dtype=float)
        self.ocv_table = np.array(ocv_table, dtype=float)
        self.docv_table = np.gradient(self.ocv_table, self.soc_table)

        self.K = np.zeros((2, 1))

    def predict(self, current, dt):
        soc = self.x[0]
        vrc = self.x[1]

        a = np.exp(-dt / (self.R1 * self.C1))

        soc_hat = soc - self.efficiency * current * dt / self.capacity
        vrc_hat = a * vrc + self.R1 * (1 - a) * current

        self.x = np.array([soc_hat, vrc_hat])
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        F = np.array(
            [
                [1.0, 0.0],
                [0.0, a],
            ]
        )

        self.P = F @ self.P @ F.T + self.Q

    def correction(self, current, voltage):
        voltage_hat = self.estimate_voltage(current)
        y = voltage - voltage_hat

        H = np.array([[self.docv_dsoc(self.x[0]), -1.0]])

        S = H @ self.P @ H.T + self.R
        self.K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (self.K.flatten() * y)
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        I = np.eye(2)
        self.P = (I - self.K @ H) @ self.P

        return {
            "soc": self.x[0],
            "vrc": self.x[1],
            "voltage_hat": voltage_hat,
            "residual": y,
            "S": float(S[0, 0]),
        }

    def update(self, current, voltage, dt):
        self.predict(current, dt)
        return self.correction(current, voltage)

    def estimate_voltage(self, current):
        soc = self.x[0]
        vrc = self.x[1]
        return self.ocv(soc) - vrc - current * self.R0

    def ocv(self, soc):
        return np.interp(soc, self.soc_table, self.ocv_table)

    def docv_dsoc(self, soc):
        return np.interp(soc, self.soc_table, self.docv_table)
