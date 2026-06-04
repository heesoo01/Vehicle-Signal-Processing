import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_ekf_data(result_data, output_path, title=None):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    if title is not None:
        fig.suptitle(title)

    axes[0].plot(result_data["time"], result_data["voltage"], label="Measured voltage")
    axes[0].plot(result_data["time"], result_data["voltage_hat"], label="Estimated normal voltage")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(result_data["time"], result_data["residual"])
    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Innovation [V]")
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
