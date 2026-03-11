from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["N", "P", "F0"]
RA_COLUMN = "Ra"
DEPTH_COLUMN = "Gylis"
DATA_FILE = Path("surikiuoti_duomenys_Nscan_3.xlsx")

TARGET_N = 8.0
TARGET_P = 1.0


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    required = INPUT_COLUMNS + [RA_COLUMN, DEPTH_COLUMN]
    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after dropping missing values.")
    return df


def train_depth_model(df: pd.DataFrame) -> GaussianProcessRegressor:
    x = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = df[DEPTH_COLUMN].to_numpy(dtype=float)

    kernel = (
        ConstantKernel(0.001, constant_value_bounds="fixed")
        * Matern(
            length_scale=[1.57227, 0.0022466, 11.43],
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-08, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=2.80953e-10,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    model.fit(x, y)
    return model


def train_ra_model(df: pd.DataFrame) -> tuple[StandardScaler, GaussianProcessRegressor]:
    x_raw = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y_raw = df[RA_COLUMN].to_numpy(dtype=float)
    if np.any(y_raw <= -1.0):
        raise ValueError("Cannot apply log1p to Ra values <= -1.")

    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw)
    y = np.log1p(y_raw)

    kernel = (
        ConstantKernel(0.42638329674243786, constant_value_bounds="fixed")
        * Matern(
            length_scale=[3.684816348935756, 8.825687203522195, 100.0],
            length_scale_bounds="fixed",
            nu=2.5,
        )
        + WhiteKernel(noise_level=1e-08, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=4.556011373225105e-12,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    model.fit(x, y)
    return scaler, model


def pareto_front_indices(depth: np.ndarray, ra: np.ndarray) -> np.ndarray:
    # Pareto objective: maximize Depth (x), minimize Ra (y).
    order = np.lexsort((ra, -depth))
    best_ra = np.inf
    front: list[int] = []
    for idx in order:
        y = ra[idx]
        if y < best_ra:
            front.append(int(idx))
            best_ra = y
    return np.array(front, dtype=int)


def front_for_plot(
    depth: np.ndarray, ra: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = depth[idx]
    y = ra[idx]
    sort_idx = np.argsort(x)
    idx_sorted = idx[sort_idx]
    return x[sort_idx], y[sort_idx], idx_sorted


def _hover_text(meta: pd.DataFrame, global_idx: int) -> str:
    row = meta.iloc[int(global_idx)]
    return (
        f"N={float(row['N']):.6g}\n"
        f"P={float(row['P']):.6g}\n"
        f"F0={float(row['F0']):.6g}"
    )


def add_hover_annotations(
    ax: plt.Axes,
    artists: list[tuple[plt.Artist, pd.DataFrame, np.ndarray | None]],
) -> None:
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )
    annot.set_visible(False)

    def on_move(event: object) -> None:
        mouse_event = event
        if mouse_event.inaxes != ax:
            if annot.get_visible():
                annot.set_visible(False)
                ax.figure.canvas.draw_idle()
            return

        changed = False
        for artist, meta, idx_map in artists:
            contains, info = artist.contains(mouse_event)
            ind = info.get("ind")
            if (not contains) or (ind is None) or (len(ind) == 0):
                continue

            local_idx = int(ind[0])
            global_idx = int(idx_map[local_idx]) if idx_map is not None else local_idx
            offset_xy = artist.get_offsets()[local_idx]
            annot.xy = (float(offset_xy[0]), float(offset_xy[1]))
            annot.set_text(_hover_text(meta, global_idx))
            annot.set_visible(True)
            changed = True
            break

        if not changed and annot.get_visible():
            annot.set_visible(False)
            changed = True

        if changed:
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("motion_notify_event", on_move)


def main() -> None:
    df = load_data(DATA_FILE)
    df_np = df[
        np.isclose(df["N"].to_numpy(dtype=float), TARGET_N)
        & np.isclose(df["P"].to_numpy(dtype=float), TARGET_P)
    ].copy()
    if df_np.empty:
        raise ValueError(f"No measured rows found for N={TARGET_N:g}, P={TARGET_P:g}.")

    depth_model = train_depth_model(df)
    ra_scaler, ra_model = train_ra_model(df)

    measured_meta = df_np[INPUT_COLUMNS].reset_index(drop=True)
    measured_depth = df_np[DEPTH_COLUMN].to_numpy(dtype=float)
    measured_ra = df_np[RA_COLUMN].to_numpy(dtype=float)

    f0_min = float(df_np["F0"].min())
    f0_max = float(df_np["F0"].max())
    if np.isclose(f0_min, f0_max):
        f0_grid = np.array([f0_min], dtype=float)
    else:
        f0_grid = np.linspace(f0_min, f0_max, 80, dtype=float)
    pred_grid = pd.DataFrame(
        {
            "N": np.full(f0_grid.size, TARGET_N, dtype=float),
            "P": np.full(f0_grid.size, TARGET_P, dtype=float),
            "F0": f0_grid,
        }
    )
    predicted_meta = pred_grid[INPUT_COLUMNS].reset_index(drop=True)

    x_pred_raw = pred_grid[INPUT_COLUMNS].to_numpy(dtype=float)
    pred_depth = depth_model.predict(x_pred_raw)
    pred_ra = np.expm1(ra_model.predict(ra_scaler.transform(x_pred_raw)))
    x_actual_raw = df_np[INPUT_COLUMNS].to_numpy(dtype=float)
    pred_depth_at_actual = depth_model.predict(x_actual_raw)
    pred_ra_at_actual = np.expm1(ra_model.predict(ra_scaler.transform(x_actual_raw)))

    meas_front_idx = pareto_front_indices(measured_depth, measured_ra)
    pred_front_idx = pareto_front_indices(pred_depth, pred_ra)
    meas_fx, meas_fy, meas_front_sorted_idx = front_for_plot(
        measured_depth, measured_ra, meas_front_idx
    )
    pred_fx, pred_fy, pred_front_sorted_idx = front_for_plot(
        pred_depth, pred_ra, pred_front_idx
    )
    f0_actual = df_np["F0"].to_numpy(dtype=float)
    f0_actual_sort = np.argsort(f0_actual)
    f0_grid_sort = np.argsort(f0_grid)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True
    )

    meas_scatter = axes[0, 0].scatter(
        measured_depth,
        measured_ra,
        s=28,
        alpha=0.65,
        color="#5b9bd5",
        label="Measured points",
    )
    meas_front_scatter = axes[0, 0].scatter(
        meas_fx,
        meas_fy,
        s=42,
        color="#5b9bd5",
        edgecolors="black",
        linewidths=0.9,
        label="Measured Pareto front",
    )
    axes[0, 0].plot(
        meas_fx, meas_fy, color="black", linewidth=1.8, alpha=0.95, zorder=4
    )
    axes[0, 0].set_title("Measured Pareto Front (N=8, P=1)")
    axes[0, 0].set_xlabel("Gylis")
    axes[0, 0].set_ylabel("Ra")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")

    pred_scatter = axes[0, 1].scatter(
        pred_depth,
        pred_ra,
        s=28,
        alpha=0.65,
        color="#ff8c69",
        label="Predicted points",
    )
    pred_front_scatter = axes[0, 1].scatter(
        pred_fx,
        pred_fy,
        s=42,
        color="#ff8c69",
        edgecolors="black",
        linewidths=0.9,
        label="Predicted Pareto front",
    )
    axes[0, 1].plot(
        pred_fx, pred_fy, color="black", linewidth=1.8, alpha=0.95, zorder=4
    )
    axes[0, 1].set_title("Predicted Pareto Front (N=8, P=1)")
    axes[0, 1].set_xlabel("Gylis")
    axes[0, 1].set_ylabel("Ra")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc="best")

    axes[1, 0].scatter(
        f0_actual,
        measured_ra,
        s=42,
        color="#5b9bd5",
        alpha=0.9,
        label="Real Ra",
    )
    axes[1, 0].plot(
        f0_grid[f0_grid_sort],
        pred_ra[f0_grid_sort],
        color="#ff8c69",
        linewidth=2.0,
        alpha=0.9,
        label="Predicted Ra (grid)",
    )
    axes[1, 0].scatter(
        f0_actual,
        pred_ra_at_actual,
        s=28,
        color="#ff8c69",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        label="Predicted Ra (real F0)",
    )
    axes[1, 0].plot(
        f0_actual[f0_actual_sort],
        measured_ra[f0_actual_sort],
        color="#5b9bd5",
        linewidth=1.4,
        alpha=0.7,
    )
    axes[1, 0].set_title("F0 vs Ra (N=8, P=1)")
    axes[1, 0].set_xlabel("F0")
    axes[1, 0].set_ylabel("Ra")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(loc="best")

    axes[1, 1].scatter(
        f0_actual,
        measured_depth,
        s=42,
        color="#5b9bd5",
        alpha=0.9,
        label="Real Gylis",
    )
    axes[1, 1].plot(
        f0_grid[f0_grid_sort],
        pred_depth[f0_grid_sort],
        color="#ff8c69",
        linewidth=2.0,
        alpha=0.9,
        label="Predicted Gylis (grid)",
    )
    axes[1, 1].scatter(
        f0_actual,
        pred_depth_at_actual,
        s=28,
        color="#ff8c69",
        edgecolors="black",
        linewidths=0.6,
        alpha=0.9,
        label="Predicted Gylis (real F0)",
    )
    axes[1, 1].plot(
        f0_actual[f0_actual_sort],
        measured_depth[f0_actual_sort],
        color="#5b9bd5",
        linewidth=1.4,
        alpha=0.7,
    )
    axes[1, 1].set_title("F0 vs Gylis (N=8, P=1)")
    axes[1, 1].set_xlabel("F0")
    axes[1, 1].set_ylabel("Gylis")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc="best")

    add_hover_annotations(
        axes[0, 0],
        [
            (meas_front_scatter, measured_meta, meas_front_sorted_idx),
            (meas_scatter, measured_meta, None),
        ],
    )
    add_hover_annotations(
        axes[0, 1],
        [
            (pred_front_scatter, predicted_meta, pred_front_sorted_idx),
            (pred_scatter, predicted_meta, None),
        ],
    )

    fig.suptitle(
        "N=8, P=1: Pareto and F0 response plots\n"
        "Predictions use Matern 2.5 ARD models within measured F0 range",
        fontsize=11,
    )
    plt.show()

    print(f"Measured rows for N={TARGET_N:g}, P={TARGET_P:g}: {len(df_np)}")
    print(f"Predicted points for N={TARGET_N:g}, P={TARGET_P:g}: {len(pred_grid)}")
    print(f"F0 measured range used for prediction: [{f0_min:.6g}, {f0_max:.6g}]")


if __name__ == "__main__":
    main()
