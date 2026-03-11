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


def train_ra_model(
    df: pd.DataFrame,
) -> tuple[StandardScaler, GaussianProcessRegressor]:
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


def build_prediction_grid_within_measured_ranges(
    df: pd.DataFrame,
    points_per_pair: int = 60,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    bounds = (
        df.groupby(["N", "P"], as_index=False)["F0"]
        .agg(f0_min="min", f0_max="max")
        .reset_index(drop=True)
    )

    for row in bounds.itertuples(index=False):
        n_val = float(row.N)
        p_val = float(row.P)
        f0_min = float(row.f0_min)
        f0_max = float(row.f0_max)
        if np.isclose(f0_min, f0_max):
            f0_values = np.array([f0_min], dtype=float)
        else:
            f0_values = np.linspace(f0_min, f0_max, points_per_pair, dtype=float)
        for f0 in f0_values:
            rows.append({"N": n_val, "P": p_val, "F0": float(f0)})

    return pd.DataFrame(rows, columns=INPUT_COLUMNS)


def pareto_front_indices(depth: np.ndarray, ra: np.ndarray) -> np.ndarray:
    # Pareto objective: maximize Depth (x) and minimize Ra (y).
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
    return f"N={float(row['N']):.6g}\nP={float(row['P']):.6g}\nF0={float(row['F0']):.6g}"


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
        mouse_event = event  # keep name explicit for readability
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

    depth_model = train_depth_model(df)
    ra_scaler, ra_model = train_ra_model(df)

    measured_depth = df[DEPTH_COLUMN].to_numpy(dtype=float)
    measured_ra = df[RA_COLUMN].to_numpy(dtype=float)

    pred_grid = build_prediction_grid_within_measured_ranges(df, points_per_pair=60)
    x_pred_raw = pred_grid[INPUT_COLUMNS].to_numpy(dtype=float)

    pred_depth = depth_model.predict(x_pred_raw)
    pred_ra_log = ra_model.predict(ra_scaler.transform(x_pred_raw))
    pred_ra = np.expm1(pred_ra_log)

    meas_front_idx = pareto_front_indices(measured_depth, measured_ra)
    pred_front_idx = pareto_front_indices(pred_depth, pred_ra)
    meas_fx, meas_fy, meas_front_sorted_idx = front_for_plot(
        measured_depth, measured_ra, meas_front_idx
    )
    pred_fx, pred_fy, pred_front_sorted_idx = front_for_plot(
        pred_depth, pred_ra, pred_front_idx
    )

    measured_meta = df[INPUT_COLUMNS].reset_index(drop=True)
    predicted_meta = pred_grid[INPUT_COLUMNS].reset_index(drop=True)
    combo_lookup = (
        df[["N", "P"]]
        .drop_duplicates()
        .sort_values(["N", "P"])
        .reset_index(drop=True)
    )
    combo_lookup["combo_id"] = np.arange(len(combo_lookup), dtype=int)

    measured_meta = measured_meta.merge(combo_lookup, on=["N", "P"], how="left")
    predicted_meta = predicted_meta.merge(combo_lookup, on=["N", "P"], how="left")
    measured_combo = measured_meta["combo_id"].to_numpy(dtype=float)
    predicted_combo = predicted_meta["combo_id"].to_numpy(dtype=float)
    meas_front_combo = measured_combo[meas_front_sorted_idx]
    pred_front_combo = predicted_combo[pred_front_sorted_idx]

    n_combos = int(len(combo_lookup))
    norm = plt.Normalize(vmin=0, vmax=max(n_combos - 1, 1))
    cmap = plt.cm.turbo

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True
    )

    meas_scatter = axes[0].scatter(
        measured_depth,
        measured_ra,
        c=measured_combo,
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.65,
        label="Measured points",
    )
    meas_front_scatter = axes[0].scatter(
        meas_fx,
        meas_fy,
        c=meas_front_combo,
        cmap=cmap,
        norm=norm,
        s=34,
        edgecolors="white",
        linewidths=0.8,
        label="Measured Pareto front",
    )
    axes[0].plot(
        meas_fx, meas_fy, color="black", linewidth=1.8, alpha=0.95, zorder=4
    )
    axes[0].set_title("Measured Data Pareto Front")
    axes[0].set_xlabel("Gylis")
    axes[0].set_ylabel("Ra")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    pred_scatter = axes[1].scatter(
        pred_depth,
        pred_ra,
        c=predicted_combo,
        cmap=cmap,
        norm=norm,
        s=20,
        alpha=0.65,
        label="Predicted points",
    )
    pred_front_scatter = axes[1].scatter(
        pred_fx,
        pred_fy,
        c=pred_front_combo,
        cmap=cmap,
        norm=norm,
        s=34,
        edgecolors="white",
        linewidths=0.8,
        label="Predicted Pareto front",
    )
    axes[1].plot(
        pred_fx, pred_fy, color="black", linewidth=1.8, alpha=0.95, zorder=4
    )
    axes[1].set_title("Predicted Pareto Front (Matern 2.5 ARD)")
    axes[1].set_xlabel("Gylis")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        fraction=0.035,
        pad=0.04,
    )
    colorbar.set_label("(N, P) combination index (sorted by N, then P)")

    add_hover_annotations(
        axes[0],
        [
            (meas_front_scatter, measured_meta, meas_front_sorted_idx),
            (meas_scatter, measured_meta, None),
        ],
    )
    add_hover_annotations(
        axes[1],
        [
            (pred_front_scatter, predicted_meta, pred_front_sorted_idx),
            (pred_scatter, predicted_meta, None),
        ],
    )

    fig.suptitle(
        "Pareto fronts (objective: maximize Gylis, minimize Ra)\n"
        "Predictions restricted to measured F0 min/max range for each (N, P)",
        fontsize=11,
    )
    plt.show()


if __name__ == "__main__":
    main()
