#Kaip paleisti: python pranasyste2.py --n-start 1 --n-end 9 --interval-ms 700
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["N", "P", "F0"]
TARGET_COLUMN = "Ra"

# Fixed model parameters (no ARD)
FEATURE_MODE = "scale_all"
USE_LOG_TARGET = True
C_VALUE = 0.001
LS_VALUE = 1.91562
NOISE_LEVEL = 5.14043e-07
GPR_ALPHA = 5.95214e-11
MATERN_NU = 2.5
_ANIMATIONS: list[FuncAnimation] = []


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows available after dropping missing values.")
    return df


def build_preprocessor() -> ColumnTransformer:
    if FEATURE_MODE != "scale_all":
        raise ValueError(f"Unsupported feature mode: {FEATURE_MODE}")
    return ColumnTransformer([("scale", StandardScaler(), INPUT_COLUMNS)], remainder="drop")


def train_model(df: pd.DataFrame) -> tuple[ColumnTransformer, GaussianProcessRegressor]:
    x_df = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    if USE_LOG_TARGET and np.any(y <= -1.0):
        raise ValueError("Cannot use log1p target transform because Ra contains values <= -1.")
    y_model = np.log1p(y) if USE_LOG_TARGET else y

    preprocessor = build_preprocessor()
    x = preprocessor.fit_transform(x_df)

    kernel = (
        ConstantKernel(constant_value=C_VALUE, constant_value_bounds="fixed")
        * Matern(length_scale=LS_VALUE, length_scale_bounds="fixed", nu=MATERN_NU)
        + WhiteKernel(noise_level=NOISE_LEVEL, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GPR_ALPHA,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    model.fit(x, y_model)
    return preprocessor, model


def predict_with_inverse(model: GaussianProcessRegressor, x: np.ndarray) -> np.ndarray:
    pred_model = model.predict(x)
    return np.expm1(pred_model) if USE_LOG_TARGET else pred_model


def _compute_frame_data(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: GaussianProcessRegressor,
    n_value: float,
    p_min: float,
    p_max: float,
    f0_min: float,
    f0_max: float,
    grid_points: int,
    use_measured_p_levels: bool,
) -> dict:
    mask = np.isclose(df["N"].to_numpy(dtype=float), n_value)
    mask = (
        mask
        & (df["P"].to_numpy(dtype=float) >= p_min)
        & (df["P"].to_numpy(dtype=float) <= p_max)
        & (df["F0"].to_numpy(dtype=float) >= f0_min)
        & (df["F0"].to_numpy(dtype=float) <= f0_max)
    )
    actual = df.loc[mask, ["P", "F0", TARGET_COLUMN]].copy()

    f0_grid = np.linspace(f0_min, f0_max, grid_points, dtype=float)
    if use_measured_p_levels:
        if actual.empty:
            p_source = df.loc[
                (df["P"].to_numpy(dtype=float) >= p_min) & (df["P"].to_numpy(dtype=float) <= p_max),
                "P",
            ]
        else:
            p_source = actual["P"]
        p_grid = np.sort(p_source.astype(float).unique())
        if p_grid.size < 2:
            p_grid = np.linspace(p_min, p_max, max(2, grid_points), dtype=float)
    else:
        p_grid = np.linspace(p_min, p_max, grid_points, dtype=float)

    f0_mesh, p_mesh = np.meshgrid(f0_grid, p_grid)
    pred_df = pd.DataFrame(
        {
            "N": np.full(f0_mesh.size, n_value, dtype=float),
            "P": p_mesh.ravel(),
            "F0": f0_mesh.ravel(),
        }
    )
    x_pred = preprocessor.transform(pred_df)
    z_pred_grid = predict_with_inverse(model, x_pred)
    z_pred_surface = z_pred_grid.reshape(f0_mesh.shape)

    if actual.empty:
        z_pred_at_actual = np.array([], dtype=float)
        z_actual = np.array([], dtype=float)
        rmse_points = np.nan
    else:
        pred_at_actual_df = pd.DataFrame(
            {
                "N": np.full(len(actual), n_value, dtype=float),
                "P": actual["P"].to_numpy(dtype=float),
                "F0": actual["F0"].to_numpy(dtype=float),
            }
        )
        z_pred_at_actual = predict_with_inverse(model, preprocessor.transform(pred_at_actual_df))
        z_actual = actual[TARGET_COLUMN].to_numpy(dtype=float)
        rmse_points = float(np.sqrt(np.mean((z_actual - z_pred_at_actual) ** 2)))

    return {
        "has_actual": not actual.empty,
        "actual": actual,
        "f0_mesh": f0_mesh,
        "p_mesh": p_mesh,
        "z_pred_surface": z_pred_surface,
        "z_pred_at_actual": z_pred_at_actual,
        "z_actual": z_actual,
        "rmse": rmse_points,
    }


def animate_plot_3d(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: GaussianProcessRegressor,
    n_start: int,
    n_end: int,
    p_min: float,
    p_max: float,
    f0_min: float,
    f0_max: float,
    grid_points: int,
    use_measured_p_levels: bool,
    interval_ms: int,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    frames = list(range(n_start, n_end + 1))
    p_mode = "measured P levels" if use_measured_p_levels else "uniform P grid"

    def update(frame_n: int) -> None:
        ax.cla()
        frame_data = _compute_frame_data(
            df=df,
            preprocessor=preprocessor,
            model=model,
            n_value=float(frame_n),
            p_min=p_min,
            p_max=p_max,
            f0_min=f0_min,
            f0_max=f0_max,
            grid_points=grid_points,
            use_measured_p_levels=use_measured_p_levels,
        )

        ax.set_xlim(f0_min, f0_max)
        ax.set_ylim(p_min, p_max)
        ax.set_xlabel("F0")
        ax.set_ylabel("P")
        ax.set_zlabel("Ra")

        ax.plot_surface(
            frame_data["f0_mesh"],
            frame_data["p_mesh"],
            frame_data["z_pred_surface"],
            cmap="viridis",
            alpha=0.6,
            linewidth=0,
            antialiased=True,
        )

        if frame_data["has_actual"]:
            actual = frame_data["actual"]
            ax.scatter(
                actual["F0"].to_numpy(dtype=float),
                actual["P"].to_numpy(dtype=float),
                frame_data["z_actual"],
                color="red",
                s=42,
                alpha=0.95,
                label="Actual Ra",
            )
            ax.scatter(
                actual["F0"].to_numpy(dtype=float),
                actual["P"].to_numpy(dtype=float),
                frame_data["z_pred_at_actual"],
                color="blue",
                s=28,
                alpha=0.9,
                label="Predicted at actual points",
            )
            ax.set_title(
                f"Ra predicted surface (N={frame_n}, Matern 2.5 no ARD, {p_mode}) | "
                f"RMSE={frame_data['rmse']:.4f}"
            )
            ax.legend(loc="upper left")
        else:
            ax.set_title(
                f"Ra predicted surface (N={frame_n}, Matern 2.5 no ARD, {p_mode}) | "
                "no actual points for this N"
            )

    anim = FuncAnimation(fig, update, frames=frames, interval=interval_ms, repeat=True)
    _ANIMATIONS.append(anim)  # keep reference alive until process ends
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matplotlib animation of Ra predicted surface for N from 1 to 9."
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys_Nscan_3.xlsx")
    parser.add_argument("--n-start", type=int, default=1)
    parser.add_argument("--n-end", type=int, default=9)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=25.0)
    parser.add_argument("--f0-min", type=float, default=0.0)
    parser.add_argument("--f0-max", type=float, default=10.0)
    parser.add_argument("--grid-points", type=int, default=50)
    parser.add_argument("--interval-ms", type=int, default=900)
    parser.add_argument(
        "--uniform-p-grid",
        action="store_true",
        help="Use uniform P grid instead of measured P levels.",
    )
    args = parser.parse_args()

    if args.n_end < args.n_start:
        raise ValueError("--n-end must be >= --n-start")

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = load_data(file_path)
    preprocessor, model = train_model(df)

    animate_plot_3d(
        df=df,
        preprocessor=preprocessor,
        model=model,
        n_start=int(args.n_start),
        n_end=int(args.n_end),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        f0_min=float(args.f0_min),
        f0_max=float(args.f0_max),
        grid_points=int(args.grid_points),
        use_measured_p_levels=not bool(args.uniform_p_grid),
        interval_ms=int(args.interval_ms),
    )


if __name__ == "__main__":
    main()
