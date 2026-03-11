#python pranasyste.py --n 8

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

INPUT_COLUMNS = ["N", "P", "F0"]
TARGET_COLUMN = "Gylis"

# Fixed model parameters (Best balance so far)
C_VALUE = 0.001
LS_N = 1.57227
LS_P = 0.0022466
LS_F0 = 11.43
NOISE_LEVEL = 1e-08
GPR_ALPHA = 2.80953e-10
MATERN_NU = 2.5


def load_data(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows available after dropping missing values.")
    return df


def train_model(df: pd.DataFrame) -> tuple[ColumnTransformer, GaussianProcessRegressor]:
    x_df = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    preprocessor = ColumnTransformer([("raw", "passthrough", INPUT_COLUMNS)], remainder="drop")
    x = preprocessor.fit_transform(x_df)

    kernel = (
        ConstantKernel(constant_value=C_VALUE, constant_value_bounds="fixed")
        * Matern(
            length_scale=np.array([LS_N, LS_P, LS_F0], dtype=float),
            length_scale_bounds="fixed",
            nu=MATERN_NU,
        )
        + WhiteKernel(noise_level=NOISE_LEVEL, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=GPR_ALPHA,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    model.fit(x, y)
    return preprocessor, model


def make_plot_3d(
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
) -> None:
    mask = np.isclose(df["N"].to_numpy(dtype=float), n_value)
    mask = (
        mask
        & (df["P"].to_numpy(dtype=float) >= p_min)
        & (df["P"].to_numpy(dtype=float) <= p_max)
        & (df["F0"].to_numpy(dtype=float) >= f0_min)
        & (df["F0"].to_numpy(dtype=float) <= f0_max)
    )
    actual = df.loc[mask, ["P", "F0", TARGET_COLUMN]].copy()
    if actual.empty:
        raise ValueError(f"No actual points found for N={n_value:g} in the selected P/F0 range.")

    f0_grid = np.linspace(f0_min, f0_max, grid_points, dtype=float)
    if use_measured_p_levels:
        p_grid = np.sort(actual["P"].astype(float).unique())
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
    z_pred_grid = model.predict(x_pred)

    # Also compute prediction exactly at measured points for a direct RMSE check.
    pred_at_actual_df = pd.DataFrame(
        {
            "N": np.full(len(actual), n_value, dtype=float),
            "P": actual["P"].to_numpy(dtype=float),
            "F0": actual["F0"].to_numpy(dtype=float),
        }
    )
    z_pred_at_actual = model.predict(preprocessor.transform(pred_at_actual_df))
    z_actual = actual[TARGET_COLUMN].to_numpy(dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        actual["F0"].to_numpy(dtype=float),
        actual["P"].to_numpy(dtype=float),
        z_actual,
        color="red",
        s=42,
        alpha=0.95,
        label="Actual Gylis",
    )
    ax.scatter(
        f0_mesh.ravel(),
        p_mesh.ravel(),
        z_pred_grid,
        c=z_pred_grid,
        cmap="viridis",
        s=10,
        alpha=0.45,
        label="Predicted grid points",
    )
    ax.scatter(
        actual["F0"].to_numpy(dtype=float),
        actual["P"].to_numpy(dtype=float),
        z_pred_at_actual,
        color="blue",
        s=28,
        alpha=0.9,
        label="Predicted at actual points",
    )
    ax.set_xlim(f0_min, f0_max)
    ax.set_ylim(p_min, p_max)
    ax.set_xlabel("F0")
    ax.set_ylabel("P")
    ax.set_zlabel("Gylis")
    p_mode = "measured P levels" if use_measured_p_levels else "uniform P grid"
    ax.set_title(f"Gylis predicted grid for N={n_value:g} (Matern 2.5 ARD, {p_mode})")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    rmse_points = float(np.sqrt(np.mean((z_actual - z_pred_at_actual) ** 2)))
    print(f"Actual points plotted: {len(actual)}")
    print(f"Predicted grid points plotted: {len(z_pred_grid)}")
    print(
        f"Actual ranges for N={n_value:g}: "
        f"P [{float(actual['P'].min()):.6g}, {float(actual['P'].max()):.6g}], "
        f"F0 [{float(actual['F0'].min()):.6g}, {float(actual['F0'].max()):.6g}]"
    )
    print(f"RMSE on displayed points: {rmse_points:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive 3D plot: F0 (x), P (y), Predicted Gylis (z) for fixed N."
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys_Nscan_3.xlsx")
    parser.add_argument("--n", type=float, default=1.0)
    parser.add_argument("--p-min", type=float, default=0.0)
    parser.add_argument("--p-max", type=float, default=25.0)
    parser.add_argument("--f0-min", type=float, default=0.0)
    parser.add_argument("--f0-max", type=float, default=10.0)
    parser.add_argument("--grid-points", type=int, default=50)
    parser.add_argument(
        "--uniform-p-grid",
        action="store_true",
        help="Use uniform P grid instead of measured P levels.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = load_data(file_path)
    preprocessor, model = train_model(df)

    make_plot_3d(
        df=df,
        preprocessor=preprocessor,
        model=model,
        n_value=float(args.n),
        p_min=float(args.p_min),
        p_max=float(args.p_max),
        f0_min=float(args.f0_min),
        f0_max=float(args.f0_max),
        grid_points=int(args.grid_points),
        use_measured_p_levels=not bool(args.uniform_p_grid),
    )


if __name__ == "__main__":
    main()
