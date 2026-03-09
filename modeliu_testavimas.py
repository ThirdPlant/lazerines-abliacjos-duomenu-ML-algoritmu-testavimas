import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

INPUT_COLUMNS = ["N", "P", "F0"]
RA_TARGET = "Ra"
GYLIS_TARGET = "Gylis"
RA_NSCAN_VALUE = 3

# Ra model (Gaussian Process + Matern 2.5) hyperparameters
RA_C = 6.7021
RA_LENGTH_SCALE = 20.3326
RA_NOISE = 3.19913e-07
RA_ALPHA = 1.54732e-07

# Gylis model (HistGradientBoostingRegressor) hyperparameters
GYLIS_LEARNING_RATE = 0.215907
GYLIS_MAX_DEPTH = 11
GYLIS_MAX_ITER = 852
GYLIS_MIN_SAMPLES_LEAF = 29
GYLIS_L2 = 3.73731e-10
GYLIS_MAX_LEAF_NODES = 149


def build_ra_preprocessor() -> ColumnTransformer:
    pf_pipeline = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("n_scale", StandardScaler(), ["N"]),
            ("pf_log_scale", pf_pipeline, ["P", "F0"]),
        ],
        remainder="drop",
    )


def filter_ra_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1 = float(df[RA_TARGET].quantile(0.25))
    q3 = float(df[RA_TARGET].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    return df[df[RA_TARGET].between(lower, upper, inclusive="both")].reset_index(drop=True)


def train_ra_model(df: pd.DataFrame) -> tuple[ColumnTransformer, GaussianProcessRegressor]:
    ra_df = df[df["Nscan"] == RA_NSCAN_VALUE].copy()
    ra_df = ra_df.dropna(subset=INPUT_COLUMNS + [RA_TARGET]).reset_index(drop=True)
    if ra_df.empty:
        raise ValueError(f"Nerasta eiluciu su Nscan == {RA_NSCAN_VALUE} Ra modeliui.")

    ra_df = filter_ra_outliers(ra_df)
    if ra_df.empty:
        raise ValueError("Po Ra outlier filtravimo neliko duomenu.")

    X_ra_df = ra_df[INPUT_COLUMNS].copy()
    y_ra = ra_df[RA_TARGET].to_numpy(dtype=float)
    y_ra_model = np.log1p(y_ra)

    preprocessor = build_ra_preprocessor()
    X_ra = preprocessor.fit_transform(X_ra_df)

    kernel = (
        ConstantKernel(constant_value=RA_C, constant_value_bounds="fixed")
        * Matern(length_scale=RA_LENGTH_SCALE, length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=RA_NOISE, noise_level_bounds="fixed")
    )
    ra_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=RA_ALPHA,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    ra_model.fit(X_ra, y_ra_model)
    return preprocessor, ra_model


def train_gylis_model(df: pd.DataFrame) -> HistGradientBoostingRegressor:
    gylis_df = df.dropna(subset=INPUT_COLUMNS + [GYLIS_TARGET]).reset_index(drop=True)
    if gylis_df.empty:
        raise ValueError("Nerasta eiluciu Gylis modeliui.")

    X_gylis = gylis_df[INPUT_COLUMNS].to_numpy(dtype=float)
    y_gylis = gylis_df[GYLIS_TARGET].to_numpy(dtype=float)
    y_gylis_model = np.log1p(y_gylis)

    gylis_model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=GYLIS_LEARNING_RATE,
        max_depth=GYLIS_MAX_DEPTH,
        max_iter=GYLIS_MAX_ITER,
        min_samples_leaf=GYLIS_MIN_SAMPLES_LEAF,
        l2_regularization=GYLIS_L2,
        max_leaf_nodes=GYLIS_MAX_LEAF_NODES,
        early_stopping=False,
        random_state=0,
    )
    gylis_model.fit(X_gylis, y_gylis_model)
    return gylis_model


def parse_float_input(prompt: str) -> float:
    value_str = input(prompt).strip().replace(",", ".")
    return float(value_str)


def predict_one(
    n_value: float,
    p_value: float,
    f0_value: float,
    ra_preprocessor: ColumnTransformer,
    ra_model: GaussianProcessRegressor,
    gylis_model: HistGradientBoostingRegressor,
) -> tuple[float, float]:
    if p_value <= -1.0 or f0_value <= -1.0:
        raise ValueError("P ir F0 turi buti > -1 (naudojama log1p transformacija).")

    x_df = pd.DataFrame(
        [{"N": float(n_value), "P": float(p_value), "F0": float(f0_value)}],
        columns=INPUT_COLUMNS,
    )

    x_ra = ra_preprocessor.transform(x_df)
    ra_pred = float(np.expm1(ra_model.predict(x_ra)[0]))

    x_gylis = x_df.to_numpy(dtype=float)
    gylis_pred = float(np.expm1(gylis_model.predict(x_gylis)[0]))
    return ra_pred, gylis_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apmoko Ra ir Gylis modelius ir prognozuoja pagal ivestus N, P, F0."
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys.xlsx")
    args = parser.parse_args()

    df = pd.read_excel(args.file)
    required_columns = INPUT_COLUMNS + ["Nscan", RA_TARGET, GYLIS_TARGET]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faile truksta stulpeliu: {missing}")

    print("Treniruojami modeliai...")
    ra_preprocessor, ra_model = train_ra_model(df)
    gylis_model = train_gylis_model(df)
    print("Modeliai istreniruoti.")
    print("Iveskite N, P, F0 reiksmes prognozei.")

    n_value = parse_float_input("N: ")
    p_value = parse_float_input("P: ")
    f0_value = parse_float_input("F0: ")

    ra_pred, gylis_pred = predict_one(
        n_value=n_value,
        p_value=p_value,
        f0_value=f0_value,
        ra_preprocessor=ra_preprocessor,
        ra_model=ra_model,
        gylis_model=gylis_model,
    )

    print(f"Prognozuotas siurkstumas Ra: {ra_pred:.6f}")
    print(f"Prognozuotas ertmes gylis Gylis: {gylis_pred:.6f}")


if __name__ == "__main__":
    main()
