import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["N", "P", "F0"]
TARGET_COLUMN = "Ra"

# Fixed model from your "best balance so far"
FEATURE_MODE = "scale_all"
USE_LOG_TARGET = True
RA_C = 0.001
RA_LENGTH_SCALE = 1.86182
RA_NOISE = 1.06964e-07
RA_ALPHA = 3.08983e-07
RA_NU = 2.5


def build_preprocessor() -> ColumnTransformer:
    if FEATURE_MODE != "scale_all":
        raise ValueError(f"Unsupported FEATURE_MODE: {FEATURE_MODE}")
    return ColumnTransformer([("scale", StandardScaler(), INPUT_COLUMNS)], remainder="drop")


def train_ra_model(df: pd.DataFrame) -> tuple[ColumnTransformer, GaussianProcessRegressor]:
    train_df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No rows available to train Ra model.")

    X_df = train_df[INPUT_COLUMNS].copy()
    y = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    if USE_LOG_TARGET and np.any(y <= -1.0):
        raise ValueError("Cannot use log1p target transform because Ra contains values <= -1.")
    y_model = np.log1p(y) if USE_LOG_TARGET else y

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_df)

    kernel = (
        ConstantKernel(constant_value=RA_C, constant_value_bounds="fixed")
        * Matern(length_scale=RA_LENGTH_SCALE, length_scale_bounds="fixed", nu=RA_NU)
        + WhiteKernel(noise_level=RA_NOISE, noise_level_bounds="fixed")
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=RA_ALPHA,
        optimizer=None,
        normalize_y=True,
        random_state=0,
    )
    model.fit(X, y_model)
    return preprocessor, model


def predict_ra(df: pd.DataFrame, preprocessor: ColumnTransformer, model: GaussianProcessRegressor) -> tuple[list[float], int]:
    preds = []
    skipped = 0

    for _, row in df.iterrows():
        if row[INPUT_COLUMNS].isna().any():
            preds.append(np.nan)
            skipped += 1
            continue

        x_df = pd.DataFrame(
            [{"N": float(row["N"]), "P": float(row["P"]), "F0": float(row["F0"])}],
            columns=INPUT_COLUMNS,
        )
        x = preprocessor.transform(x_df)
        pred_model = float(model.predict(x)[0])
        pred = float(np.expm1(pred_model)) if USE_LOG_TARGET else pred_model
        preds.append(pred)

    return preds, skipped


def write_to_column_3(file_path: Path, preds: list[float]) -> None:
    wb = load_workbook(file_path)
    ws = wb.active

    # Column C (3): overwrite with predicted Ra
    ws.cell(row=1, column=3, value="Pred_Ra")
    for excel_row, value in enumerate(preds, start=2):
        ws.cell(row=excel_row, column=3, value=None if pd.isna(value) else float(value))

    wb.save(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed Ra GP model (best-balance params) and write predicted Ra "
            "to column 3 (C) in surikiuoti_duomenys_Nscan_3.xlsx."
        )
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys_Nscan_3.xlsx")
    parser.add_argument("--dry-run", action="store_true", help="Do not write to Excel.")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path)
    missing = [col for col in INPUT_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Training fixed Ra model...")
    preprocessor, model = train_ra_model(df)
    preds, skipped = predict_ra(df, preprocessor, model)
    print(f"Rows total: {len(df)}")
    print(f"Rows skipped (missing inputs): {skipped}")
    print("Write target: column 3 (C)")

    if args.dry_run:
        print("Dry-run: no write performed.")
        return

    write_to_column_3(file_path=file_path, preds=preds)
    print(f"Done. Predicted Ra written to column C in {file_path}.")


if __name__ == "__main__":
    main()
