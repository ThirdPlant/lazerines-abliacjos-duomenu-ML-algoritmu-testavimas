import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["N", "P", "F0"]
TARGET_COLUMN = "Gylis"

# Fixed model from boosted_decision_tree.py "Best balance so far"
FEATURE_MODE = "scale_all"
USE_LOG_TARGET = True
GYLIS_LEARNING_RATE = 0.170511
GYLIS_MAX_DEPTH = 7
GYLIS_MAX_ITER = 1614
GYLIS_MIN_SAMPLES_LEAF = 30
GYLIS_L2 = 1.13107e-05
GYLIS_MAX_LEAF_NODES = 234


def build_preprocessor() -> ColumnTransformer:
    if FEATURE_MODE != "scale_all":
        raise ValueError(f"Unsupported FEATURE_MODE: {FEATURE_MODE}")
    return ColumnTransformer([("scale", StandardScaler(), INPUT_COLUMNS)], remainder="drop")


def train_gylis_model(df: pd.DataFrame) -> tuple[ColumnTransformer, HistGradientBoostingRegressor]:
    train_df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if train_df.empty:
        raise ValueError("No rows available to train Gylis model.")

    X_df = train_df[INPUT_COLUMNS].copy()
    y = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    if USE_LOG_TARGET and np.any(y <= -1.0):
        raise ValueError("Cannot use log1p target transform because Gylis contains values <= -1.")
    y_model = np.log1p(y) if USE_LOG_TARGET else y

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(X_df)

    model = HistGradientBoostingRegressor(
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
    model.fit(X, y_model)
    return preprocessor, model


def predict_gylis(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: HistGradientBoostingRegressor,
) -> tuple[list[float], int]:
    preds: list[float] = []
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


def write_to_column_h(file_path: Path, preds: list[float]) -> None:
    wb = load_workbook(file_path)
    ws = wb.active

    # Column H (8): predicted Gylis
    ws.cell(row=1, column=8, value="Pred_Gylis")
    for excel_row, value in enumerate(preds, start=2):
        ws.cell(row=excel_row, column=8, value=None if pd.isna(value) else float(value))

    wb.save(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train fixed Gylis HistGradientBoostingRegressor (best-balance params) "
            "and write predicted Gylis to column H in surikiuoti_duomenys_Nscan_3.xlsx."
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

    print("Training fixed Gylis model...")
    preprocessor, model = train_gylis_model(df)
    preds, skipped = predict_gylis(df, preprocessor, model)
    print(f"Rows total: {len(df)}")
    print(f"Rows skipped (missing inputs): {skipped}")
    print("Write target: column H (8)")
    print(
        "Config: "
        f"feature_mode={FEATURE_MODE}, log_target={USE_LOG_TARGET}, "
        f"lr={GYLIS_LEARNING_RATE}, depth={GYLIS_MAX_DEPTH}, iter={GYLIS_MAX_ITER}, "
        f"leaf={GYLIS_MIN_SAMPLES_LEAF}, l2={GYLIS_L2}, max_leaf_nodes={GYLIS_MAX_LEAF_NODES}"
    )

    if args.dry_run:
        print("Dry-run: no write performed.")
        return

    write_to_column_h(file_path=file_path, preds=preds)
    print(f"Done. Predicted Gylis written to column H in {file_path}.")


if __name__ == "__main__":
    main()
