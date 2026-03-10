import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold

from modeliu_testavimas import (
    GYLIS_L2,
    GYLIS_LEARNING_RATE,
    GYLIS_MAX_DEPTH,
    GYLIS_MAX_ITER,
    GYLIS_MAX_LEAF_NODES,
    GYLIS_MIN_SAMPLES_LEAF,
    GYLIS_TARGET,
    INPUT_COLUMNS,
    RA_ALPHA,
    RA_C,
    RA_LENGTH_SCALE,
    RA_NOISE,
    RA_NSCAN_VALUE,
    RA_TARGET,
    build_ra_preprocessor,
)

N_SPLITS = 5
CV_SEEDS = [0, 1, 2, 3, 4]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def apply_outlier_filter(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, dict]:
    q1 = float(df[target_col].quantile(0.25))
    q3 = float(df[target_col].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    keep_mask = df[target_col].between(lower, upper, inclusive="both")
    removed = int((~keep_mask).sum())
    filtered = df.loc[keep_mask].reset_index(drop=True)
    return filtered, {"lower": lower, "upper": upper, "removed": removed}


def get_ra_base_df(df: pd.DataFrame) -> pd.DataFrame:
    ra_df = df.copy()
    if "Nscan" in ra_df.columns:
        ra_df = ra_df[ra_df["Nscan"] == RA_NSCAN_VALUE].copy()
    ra_df = ra_df.dropna(subset=INPUT_COLUMNS + [RA_TARGET]).reset_index(drop=True)
    if ra_df.empty:
        raise ValueError("Nerasta tinkamu eiluciu Ra modeliui.")
    return ra_df


def get_gylis_base_df(df: pd.DataFrame) -> pd.DataFrame:
    gylis_df = df.dropna(subset=INPUT_COLUMNS + [GYLIS_TARGET]).reset_index(drop=True)
    if gylis_df.empty:
        raise ValueError("Nerasta tinkamu eiluciu Gylis modeliui.")
    return gylis_df


def evaluate_ra_strategy(
    base_df: pd.DataFrame,
    use_outlier_filter: bool,
    use_log_target: bool,
) -> dict:
    current_df = base_df.copy()
    outlier_info = {"lower": None, "upper": None, "removed": 0}
    if use_outlier_filter:
        current_df, outlier_info = apply_outlier_filter(current_df, RA_TARGET)
        if current_df.empty:
            return {
                "name": "",
                "use_outlier_filter": use_outlier_filter,
                "use_log_target": use_log_target,
                "rows_used": 0,
                "outliers_removed": outlier_info["removed"],
                "rmse_mean": np.inf,
                "rmse_fold_std": np.inf,
                "rmse_seed_std": np.inf,
                "outlier_lower": outlier_info["lower"],
                "outlier_upper": outlier_info["upper"],
            }

    X_df = current_df[INPUT_COLUMNS].copy()
    y = current_df[RA_TARGET].to_numpy(dtype=float)
    if use_log_target and np.any(y <= -1.0):
        return {
            "name": "",
            "use_outlier_filter": use_outlier_filter,
            "use_log_target": use_log_target,
            "rows_used": len(current_df),
            "outliers_removed": outlier_info["removed"],
            "rmse_mean": np.inf,
            "rmse_fold_std": np.inf,
            "rmse_seed_std": np.inf,
            "outlier_lower": outlier_info["lower"],
            "outlier_upper": outlier_info["upper"],
        }

    fold_scores = []
    seed_means = []

    for seed in CV_SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_scores = []

        for train_idx, test_idx in kf.split(X_df):
            X_train_df = X_df.iloc[train_idx]
            X_test_df = X_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            preprocessor = build_ra_preprocessor()
            X_train = preprocessor.fit_transform(X_train_df)
            X_test = preprocessor.transform(X_test_df)

            y_train_model = np.log1p(y_train) if use_log_target else y_train
            kernel = (
                ConstantKernel(constant_value=RA_C, constant_value_bounds="fixed")
                * Matern(length_scale=RA_LENGTH_SCALE, length_scale_bounds="fixed", nu=2.5)
                + WhiteKernel(noise_level=RA_NOISE, noise_level_bounds="fixed")
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=RA_ALPHA,
                optimizer=None,
                normalize_y=True,
                random_state=seed,
            )
            model.fit(X_train, y_train_model)
            y_pred_model = model.predict(X_test)
            y_pred = np.expm1(y_pred_model) if use_log_target else y_pred_model

            score = rmse(y_test, y_pred)
            fold_scores.append(score)
            seed_scores.append(score)

        seed_means.append(float(np.mean(seed_scores)))

    fold_arr = np.array(fold_scores, dtype=float)
    seed_arr = np.array(seed_means, dtype=float)
    return {
        "name": "",
        "use_outlier_filter": use_outlier_filter,
        "use_log_target": use_log_target,
        "rows_used": len(current_df),
        "outliers_removed": outlier_info["removed"],
        "rmse_mean": float(fold_arr.mean()),
        "rmse_fold_std": float(fold_arr.std()),
        "rmse_seed_std": float(seed_arr.std()),
        "outlier_lower": outlier_info["lower"],
        "outlier_upper": outlier_info["upper"],
    }


def evaluate_gylis_strategy(
    base_df: pd.DataFrame,
    use_outlier_filter: bool,
    use_log_target: bool,
) -> dict:
    current_df = base_df.copy()
    outlier_info = {"lower": None, "upper": None, "removed": 0}
    if use_outlier_filter:
        current_df, outlier_info = apply_outlier_filter(current_df, GYLIS_TARGET)
        if current_df.empty:
            return {
                "name": "",
                "use_outlier_filter": use_outlier_filter,
                "use_log_target": use_log_target,
                "rows_used": 0,
                "outliers_removed": outlier_info["removed"],
                "rmse_mean": np.inf,
                "rmse_fold_std": np.inf,
                "rmse_seed_std": np.inf,
                "outlier_lower": outlier_info["lower"],
                "outlier_upper": outlier_info["upper"],
            }

    X = current_df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = current_df[GYLIS_TARGET].to_numpy(dtype=float)
    if use_log_target and np.any(y <= -1.0):
        return {
            "name": "",
            "use_outlier_filter": use_outlier_filter,
            "use_log_target": use_log_target,
            "rows_used": len(current_df),
            "outliers_removed": outlier_info["removed"],
            "rmse_mean": np.inf,
            "rmse_fold_std": np.inf,
            "rmse_seed_std": np.inf,
            "outlier_lower": outlier_info["lower"],
            "outlier_upper": outlier_info["upper"],
        }

    fold_scores = []
    seed_means = []

    for seed in CV_SEEDS:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_train_model = np.log1p(y_train) if use_log_target else y_train
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=GYLIS_LEARNING_RATE,
                max_depth=GYLIS_MAX_DEPTH,
                max_iter=GYLIS_MAX_ITER,
                min_samples_leaf=GYLIS_MIN_SAMPLES_LEAF,
                l2_regularization=GYLIS_L2,
                max_leaf_nodes=GYLIS_MAX_LEAF_NODES,
                early_stopping=False,
                random_state=seed,
            )
            model.fit(X_train, y_train_model)
            y_pred_model = model.predict(X_test)
            y_pred = np.expm1(y_pred_model) if use_log_target else y_pred_model

            score = rmse(y_test, y_pred)
            fold_scores.append(score)
            seed_scores.append(score)

        seed_means.append(float(np.mean(seed_scores)))

    fold_arr = np.array(fold_scores, dtype=float)
    seed_arr = np.array(seed_means, dtype=float)
    return {
        "name": "",
        "use_outlier_filter": use_outlier_filter,
        "use_log_target": use_log_target,
        "rows_used": len(current_df),
        "outliers_removed": outlier_info["removed"],
        "rmse_mean": float(fold_arr.mean()),
        "rmse_fold_std": float(fold_arr.std()),
        "rmse_seed_std": float(seed_arr.std()),
        "outlier_lower": outlier_info["lower"],
        "outlier_upper": outlier_info["upper"],
    }


def choose_best_strategy(
    base_df: pd.DataFrame,
    target_name: str,
    evaluator,
) -> dict:
    strategies = [
        ("none", False, False),
        ("log_target_only", False, True),
        ("outlier_only", True, False),
        ("outlier_and_log_target", True, True),
    ]

    rows = []
    for name, use_outlier_filter, use_log_target in strategies:
        result = evaluator(
            base_df=base_df,
            use_outlier_filter=use_outlier_filter,
            use_log_target=use_log_target,
        )
        result["name"] = name
        rows.append(result)
        print(
            f"[{target_name}] Strategy {name}: mean={result['rmse_mean']:.6f}, "
            f"fold_std={result['rmse_fold_std']:.6f}, seed_std={result['rmse_seed_std']:.6f}, "
            f"rows={result['rows_used']}, removed={result['outliers_removed']}"
        )

    strategy_df = pd.DataFrame(rows)
    best = strategy_df.sort_values(["rmse_mean", "rmse_fold_std"], ascending=[True, True]).iloc[0].to_dict()
    print(f"\n[{target_name}] Chosen strategy:")
    print(best)
    return best


def train_ra_model_with_strategy(df: pd.DataFrame, strategy: dict):
    ra_df = get_ra_base_df(df)
    if bool(strategy["use_outlier_filter"]):
        ra_df, _ = apply_outlier_filter(ra_df, RA_TARGET)
    if ra_df.empty:
        raise ValueError("Po pasirinktos strategijos neliko Ra duomenu.")

    X_df = ra_df[INPUT_COLUMNS].copy()
    y = ra_df[RA_TARGET].to_numpy(dtype=float)
    y_model = np.log1p(y) if bool(strategy["use_log_target"]) else y

    preprocessor = build_ra_preprocessor()
    X = preprocessor.fit_transform(X_df)

    kernel = (
        ConstantKernel(constant_value=RA_C, constant_value_bounds="fixed")
        * Matern(length_scale=RA_LENGTH_SCALE, length_scale_bounds="fixed", nu=2.5)
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


def train_gylis_model_with_strategy(df: pd.DataFrame, strategy: dict):
    gylis_df = get_gylis_base_df(df)
    if bool(strategy["use_outlier_filter"]):
        gylis_df, _ = apply_outlier_filter(gylis_df, GYLIS_TARGET)
    if gylis_df.empty:
        raise ValueError("Po pasirinktos strategijos neliko Gylis duomenu.")

    X = gylis_df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = gylis_df[GYLIS_TARGET].to_numpy(dtype=float)
    y_model = np.log1p(y) if bool(strategy["use_log_target"]) else y

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
    return model


def predict_one_row(
    n_value: float,
    p_value: float,
    f0_value: float,
    ra_preprocessor,
    ra_model,
    gylis_model,
    ra_use_log_target: bool,
    gylis_use_log_target: bool,
) -> tuple[float, float]:
    x_df = pd.DataFrame([{"N": n_value, "P": p_value, "F0": f0_value}], columns=INPUT_COLUMNS)

    x_ra = ra_preprocessor.transform(x_df)
    ra_pred_model = float(ra_model.predict(x_ra)[0])
    ra_pred = float(np.expm1(ra_pred_model)) if ra_use_log_target else ra_pred_model

    x_gylis = x_df.to_numpy(dtype=float)
    gylis_pred_model = float(gylis_model.predict(x_gylis)[0])
    gylis_pred = float(np.expm1(gylis_pred_model)) if gylis_use_log_target else gylis_pred_model

    return ra_pred, gylis_pred


def build_predictions(
    df: pd.DataFrame,
    ra_preprocessor,
    ra_model,
    gylis_model,
    ra_use_log_target: bool,
    gylis_use_log_target: bool,
) -> tuple[list[float], list[float], int]:
    pred_ra = []
    pred_gylis = []
    skipped = 0

    for _, row in df.iterrows():
        if row[INPUT_COLUMNS].isna().any():
            pred_ra.append(np.nan)
            pred_gylis.append(np.nan)
            skipped += 1
            continue

        n_value = float(row["N"])
        p_value = float(row["P"])
        f0_value = float(row["F0"])

        if p_value <= -1.0 or f0_value <= -1.0:
            pred_ra.append(np.nan)
            pred_gylis.append(np.nan)
            skipped += 1
            continue

        ra_value, gylis_value = predict_one_row(
            n_value=n_value,
            p_value=p_value,
            f0_value=f0_value,
            ra_preprocessor=ra_preprocessor,
            ra_model=ra_model,
            gylis_model=gylis_model,
            ra_use_log_target=ra_use_log_target,
            gylis_use_log_target=gylis_use_log_target,
        )
        pred_ra.append(float(ra_value))
        pred_gylis.append(float(gylis_value))

    return pred_ra, pred_gylis, skipped


def write_predictions_to_excel(file_path: Path, pred_ra: list[float], pred_gylis: list[float]) -> None:
    wb = load_workbook(file_path)
    ws = wb.active

    ws.cell(row=1, column=8, value="Pred_Ra")
    ws.cell(row=1, column=9, value="Pred_Gylis")

    for excel_row, (ra_value, gylis_value) in enumerate(zip(pred_ra, pred_gylis), start=2):
        ws.cell(row=excel_row, column=8, value=None if pd.isna(ra_value) else float(ra_value))
        ws.cell(row=excel_row, column=9, value=None if pd.isna(gylis_value) else float(gylis_value))

    wb.save(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apmoko Ra/Gylis modelius su auto strategijos parinkimu (outlier/log), "
            "isveda RMSE ir iraso prognozes i H/I stulpelius."
        )
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys_Nscan_3.xlsx")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Neraso i Excel faila, tik paskaiciuoja strategijas/RMSE ir prognozes.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"Nerastas failas: {file_path}")

    df = pd.read_excel(file_path)
    required_cols = INPUT_COLUMNS + [RA_TARGET, GYLIS_TARGET]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Faile truksta stulpeliu: {missing}")

    print("Parenkamos strategijos pagal 5 seeds x 5-fold CV...")
    ra_base_df = get_ra_base_df(df)
    gylis_base_df = get_gylis_base_df(df)

    ra_strategy = choose_best_strategy(ra_base_df, target_name=RA_TARGET, evaluator=evaluate_ra_strategy)
    gylis_strategy = choose_best_strategy(gylis_base_df, target_name=GYLIS_TARGET, evaluator=evaluate_gylis_strategy)

    print("\nFinal RMSE summary:")
    print(
        f"Ra    | mean={ra_strategy['rmse_mean']:.6f}, fold_std={ra_strategy['rmse_fold_std']:.6f}, "
        f"seed_std={ra_strategy['rmse_seed_std']:.6f}"
    )
    print(
        f"Gylis | mean={gylis_strategy['rmse_mean']:.6f}, fold_std={gylis_strategy['rmse_fold_std']:.6f}, "
        f"seed_std={gylis_strategy['rmse_seed_std']:.6f}"
    )

    print("\nTreniruojami galutiniai modeliai su parinktomis strategijomis...")
    ra_preprocessor, ra_model = train_ra_model_with_strategy(df, strategy=ra_strategy)
    gylis_model = train_gylis_model_with_strategy(df, strategy=gylis_strategy)
    print("Modeliai istreniruoti.")

    pred_ra, pred_gylis, skipped = build_predictions(
        df=df,
        ra_preprocessor=ra_preprocessor,
        ra_model=ra_model,
        gylis_model=gylis_model,
        ra_use_log_target=bool(ra_strategy["use_log_target"]),
        gylis_use_log_target=bool(gylis_strategy["use_log_target"]),
    )
    print(f"Eiluciu viso: {len(df)}")
    print(f"Eiluciu praleista (truksta/neteisingi input): {skipped}")

    if args.dry_run:
        print("Dry-run: i Excel faila nerasoma.")
        return

    write_predictions_to_excel(file_path=file_path, pred_ra=pred_ra, pred_gylis=pred_gylis)
    print(f"Baigta. Prognozuotas Ra/Gylis irasyti i {file_path} H ir I stulpelius.")


if __name__ == "__main__":
    main()
