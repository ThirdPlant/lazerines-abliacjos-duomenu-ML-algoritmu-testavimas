import argparse

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from sklearn.compose import ColumnTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

INPUT_COLUMNS = ["N", "P", "F0", "Gylis"]
TARGET_COLUMN = "Ra"
N_SCAN_VALUE = 3
N_SPLITS = 5
CV_SEEDS = [0, 1, 2, 3, 4]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_preprocessor() -> ColumnTransformer:
    # N is kept numeric (no one-hot). P, F0, and Gylis are log-scaled due to skew.
    pf_pipe = Pipeline(
        [
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("n_scale", StandardScaler(), ["N"]),
            ("pf_log_scale", pf_pipe, ["P", "F0", "Gylis"]),
        ],
        remainder="drop",
    )


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df[df["Nscan"] == N_SCAN_VALUE].copy()
    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows available after filtering Nscan == {N_SCAN_VALUE}.")
    return df


def remove_definite_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # Strict outlier rule on target only: Ra outside [Q1 - 3*IQR, Q3 + 3*IQR].
    q1 = float(df[TARGET_COLUMN].quantile(0.25))
    q3 = float(df[TARGET_COLUMN].quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr

    keep_mask = df[TARGET_COLUMN].between(lower, upper, inclusive="both")
    removed = int((~keep_mask).sum())
    filtered = df.loc[keep_mask].reset_index(drop=True)
    return filtered, {"lower": lower, "upper": upper, "removed": removed}


def evaluate_candidate(
    X_df: pd.DataFrame,
    y: np.ndarray,
    params: dict,
    cv_seeds: list[int],
) -> tuple[float, float, float]:
    scores = []
    seed_means = []

    for seed in cv_seeds:
        seed_scores = []
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X_df):
            X_train_df = X_df.iloc[train_idx]
            X_test_df = X_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            preprocessor = build_preprocessor()
            X_train = preprocessor.fit_transform(X_train_df)
            X_test = preprocessor.transform(X_test_df)

            # Forced log-transform on target Ra.
            y_train_model = np.log1p(y_train)

            kernel = (
                ConstantKernel(
                    constant_value=float(params["c_value"]),
                    constant_value_bounds="fixed",
                )
                * Matern(
                    length_scale=float(params["length_scale"]),
                    length_scale_bounds="fixed",
                    nu=float(params["matern_nu"]),
                )
                + WhiteKernel(
                    noise_level=float(params["noise_level"]),
                    noise_level_bounds="fixed",
                )
            )

            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=float(params["alpha"]),
                optimizer=None,
                normalize_y=True,
                random_state=seed,
            )
            model.fit(X_train, y_train_model)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            fold_rmse = rmse(y_test, y_pred)
            scores.append(fold_rmse)
            seed_scores.append(fold_rmse)

        seed_means.append(float(np.mean(seed_scores)))

    arr = np.array(scores, dtype=float)
    seed_arr = np.array(seed_means, dtype=float)
    return float(arr.mean()), float(arr.std()), float(seed_arr.std())


def summarize_trials(results: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = results.copy()
    work["rank_mean"] = work["rmse_mean"].rank(method="min")
    work["rank_std"] = work["rmse_std"].rank(method="min")
    work["rank_sum"] = work["rank_mean"] + work["rank_std"]

    best_mean = work.sort_values(["rmse_mean", "rmse_std"], ascending=[True, True]).iloc[0]
    best_std = work.sort_values(["rmse_std", "rmse_mean"], ascending=[True, True]).iloc[0]
    best_balance = work.sort_values(["rank_sum", "rmse_mean"], ascending=[True, True]).iloc[0]
    return best_mean, best_std, best_balance


def row_to_parameterization(row: pd.Series | dict) -> dict:
    return {
        "c_value": float(row["c_value"]),
        "length_scale": float(row["length_scale"]),
        "noise_level": float(row["noise_level"]),
        "alpha": float(row["alpha"]),
        "matern_nu": float(row["matern_nu"]),
    }


def get_surrogate_prediction(ax_client: AxClient, parameterization: dict) -> tuple[float, float] | None:
    try:
        preds = ax_client.get_model_predictions_for_parameterizations(
            parameterizations=[parameterization],
            metric_names=["rmse_mean"],
        )
        if preds and "rmse_mean" in preds[0]:
            pred_mean, pred_sem = preds[0]["rmse_mean"]
            return float(pred_mean), float(pred_sem)
    except Exception:
        return None
    return None


def bayesian_optimize(
    X_df: pd.DataFrame,
    y: np.ndarray,
    cv_seeds: list[int],
    max_trials: int | None,
) -> pd.DataFrame:
    ax_client = AxClient(random_seed=123, verbose_logging=False)
    ax_client.create_experiment(
        name="gp_matern_ra_continuous",
        parameters=[
            {
                "name": "c_value",
                "type": "range",
                "bounds": [1e-3, 100.0],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "length_scale",
                "type": "range",
                "bounds": [1e-3, 100.0],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "noise_level",
                "type": "range",
                "bounds": [1e-8, 1e-1],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "alpha",
                "type": "range",
                "bounds": [1e-12, 1e-3],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "matern_nu",
                "type": "choice",
                "values": [1.5, 2.5],
                "value_type": "float",
                "is_ordered": True,
                "sort_values": True,
            },
        ],
        objectives={"rmse_mean": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={"num_initialization_trials": 5},
        is_test=True,
    )

    trial_rows = []
    trial_no = 0

    while True:
        if max_trials is not None and trial_no >= max_trials:
            break

        trial_no += 1
        params = None
        try:
            params, trial_index = ax_client.get_next_trial()
            rmse_mean, rmse_std, rmse_std_seed = evaluate_candidate(
                X_df=X_df,
                y=y,
                params=params,
                cv_seeds=cv_seeds,
            )
            ax_client.complete_trial(trial_index=trial_index, raw_data={"rmse_mean": rmse_mean})
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            break
        except Exception as exc:
            rmse_mean = 1e6
            rmse_std = 1e6
            rmse_std_seed = 1e6
            print(f"Trial {trial_no} failed: {exc}")
            if params is None:
                continue

        row = {
            "trial": trial_no,
            "rmse_mean": rmse_mean,
            "rmse_std": rmse_std,
            "rmse_std_seed": rmse_std_seed,
            "c_value": float(params["c_value"]),
            "length_scale": float(params["length_scale"]),
            "noise_level": float(params["noise_level"]),
            "alpha": float(params["alpha"]),
            "matern_nu": float(params["matern_nu"]),
        }
        trial_rows.append(row)
        results = pd.DataFrame(trial_rows)

        best_mean, best_std, best_balance = summarize_trials(results)
        pred_current = get_surrogate_prediction(ax_client, row_to_parameterization(row))
        pred_best_mean = get_surrogate_prediction(ax_client, row_to_parameterization(best_mean))
        pred_best_balance = get_surrogate_prediction(ax_client, row_to_parameterization(best_balance))

        print(
            f"Trial {trial_no} | RMSE mean={rmse_mean:.6f}, fold_std={rmse_std:.6f}, "
            f"seed_std={rmse_std_seed:.6f} | "
            f"C={row['c_value']:.6g}, ls={row['length_scale']:.6g}, "
            f"noise={row['noise_level']:.6g}, alpha={row['alpha']:.6g}, nu={row['matern_nu']}"
        )
        if pred_current is not None:
            print(
                f"Surrogate(current): pred_mean={pred_current[0]:.6f}, pred_sem={pred_current[1]:.6f}"
            )
        else:
            print("Surrogate(current): unavailable (likely Sobol warm-up before GP surrogate fit)")
        print(
            "Best RMSE so far: "
            f"mean={best_mean['rmse_mean']:.6f}, fold_std={best_mean['rmse_std']:.6f}, "
            f"seed_std={best_mean['rmse_std_seed']:.6f}, "
            f"params={{C={best_mean['c_value']:.6g}, ls={best_mean['length_scale']:.6g}, "
            f"noise={best_mean['noise_level']:.6g}, alpha={best_mean['alpha']:.6g}, "
            f"nu={best_mean['matern_nu']}}}"
        )
        if pred_best_mean is not None:
            print(
                f"Surrogate(best RMSE): pred_mean={pred_best_mean[0]:.6f}, pred_sem={pred_best_mean[1]:.6f}"
            )
        else:
            print("Surrogate(best RMSE): unavailable (likely Sobol warm-up before GP surrogate fit)")
        print(
            "Best balance so far: "
            f"mean={best_balance['rmse_mean']:.6f}, fold_std={best_balance['rmse_std']:.6f}, "
            f"seed_std={best_balance['rmse_std_seed']:.6f}, "
            f"params={{C={best_balance['c_value']:.6g}, ls={best_balance['length_scale']:.6g}, "
            f"noise={best_balance['noise_level']:.6g}, alpha={best_balance['alpha']:.6g}, "
            f"nu={best_balance['matern_nu']}}}"
        )
        if pred_best_balance is not None:
            print(
                f"Surrogate(best balance): pred_mean={pred_best_balance[0]:.6f}, pred_sem={pred_best_balance[1]:.6f}"
            )
        else:
            print("Surrogate(best balance): unavailable (likely Sobol warm-up before GP surrogate fit)")
        print(
            "Best std so far: "
            f"mean={best_std['rmse_mean']:.6f}, fold_std={best_std['rmse_std']:.6f}, "
            f"seed_std={best_std['rmse_std_seed']:.6f}, "
            f"params={{C={best_std['c_value']:.6g}, ls={best_std['length_scale']:.6g}, "
            f"noise={best_std['noise_level']:.6g}, alpha={best_std['alpha']:.6g}, "
            f"nu={best_std['matern_nu']}}}\n"
        )

    return pd.DataFrame(trial_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous Bayesian optimization for Ra with GP kernel "
            "C*Matern(nu in {1.5, 2.5}) + WhiteKernel, 5-fold CV over 5 seeds."
        )
    )
    parser.add_argument("--file", type=str, default="surikiuoti_duomenys.xlsx")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap for testing; default None means run until interrupted.",
    )
    args = parser.parse_args()

    print(f"Configuration: Nscan={N_SCAN_VALUE}, seeds={CV_SEEDS}, folds={N_SPLITS}")
    print("Input handling: N kept numeric (no one-hot), P/F0/Gylis log1p+scaled")
    print("Target handling: forced log1p transform on Ra")
    print("Kernel: C*Matern(nu={1.5, 2.5}) + WhiteKernel")
    if args.max_trials is None:
        print("Optimization mode: continuous (press Ctrl+C to stop)\n")
    else:
        print(f"Optimization mode: capped to {args.max_trials} trials\n")

    df = load_data(args.file)
    print(f"Rows after base filtering: {len(df)}")

    df, outlier_info = remove_definite_outliers(df)
    print(
        f"Outlier filter on Ra (3*IQR): removed={outlier_info['removed']}, "
        f"bounds=({outlier_info['lower']:.6f}, {outlier_info['upper']:.6f}), "
        f"rows_left={len(df)}\n"
    )

    X_df = df[INPUT_COLUMNS].copy()
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    results = bayesian_optimize(
        X_df=X_df,
        y=y,
        cv_seeds=CV_SEEDS,
        max_trials=args.max_trials,
    )

    if results.empty:
        print("No completed trials.")
        return

    best_mean, best_std, best_balance = summarize_trials(results)

    print("Final best by mean RMSE:")
    print(best_mean.to_dict())
    print("\nFinal best by std:")
    print(best_std.to_dict())
    print("\nFinal best balance:")
    print(best_balance.to_dict())
    print("\nTop 10 by mean RMSE:")
    print(results.sort_values(["rmse_mean", "rmse_std"], ascending=[True, True]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
