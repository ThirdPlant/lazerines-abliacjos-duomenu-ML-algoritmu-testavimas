import argparse

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

INPUT_COLUMNS = ["N", "P", "F0"]
TARGET_COLUMN = "Gylis"
N_SPLITS = 5
CV_SEEDS = [0, 1, 2, 3, 4]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    df = df.dropna(subset=INPUT_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows left after dropping missing values.")
    return df


def apply_outlier_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    df: pd.DataFrame,
    params: dict,
    cv_seeds: list[int],
    use_log_target: bool,
) -> tuple[float, float, float]:
    X = df[INPUT_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    fold_scores = []
    seed_means = []

    for seed in cv_seeds:
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        seed_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            y_train_model = np.log1p(y_train) if use_log_target else y_train

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=float(params["learning_rate"]),
                max_depth=int(params["max_depth"]),
                max_iter=int(params["max_iter"]),
                min_samples_leaf=int(params["min_samples_leaf"]),
                l2_regularization=float(params["l2_regularization"]),
                max_leaf_nodes=int(params["max_leaf_nodes"]),
                early_stopping=False,
                random_state=seed,
            )
            model.fit(X_train, y_train_model)
            y_pred_model = model.predict(X_test)
            y_pred = np.expm1(y_pred_model) if use_log_target else y_pred_model

            fold_rmse = rmse(y_test, y_pred)
            fold_scores.append(fold_rmse)
            seed_scores.append(fold_rmse)

        seed_means.append(float(np.mean(seed_scores)))

    fold_arr = np.array(fold_scores, dtype=float)
    seed_arr = np.array(seed_means, dtype=float)
    return float(fold_arr.mean()), float(fold_arr.std()), float(seed_arr.std())


def choose_preprocessing_strategy(df: pd.DataFrame, cv_seeds: list[int]) -> dict:
    strategies = [
        {"name": "none", "use_outlier_filter": False, "use_log_target": False},
        {"name": "log_target_only", "use_outlier_filter": False, "use_log_target": True},
    ]

    baseline_params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 500,
        "min_samples_leaf": 10,
        "l2_regularization": 0.0,
        "max_leaf_nodes": 63,
    }

    rows = []
    for strategy in strategies:
        current_df = df.copy()
        outlier_info = {"lower": None, "upper": None, "removed": 0}
        if strategy["use_outlier_filter"]:
            current_df, outlier_info = apply_outlier_filter(current_df)

        mean_rmse, fold_std, seed_std = evaluate_candidate(
            df=current_df,
            params=baseline_params,
            cv_seeds=cv_seeds,
            use_log_target=strategy["use_log_target"],
        )

        row = {
            "name": strategy["name"],
            "use_outlier_filter": strategy["use_outlier_filter"],
            "use_log_target": strategy["use_log_target"],
            "rows_used": len(current_df),
            "outliers_removed": outlier_info["removed"],
            "rmse_mean": mean_rmse,
            "rmse_fold_std": fold_std,
            "rmse_seed_std": seed_std,
            "outlier_lower": outlier_info["lower"],
            "outlier_upper": outlier_info["upper"],
        }
        rows.append(row)
        print(
            f"Strategy {row['name']}: mean={mean_rmse:.6f}, fold_std={fold_std:.6f}, "
            f"seed_std={seed_std:.6f}, rows={row['rows_used']}, removed={row['outliers_removed']}"
        )

    results = pd.DataFrame(rows)
    best = results.sort_values(["rmse_mean", "rmse_fold_std"], ascending=[True, True]).iloc[0].to_dict()
    print("\nChosen preprocessing strategy:")
    print(best)
    return best


def summarize_trials(results: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    work = results.copy()
    work["rank_mean"] = work["rmse_mean"].rank(method="min")
    work["rank_std"] = work["rmse_fold_std"].rank(method="min")
    work["rank_sum"] = work["rank_mean"] + work["rank_std"]

    best_mean = work.sort_values(["rmse_mean", "rmse_fold_std"], ascending=[True, True]).iloc[0]
    best_std = work.sort_values(["rmse_fold_std", "rmse_mean"], ascending=[True, True]).iloc[0]
    best_balance = work.sort_values(["rank_sum", "rmse_mean"], ascending=[True, True]).iloc[0]
    return best_mean, best_std, best_balance


def row_to_parameterization(row: pd.Series | dict) -> dict:
    return {
        "learning_rate": float(row["learning_rate"]),
        "max_depth": int(row["max_depth"]),
        "max_iter": int(row["max_iter"]),
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "l2_regularization": float(row["l2_regularization"]),
        "max_leaf_nodes": int(row["max_leaf_nodes"]),
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
    df: pd.DataFrame,
    cv_seeds: list[int],
    use_log_target: bool,
    max_trials: int | None,
) -> pd.DataFrame:
    ax_client = AxClient(random_seed=123, verbose_logging=False)
    ax_client.create_experiment(
        name="hgb_gylis_continuous",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [0.005, 0.3],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "max_depth",
                "type": "range",
                "bounds": [2, 14],
                "value_type": "int",
            },
            {
                "name": "max_iter",
                "type": "range",
                "bounds": [100, 2500],
                "value_type": "int",
            },
            {
                "name": "min_samples_leaf",
                "type": "range",
                "bounds": [1, 80],
                "value_type": "int",
            },
            {
                "name": "l2_regularization",
                "type": "range",
                "bounds": [1e-10, 20.0],
                "value_type": "float",
                "log_scale": True,
            },
            {
                "name": "max_leaf_nodes",
                "type": "range",
                "bounds": [15, 255],
                "value_type": "int",
            },
        ],
        objectives={"rmse_mean": ObjectiveProperties(minimize=True)},
        choose_generation_strategy_kwargs={"force_random_search": True},
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
            rmse_mean, rmse_fold_std, rmse_seed_std = evaluate_candidate(
                df=df,
                params=params,
                cv_seeds=cv_seeds,
                use_log_target=use_log_target,
            )
            ax_client.complete_trial(trial_index=trial_index, raw_data={"rmse_mean": rmse_mean})
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")
            break
        except Exception as exc:
            rmse_mean = 1e6
            rmse_fold_std = 1e6
            rmse_seed_std = 1e6
            print(f"Trial {trial_no} failed: {exc}")
            if params is None:
                continue

        row = {
            "trial": trial_no,
            "rmse_mean": rmse_mean,
            "rmse_fold_std": rmse_fold_std,
            "rmse_seed_std": rmse_seed_std,
            "learning_rate": float(params["learning_rate"]),
            "max_depth": int(params["max_depth"]),
            "max_iter": int(params["max_iter"]),
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "l2_regularization": float(params["l2_regularization"]),
            "max_leaf_nodes": int(params["max_leaf_nodes"]),
        }
        trial_rows.append(row)
        results = pd.DataFrame(trial_rows)

        best_mean, best_std, best_balance = summarize_trials(results)
        pred_current = get_surrogate_prediction(ax_client, row_to_parameterization(row))
        pred_best_mean = get_surrogate_prediction(ax_client, row_to_parameterization(best_mean))
        pred_best_balance = get_surrogate_prediction(ax_client, row_to_parameterization(best_balance))

        print(
            f"Trial {trial_no} | RMSE mean={rmse_mean:.6f}, fold_std={rmse_fold_std:.6f}, "
            f"seed_std={rmse_seed_std:.6f} | lr={row['learning_rate']:.6g}, "
            f"depth={row['max_depth']}, iter={row['max_iter']}, leaf={row['min_samples_leaf']}, "
            f"l2={row['l2_regularization']:.6g}, max_leaf_nodes={row['max_leaf_nodes']}"
        )

        if pred_current is not None:
            print(f"Surrogate(current): pred_mean={pred_current[0]:.6f}, pred_sem={pred_current[1]:.6f}")
        else:
            print("Surrogate(current): unavailable (likely Sobol warm-up before GP surrogate fit)")

        print(
            "Best RMSE so far: "
            f"mean={best_mean['rmse_mean']:.6f}, fold_std={best_mean['rmse_fold_std']:.6f}, "
            f"seed_std={best_mean['rmse_seed_std']:.6f}, params={{lr={best_mean['learning_rate']:.6g}, "
            f"depth={int(best_mean['max_depth'])}, iter={int(best_mean['max_iter'])}, "
            f"leaf={int(best_mean['min_samples_leaf'])}, l2={best_mean['l2_regularization']:.6g}, "
            f"max_leaf_nodes={int(best_mean['max_leaf_nodes'])}}}"
        )

        if pred_best_mean is not None:
            print(
                f"Surrogate(best RMSE): pred_mean={pred_best_mean[0]:.6f}, pred_sem={pred_best_mean[1]:.6f}"
            )
        else:
            print("Surrogate(best RMSE): unavailable (likely Sobol warm-up before GP surrogate fit)")

        print(
            "Best balance so far: "
            f"mean={best_balance['rmse_mean']:.6f}, fold_std={best_balance['rmse_fold_std']:.6f}, "
            f"seed_std={best_balance['rmse_seed_std']:.6f}, params={{lr={best_balance['learning_rate']:.6g}, "
            f"depth={int(best_balance['max_depth'])}, iter={int(best_balance['max_iter'])}, "
            f"leaf={int(best_balance['min_samples_leaf'])}, l2={best_balance['l2_regularization']:.6g}, "
            f"max_leaf_nodes={int(best_balance['max_leaf_nodes'])}}}"
        )

        if pred_best_balance is not None:
            print(
                f"Surrogate(best balance): pred_mean={pred_best_balance[0]:.6f}, "
                f"pred_sem={pred_best_balance[1]:.6f}"
            )
        else:
            print("Surrogate(best balance): unavailable (likely Sobol warm-up before GP surrogate fit)")

        print(
            "Best std so far: "
            f"mean={best_std['rmse_mean']:.6f}, fold_std={best_std['rmse_fold_std']:.6f}, "
            f"seed_std={best_std['rmse_seed_std']:.6f}, params={{lr={best_std['learning_rate']:.6g}, "
            f"depth={int(best_std['max_depth'])}, iter={int(best_std['max_iter'])}, "
            f"leaf={int(best_std['min_samples_leaf'])}, l2={best_std['l2_regularization']:.6g}, "
            f"max_leaf_nodes={int(best_std['max_leaf_nodes'])}}}\n"
        )

    return pd.DataFrame(trial_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuous Bayesian optimization for HistGradientBoostingRegressor "
            "predicting Gylis from N, P, F0."
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

    print(f"Configuration: seeds={CV_SEEDS}, folds={N_SPLITS}")
    print("Input features: N, P, F0 (N is numeric, not one-hot)")
    print("Target: Gylis")
    print("Candidate generation: fast random/Sobol search (no slow BoTorch phase)")
    if args.max_trials is None:
        print("Optimization mode: continuous (press Ctrl+C to stop)\n")
    else:
        print(f"Optimization mode: capped to {args.max_trials} trials\n")

    raw_df = load_data(args.file)
    print(f"Rows after basic filtering: {len(raw_df)}\n")

    strategy = choose_preprocessing_strategy(raw_df, cv_seeds=CV_SEEDS)
    final_df = raw_df.copy()
    print(f"\nOutlier filter disabled. rows_left={len(final_df)}")

    print(f"Using log-transform on {TARGET_COLUMN}: {bool(strategy['use_log_target'])}\n")

    results = bayesian_optimize(
        df=final_df,
        cv_seeds=CV_SEEDS,
        use_log_target=bool(strategy["use_log_target"]),
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
    print(
        results.sort_values(["rmse_mean", "rmse_fold_std"], ascending=[True, True])
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
