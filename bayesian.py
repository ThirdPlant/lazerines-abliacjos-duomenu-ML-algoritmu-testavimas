import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


def generated_function(x: np.ndarray) -> np.ndarray:
    return (
        2.2 * np.sin(0.9 * x + 0.4)
        + 1.8 * np.sin(2.1 * x - 1.2)
        + 1.4 * np.cos(3.4 * x + 0.3)
        + 0.9 * np.sin(5.2 * x)
        + 10.0 * np.exp(-(x / 2.0) ** 2)
        - 0.03 * x**2
    )


def fit_surrogate(x_observed: np.ndarray, y_observed: np.ndarray) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0) * Matern(length_scale=2.0, nu=2.5)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=False,
        optimizer=None,
    )
    gp.fit(x_observed.reshape(-1, 1), y_observed)
    return gp


def create_ax_client(lower_bound: float, upper_bound: float, seed: int = 7) -> AxClient:
    ax_client = AxClient(random_seed=seed, verbose_logging=False)
    ax_client.create_experiment(
        name="one_dimensional_bayesian_optimization",
        parameters=[
            {
                "name": "x",
                "type": "range",
                "bounds": [float(lower_bound), float(upper_bound)],
                "value_type": "float",
            }
        ],
        objectives={"objective": ObjectiveProperties(minimize=False)},
        is_test=True,
    )
    return ax_client


def register_observations_with_ax(
    ax_client: AxClient,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
) -> None:
    for x_value, y_value in zip(x_observed, y_observed):
        _, trial_index = ax_client.attach_trial(parameters={"x": float(x_value)})
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"objective": float(y_value)},
        )


def suggest_and_evaluate_one_with_ax(ax_client: AxClient) -> tuple[float, float]:
    parameters, trial_index = ax_client.get_next_trial()
    x_candidate = float(parameters["x"])
    y_candidate = float(generated_function(np.array([x_candidate]))[0])
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={"objective": y_candidate},
    )
    return x_candidate, y_candidate


def fit_and_predict(
    x_grid: np.ndarray,
    x_observed: np.ndarray,
    y_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sorted_idx = np.argsort(x_observed)
    x_sorted = x_observed[sorted_idx]
    y_sorted = y_observed[sorted_idx]
    gp = fit_surrogate(x_sorted, y_sorted)
    mu, sigma = gp.predict(x_grid.reshape(-1, 1), return_std=True)
    return mu, sigma


def main() -> None:
    x_grid = np.linspace(-12, 12, 1200)
    y_true = generated_function(x_grid)
    rng = np.random.default_rng(7)

    n_initial_points = 6
    x_observed = np.sort(rng.uniform(x_grid.min(), x_grid.max(), size=n_initial_points))
    x_mid = 0.5 * (x_observed[0] + x_observed[1])
    x_observed = np.sort(np.append(x_observed, x_mid))
    y_observed = generated_function(x_observed)

    ax_client = create_ax_client(
        lower_bound=float(x_grid.min()),
        upper_bound=float(x_grid.max()),
        seed=7,
    )
    register_observations_with_ax(ax_client, x_observed, y_observed)

    x_new = np.array([], dtype=float)
    y_new = np.array([], dtype=float)
    x_all = x_observed.copy()
    y_all = y_observed.copy()

    mu, sigma = fit_and_predict(x_grid, x_all, y_all)
    upper = mu + 2.0 * sigma
    lower = mu - 2.0 * sigma

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x_grid, y_true, linewidth=2, alpha=0.35, label="f(x)")
    mu_line, = ax.plot(x_grid, mu, linewidth=2, color="tab:blue", label=r"$\mu(x)$")
    uncertainty_band = ax.fill_between(
        x_grid,
        lower,
        upper,
        color="tab:blue",
        alpha=0.14,
        zorder=1,
        label=r"$\mu(x) \pm 2\sigma(x)$",
    )
    ax.scatter(
        x_observed,
        y_observed,
        s=70,
        color="tab:orange",
        zorder=3,
        label="Initial observed points",
    )
    acquisition_scatter = ax.scatter(
        x_new,
        y_new,
        s=90,
        marker="X",
        color="tab:red",
        zorder=4,
        label="Ax acquisition points",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Bayesian optimization progress (step 0)")
    plt.tight_layout()
    plt.show(block=False)

    update_interval_seconds = 1.0
    max_new_points = None  # Set to an int (for example, 20) to stop automatically.
    step = 0

    while plt.fignum_exists(fig.number):
        if max_new_points is not None and step >= max_new_points:
            break
        step += 1

        x_candidate, y_candidate = suggest_and_evaluate_one_with_ax(ax_client)
        x_new = np.append(x_new, x_candidate)
        y_new = np.append(y_new, y_candidate)
        x_all = np.append(x_all, x_candidate)
        y_all = np.append(y_all, y_candidate)

        mu, sigma = fit_and_predict(x_grid, x_all, y_all)
        upper = mu + 2.0 * sigma
        lower = mu - 2.0 * sigma

        mu_line.set_ydata(mu)
        uncertainty_band.remove()
        uncertainty_band = ax.fill_between(
            x_grid,
            lower,
            upper,
            color="tab:blue",
            alpha=0.14,
            zorder=1,
        )
        acquisition_scatter.set_offsets(np.column_stack((x_new, y_new)))
        ax.set_title(f"Bayesian optimization progress (step {step})")

        print(f"Step {step:02d}: Ax suggestion x = {x_candidate:.4f}, f(x) = {y_candidate:.4f}")
        fig.canvas.draw_idle()
        plt.pause(update_interval_seconds)

    if plt.fignum_exists(fig.number):
        plt.show()


if __name__ == "__main__":
    main()
