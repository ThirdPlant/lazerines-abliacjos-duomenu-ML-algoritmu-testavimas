import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


df = pd.read_excel("surikiuoti_duomenys.xlsx")

X = df[["N", "P", "F0"]].to_numpy(dtype=float)
Y = df[["Ra", "Gylis"]].to_numpy(dtype=float)

mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Y).any(axis=1)
X, Y = X[mask], Y[mask]

random_seeds = list(range(10))
rmse_ra_by_seed = []
rmse_gylis_by_seed = []

plt.ion()
fig, ax = plt.subplots()
ra_scatter = ax.scatter([], [], color="red", label="RMSE Ra")
gylis_scatter = ax.scatter([], [], color="blue", label="RMSE Gylis")
ax.set_xlabel("Random Seed", fontsize=14)
ax.set_ylabel("RMSE", fontsize=14)
ax.set_xlim(-1, 10)
ax.legend(fontsize=12)
fig.canvas.draw()
fig.canvas.flush_events()

for seed in random_seeds:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    rmse_ra_folds = []
    rmse_gylis_folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        rf = RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            max_features="sqrt",
        )
        model = MultiOutputRegressor(rf)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_test)

        rmse_ra = np.sqrt(mean_squared_error(Y_test[:, 0], Y_pred[:, 0]))
        rmse_gylis = np.sqrt(mean_squared_error(Y_test[:, 1], Y_pred[:, 1]))
        rmse_ra_folds.append(rmse_ra)
        rmse_gylis_folds.append(rmse_gylis)

        print(
            f"Seed {seed}, Fold {fold}: RMSE Ra={rmse_ra:.6f}, RMSE Gylis={rmse_gylis:.6f}"
        )

    mean_rmse_ra = float(np.mean(rmse_ra_folds))
    mean_rmse_gylis = float(np.mean(rmse_gylis_folds))
    rmse_ra_by_seed.append(mean_rmse_ra)
    rmse_gylis_by_seed.append(mean_rmse_gylis)

    x_vals = np.array(random_seeds[: len(rmse_ra_by_seed)])
    ra_scatter.set_offsets(np.c_[x_vals, rmse_ra_by_seed])
    gylis_scatter.set_offsets(np.c_[x_vals, rmse_gylis_by_seed])

    current_min = min(min(rmse_ra_by_seed), min(rmse_gylis_by_seed))
    current_max = max(max(rmse_ra_by_seed), max(rmse_gylis_by_seed))
    pad = 0.05 * (current_max - current_min) if current_max > current_min else 0.1
    ax.set_ylim(current_min - pad, current_max + pad)
    plt.pause(0.001)

    print(
        f"Seed {seed} complete: mean RMSE Ra={mean_rmse_ra:.6f}, "
        f"mean RMSE Gylis={mean_rmse_gylis:.6f}"
    )

plt.ioff()
plt.show()
