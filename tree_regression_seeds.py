import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold


df = pd.read_excel("surikiuoti_duomenys.xlsx")


df = df[df["Nscan"] == 3].reset_index(drop=True)


X = df[["N","P","F0"]].to_numpy()
Y = df[["Ra","Gylis"]].to_numpy()

depth = np.arange(1, 30, 1)
leaf = np.arange(1, 30, 1)
rmse1 = []
rmse2 = []

xs = []
ys = []
state = 0
rmse_min_Ra = 1000
rmse_min_Gylis = 1000
rmse_gylis_list = []
rmse_Ra_list = []

plt.ion()
fig, ax = plt.subplots()
ra_scatter = ax.scatter([], [], color="red", label="Min_RMSE Ra")
gylis_scatter = ax.scatter([], [], color="blue", label="Min_RMSE Gylis")
ax.set_xlabel("Random Seed", fontsize=14)
ax.set_ylabel("RMSE", fontsize=14)
ax.set_xlim(-1, 100)
ax.legend(fontsize=14)
fig.canvas.draw()
fig.canvas.flush_events()




for k in range(10):
    kf = KFold(n_splits=5, shuffle=True, random_state=k)

    best_ra = np.inf
    best_g  = np.inf

    for i in depth:
        for j in leaf:
            fold_ra = []
            fold_g  = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = Y[train_idx], Y[test_idx]

                tree = DecisionTreeRegressor(max_depth=i, min_samples_leaf=j, random_state=0)
                tree.fit(X_train, y_train)
                pred = tree.predict(X_test)

                fold_ra.append(np.sqrt(mean_squared_error(y_test[:,0], pred[:,0])))
                fold_g.append(np.sqrt(mean_squared_error(y_test[:,1], pred[:,1])))

            mean_ra = np.mean(fold_ra)
            mean_g  = np.mean(fold_g)

            best_ra = min(best_ra, mean_ra)
            best_g  = min(best_g, mean_g)

    rmse_Ra_list.append(best_ra)
    rmse_gylis_list.append(best_g)
    seed_idx = np.arange(len(rmse_Ra_list))

    ra_scatter.set_offsets(np.c_[seed_idx, rmse_Ra_list])
    gylis_scatter.set_offsets(np.c_[seed_idx, rmse_gylis_list])

    current_min = min(min(rmse_Ra_list), min(rmse_gylis_list))
    current_max = max(max(rmse_Ra_list), max(rmse_gylis_list))
    pad = 0.05 * (current_max - current_min) if current_max > current_min else 0.1
    ax.set_ylim(current_min - pad, current_max + pad)
    plt.pause(0.001)

plt.ioff()
plt.show()



"""
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

sc1 = ax.scatter(xs, ys, rmse1, c="red", label="RMSE Ra", s=8)
#sc2 = ax.scatter(xs, ys, rmse2, c="blue", label="RMSE Gylis", marker="^", s=8)

ax.set_xlabel("depth")
ax.set_ylabel("leaf")
ax.set_zlabel("rmse")

ax.legend()

# Optional (one colorbar is usually enough; pick one)
fig.colorbar(sc1, ax=ax, label="RMSE Ra")
# fig.colorbar(sc2, ax=ax, label="RMSE Gylis")

plt.show()
"""
