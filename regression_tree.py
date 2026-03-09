import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


df = pd.read_excel("surikiuoti_duomenys.xlsx")


df = df[df["Nscan"] == 3]
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

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=state
)




for i in depth:
    for j in leaf:
        # Simple regression tree
        tree = DecisionTreeRegressor(
        max_depth=i,          # control complexity
        min_samples_leaf=j,   # smooth leaves a bit
        random_state=0
        )
        tree.fit(X_train, y_train)

        pred = tree.predict(X_test)

        rmse_Ra    = np.sqrt(mean_squared_error(y_test[:,0], pred[:,0]))
        rmse_Gylis = np.sqrt(mean_squared_error(y_test[:,1], pred[:,1]))
        rmse1.append(rmse_Ra)
        rmse2.append(rmse_Gylis)

        xs.append(i)
        ys.append(j)

        print("RMSE:", rmse_Ra, rmse_Gylis)



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