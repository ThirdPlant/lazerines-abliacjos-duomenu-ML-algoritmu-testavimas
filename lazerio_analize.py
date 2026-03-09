import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


df = pd.read_excel("surikiuoti_duomenys.xlsx")

F0_1_2 = df.loc[(df["N"] == 1) & (df["P"] == 2), "F0"].iloc[4]

Gylis_P = df.loc[df["N"] == 1].groupby("P", as_index=False)["Gylis"].max()

print(Gylis_P)


fig, ax = plt.subplots()
#ax = fig.add_subplot(111, projection="3d")

"""
for N in range(9):
    N=N+1
    Gylis_P = df.loc[df["N"] == N].groupby("P", as_index=False)["Ra"].min()
    sc = ax.scatter(Gylis_P["P"], Gylis_P["Ra"], c=np.full(len(Gylis_P), N), cmap="coolwarm", vmin=1, vmax=9)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("N", fontsize=20)
ax.set_xlabel("P", fontsize=20)
ax.set_ylabel("Ra_min", fontsize=20)
"""
cmap = plt.cm.coolwarm
norm = mpl.colors.Normalize(vmin=1, vmax=9)

for N in range(1, 10):
    g = (df.loc[df["N"] == N]
           .groupby("P", as_index=False)["Ra"]
           .min()
           .sort_values("P"))          # important for lines
    ax.plot(g["P"], g["Ra"], color=cmap(norm(N)))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("N", fontsize=20)

ax.set_xlabel("P", fontsize=20)
ax.set_ylabel("Ra_min", fontsize=20)


plt.show()

