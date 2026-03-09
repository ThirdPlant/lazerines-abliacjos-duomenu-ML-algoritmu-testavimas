import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


df = pd.read_excel("surikiuoti_duomenys.xlsx")

#F0_1_2 = df.loc[(df["N"] == 1) & (df["P"] == 2), "F0"].iloc[4]

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

nscan3 = df.loc[df["Nscan"] == 3, ["Gylis", "Rate1"]]
nscan4 = df.loc[df["Nscan"] == 4, ["Gylis", "Rate1"]]
nscan5 = df.loc[df["Nscan"] == 5, ["Gylis", "Rate1"]]
nscan8 = df.loc[df["Nscan"] == 8, ["Gylis", "Rate1"]]
nscan14 = df.loc[df["Nscan"] == 14, ["Gylis", "Rate1"]]
nscan21 = df.loc[df["Nscan"] == 21, ["Gylis", "Rate1"]]

ax.scatter(nscan3["Gylis"], nscan3["Rate1"]*3, label="nscan = 3")
ax.scatter(nscan4["Gylis"], nscan4["Rate1"]*4, label="nscan = 4")
ax.scatter(nscan5["Gylis"], nscan5["Rate1"]*5, label="nscan = 5")
ax.scatter(nscan8["Gylis"], nscan8["Rate1"]*8, label="nscan = 8")
ax.scatter(nscan14["Gylis"], nscan14["Rate1"]*14, label="nscan = 14")
ax.scatter(nscan21["Gylis"], nscan21["Rate1"]*21, label="nscan = 21")






ax.set_xlabel("gylis", fontsize=20)
ax.set_ylabel("sparta", fontsize=20)
plt.legend()

plt.show()

