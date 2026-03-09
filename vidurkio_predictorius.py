import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.model_selection import KFold


df = pd.read_excel("surikiuoti_duomenys.xlsx")


df = df[df["Nscan"] == 3]
df = df[df["Nscan"] == 3].reset_index(drop=True)

print(df)

kf = KFold(n_splits=5, shuffle=True, random_state=0)

RMSE_Ra_list = []
RMSE_Gylis_list = []






for train_idx, test_idx in kf.split(df):
    training = df.iloc[train_idx]
    testing = df.iloc[test_idx]
    mean_Ra = training["Ra"].mean()
    RMSE_Ra = np.sqrt(np.mean((testing["Ra"]-mean_Ra)**2))
    mean_Gylis = training["Gylis"].mean()
    RMSE_Gylis = np.sqrt(np.mean((testing["Gylis"]-mean_Gylis)**2))
    RMSE_Ra_list.append(RMSE_Ra)
    RMSE_Gylis_list.append(RMSE_Gylis)
    print(RMSE_Ra, RMSE_Gylis)

print("Mean RMSE Ra:", np.mean(RMSE_Ra_list))
print("Mean RMSE Gylis:", np.mean(RMSE_Gylis_list))
print("STD RMSE Ra:", np.std(RMSE_Ra_list))
print("STD RMSE Gylis:", np.std(RMSE_Gylis_list))






"""
fig, ax = plt.subplots()

F0 =df["F0"]
norm = mcolors.LogNorm(vmin=F0[F0>0].min(), vmax=F0.max())

sc = ax.scatter(df["Gylis"], df["Ra"], c = df["F0"], cmap = "viridis", norm=norm, s=8)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("F0")
ax.set_xlabel("Gylis")
ax.set_ylabel("Ra")

plt.show()
"""