import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_excel("surikiuoti_duomenys.xlsx")


df = df[df["Nscan"] == 3]
df = df[df["Nscan"] == 3].reset_index(drop=True)


X = df[["N","P","F0"]].to_numpy()
Y = df[["Ra","Gylis"]].to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=0)

rmse_folds = []   # kiekvienas elementas: [rmse_y1, rmse_y2]

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    rmse_each = np.sqrt(mean_squared_error(Y_test, Y_pred, multioutput="raw_values"))
    rmse_folds.append(rmse_each)

    print(f"Fold {fold}: RMSE Ra={rmse_each[0]:.4f}, RMSE Gylis={rmse_each[1]:.4f}")

rmse_folds = np.array(rmse_folds)  # shape (n_folds, 2)
print("Mean RMSE:", rmse_folds.mean(axis=0))
print("Std  RMSE:", rmse_folds.std(axis=0, ddof=1))