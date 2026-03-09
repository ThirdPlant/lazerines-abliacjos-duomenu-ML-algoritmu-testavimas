import numpy as np
import matplotlib.pyplot as plt

# Grid values
alphas = np.array([1, 10, 50, 100], dtype=float)
degrees = np.array([2, 3, 4, 5], dtype=float)

# Mean RMSE from your table (rows=alpha, cols=d)
rmse_ra = np.array([
    [0.133, 0.113, 0.109, 0.117],
    [0.133, 0.114, 0.111, 0.109],
    [0.134, 0.114, 0.112, 0.108],
    [0.136, 0.114, 0.112, 0.108],
], dtype=float)

rmse_gylis = np.array([
    [2.774, 2.483, 2.129, 2.483],
    [2.769, 2.545, 2.144, 2.483],
    [2.782, 2.649, 2.343, 2.450],
    [2.809, 2.714, 2.465, 2.434],
], dtype=float)

# Build scatter points
A, D = np.meshgrid(alphas, degrees, indexing="ij")  # shapes (4,4)
x_alpha = A.ravel()
x_d = D.ravel()
z_ra = rmse_ra.ravel()
z_g = rmse_gylis.ravel()

# --- Plot 1: RMSE Ra ---
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x_alpha, x_d, z_ra, c=z_ra)  # no explicit colors specified
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("d")
ax.set_zlabel("Mean RMSE (Ra)")
#ax.set_title("3D scatter: alpha vs d vs RMSE(Ra)")
fig.colorbar(sc, ax=ax, pad=0.1, label="RMSE(Ra)")
plt.show()

# --- Plot 2: RMSE Gylis ---
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x_alpha, x_d, z_g, c=z_g)
ax.set_xlabel(r"$\alpha$")
ax.set_ylabel("d")
ax.set_zlabel("Mean RMSE (Gylis)")
#ax.set_title("3D scatter: alpha vs d vs RMSE(Gylis)")
fig.colorbar(sc, ax=ax, pad=0.1, label="RMSE(Gylis)")
plt.show()