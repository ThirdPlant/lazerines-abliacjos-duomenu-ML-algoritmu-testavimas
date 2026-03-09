import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sklearn

df = pd.read_excel("matavimai metalo silimas.xlsx")
df2 = pd.read_excel("savitosios_siluma.xlsx")

print(df)



t_Al = df["t"].to_numpy(dtype=float)
y = df["T aliuminis"].to_numpy(dtype=float)

t_zal = df["t kas 1 min"]
t_zal = t_zal.dropna().to_numpy(dtype=float)*60
T_zal = df["T zalvaris"]
T_zal = T_zal.dropna().to_numpy(dtype=float)

t_Cu = df2["t (s)"].to_numpy(dtype=float)
T_Cu = df2["T_Cu"].to_numpy(dtype=float)

Ti = np.array([0, 100, 200, 300, 400])
ci = np.array([0.3810, 0.3935, 0.4082, 0.4220, 0.4359])

T_grid = np.linspace(50, 250, 300)
c1_T = np.interp(T_grid, Ti, ci)

def exp_model(x, a, b, c):
    return a*np.exp(-b*x) + c
def exp_deriv(t_Cu, a, b):
    return a*(-b)*np.e**(-b*t_Cu)
def linear_model(x, a, b):
    return a*x+b

p0 = (y.max(), 0.01, y.min())

def expon_y(x, y):
    params, _ = curve_fit(exp_model, x, y, p0=p0, maxfev=100000)
    a, b, c = params
    T_pred = exp_model(t_Al, a, b, c)
    T_pred_points = exp_model(x, a, b, c)
    r2 = 1- np.sum((y-T_pred_points)**2)/np.sum((y-y.mean())**2)
    return T_pred, a, b, c, r2

def linear_y(x, y):
    params, _ = curve_fit(linear_model, x, y, maxfev=100000)
    a, b = params
    T_pred = linear_model(t_Al, a, b)
    return T_pred, a, b

def c_pred(ro, a, b):
    return c1_pred*(89e-7 * exp_deriv(t_Al, a_Cu, b_Cu))/(ro*exp_deriv(t_Al, a, b))


def exponent(a, b, c, r2):
    return (fr"$T(t) \approx {a:.3f}\cdot e^{{-{b:.5f}x}}+{c:.3f}$"
            "\n" 
            fr"$R^{{2}} \approx {r2:.5f}$")

def dTdt_from_T(T, b, c_offset):
    return np.abs(b * (T - c_offset))   # |dT/dt| = b(T-c)

T_pred_Al, a, b, c, r2 = expon_y(t_Al, y)
T_pred_Cu, a_Cu, b_Cu, c_Cu, r2_Cu = expon_y(t_Cu, T_Cu)
T_pred_zal, a_zal, b_zal, c_zal, r2_zal = expon_y(t_zal, T_zal)
c1_pred, a1, b1 = linear_y(Ti, ci)






fig, ((ax1, ax22), (ax3, ax4)) = plt.subplots(2, 2)
fig2, ax33 = plt.subplots()
fig3, ax44 = plt.subplots()

ax1.scatter(df["t"], df["T aliuminis"], s=3, label="Al eksperimentiniai duomenys")
ax1.plot(t_Al, T_pred_Al, label = exponent(a, b, c, r2))
ax22.plot(t_Al, exp_deriv(t_Al, a, b), label = r"$dT_{Al}/{dt}$")

ax1.scatter(df2["t (s)"], df2["T_Cu"], s=3, label="Cu eksperimentiniai duomenys")
ax1.plot(t_Al, T_pred_Cu, label = exponent(a_Cu, b_Cu, c_Cu, r2_Cu))
ax22.plot(t_Al, exp_deriv(t_Al, a_Cu, b_Cu), label = r"$dT_{Cu}/{dt}$")

ax1.scatter(df["t kas 1 min"]*60, df["T zalvaris"], s=3, label="Zal eksperimentiniai duomenys")
ax1.plot(t_Al, T_pred_zal, label = exponent(a_zal, b_zal, c_zal, r2_zal))
ax22.plot(t_Al, exp_deriv(t_Al, a_zal, b_zal), label = r"$dT_{zal}/{dt}$")

ax1.set(xlabel=r"$t, s$", ylabel=r"$T, ^{\circ}C$")
ax22.set(xlabel = r"$t, s$", ylabel = r"$\frac{dT}{dt}$")


ax33.scatter(Ti, ci)
ax33.plot(t_Al, c1_pred)

ax33.set_xlim(0, 450)
ax33.set_ylim(0.37, 0.45)
ax33.set(xlabel=r"$T, ^{circ}C$", ylabel=r"$c, \frac{J}{g\cdot K}$")

Al_T_isvestine_nuo_T = dTdt_from_T(T_grid, b, c)
Cu_T_isvestine_nuo_T = dTdt_from_T(T_grid, b_Cu, c_Cu)
zal_T_isvestine_nuo_T = dTdt_from_T(T_grid, b_zal, c_zal)




"""
ax4.plot(T_pred_Al, c_pred(27e-7, a, b), label = "Aliuminis arba plienas")
ax4.plot(T_pred_zal, c_pred(86e-7, a_zal, b_zal), label = "Zalvaris")
ax4.plot(T_pred_Cu, c_pred(89e-7, a_Cu, b_Cu), label = "Varis")
"""

def c_pred_good(ro, isvestine_nuo_T):
    return c1_T*(89e-7 * Cu_T_isvestine_nuo_T)/(ro*isvestine_nuo_T)

ax44.plot(T_grid, c_pred_good(79e-7, Al_T_isvestine_nuo_T), label = r"$c_{Fe}(T)$")
ax44.plot(T_grid, c_pred_good(86e-7, zal_T_isvestine_nuo_T), label = r"$c_{zalvaris}(T)$")
ax44.plot(T_grid, c_pred_good(89e-7, Cu_T_isvestine_nuo_T), label = r"$c_{Cu}(T)$")


ax44.set(xlabel=fr"$T, ^{{\circ}}C$", ylabel=r"$c, \frac{J}{g\cdot K}$")

#c = c1_pred*(89e-7*(dT1))/(27e-7*(dT))

ax1.legend(fontsize=8)
ax22.legend(fontsize = 10)
ax33.legend()
ax44.legend()


plt.show()
