import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
import math
# 23 вариант
steps = 1000
omega = 2      # скорость кольца
omega_2 = 1    # скорость колебаний стержня
delta = 2.6

t = np.linspace(0, 10, steps)
phi = omega * t # угол между осью OX и радиусом окружности отложенным от центра к точке крепления
alpha = np.cos(omega_2 * np.pi * t) + delta # угол между осью OX и нормалью к стержню

Radius = 0.6 # радиус кольца
Rad_ax = 0.05 # радиус колечка к которому крепится
lenght = 0.6 # длина стержня

ax_X = Rad_ax * np.sin(phi)
ax_Y = Rad_ax * np.cos(phi) # колечко к которому крепится

X_trajectory = Radius * np.sin(phi)        #траектория кольца по X  и центр кольца
Y_trajectory = Radius * np.cos(phi)        #траектория кольца по Y  и центр кольца

psi = np.linspace(0,  2 * np.pi, steps + 1) # угол задающий окружность
X_Circle = Radius*np.sin(psi)
Y_Circle = Radius*np.cos(psi)

A_x = Radius*np.sin(alpha)
A_y = Radius*np.cos(alpha) # координата опорной точки стержня 

delta_angle = np.arccos((lenght**2 - 2 * Radius**2)/(-2*Radius**2))

B_x = Radius*np.sin(alpha + delta_angle)
B_y = Radius*np.cos(alpha + delta_angle) # координата опорной точки стержня 


X_Ground = [0, 0, -2]
Y_Ground = [-1.2, 0, 0]
Xarrow_X = [-2, -1.9]
Xarrow_Y = [0, 0.03]
Yarrow_X = [0, 0.03]
Yarrow_Y = [-1.2, -1.16]

fig = plt.figure(figsize=[15,15])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-1.4, 1.4], ylim=[-1.4, 1.4])
Line_OR = ax.plot([X_trajectory[0] + ax_X[0], X_trajectory[0] + ax_X[0]], [Y_trajectory[0] + ax_Y[0], Y_trajectory[0] + ax_Y[0]], color = 'red', linestyle = '-.')[0]

ax.plot(X_Ground, Y_Ground, color='black', linewidth=1, linestyle = '--')
ax.plot(Xarrow_X, Xarrow_Y, color='black', linewidth=1, linestyle = '--')[0]
ax.plot(Xarrow_X, np.negative(Xarrow_Y), color='black', linewidth=1, linestyle = '--')[0]
ax.plot(np.negative(Yarrow_X), Yarrow_Y, color='black', linewidth=1, linestyle = '--')[0]
ax.plot(Yarrow_X, Yarrow_Y, color='black', linewidth=1, linestyle = '--')[0]

Circle = ax.plot(X_trajectory[0] + ax_X[0] + X_Circle, Y_trajectory[0] + ax_Y[0] + Y_Circle, linewidth = 1)[0]
axi = ax.plot(ax_X, ax_Y, color='red', linewidth=2)[0]
point = ax.plot(0, 0, marker = 'o', color = 'red', markersize = 5)[0]
A = ax.plot(X_trajectory[0] + A_x[0] + ax_X[0], Y_trajectory[0] + A_y[0] + ax_Y[0], marker = 'o', color = 'blue', linewidth = '1')[0]
B = ax.plot(X_trajectory[0] + B_x[0] + ax_X[0], Y_trajectory[0] + B_x[0] + ax_Y[0], marker = 'o', color = 'blue', linewidth = '1')[0]
R_p = ax.plot(X_trajectory[0] + ax_X[0], Y_trajectory[0] + ax_Y[0], marker = 'o', color = 'red', linewidth = '1')[0]
Line_AB = ax.plot([A_x[0] + X_trajectory[0] + ax_X[0], A_y[0] + Y_trajectory[0] + ax_Y[0]], [X_trajectory[0] + B_x[0] + ax_X[0], Y_trajectory[0] + B_y[0] + ax_Y[0]], color = 'blue')[0]

def animation(i):
    Circle.set_data(X_trajectory[i] + ax_X[i] + X_Circle, Y_trajectory[i] + ax_Y[i] + Y_Circle)
    A.set_data(A_x[i] + X_trajectory[i] + ax_X[i], A_y[i] + Y_trajectory[i] + ax_Y[i])
    B.set_data(B_x[i] + X_trajectory[i] + ax_X[i], B_y[i] + Y_trajectory[i] + ax_Y[i])
    R_p.set_data(X_trajectory[i] + ax_X[i], Y_trajectory[i] + ax_Y[i])
    Line_AB.set_data([A_x[i] + X_trajectory[i] + ax_X[i], B_x[i] + X_trajectory[i] + ax_X[i]], [A_y[i] + Y_trajectory[i] + ax_Y[i], B_y[i] + Y_trajectory[i] + ax_Y[i]])
    Line_OR.set_data([ax_X[i], X_trajectory[i] + ax_X[i]], [ax_Y[i], Y_trajectory[i] + ax_Y[i]])
    return [Circle, axi, Line_AB, Line_OR, R_p]

anima = FuncAnimation(fig, animation, frames=steps, interval=10)

plt.show()