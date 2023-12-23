import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
import math
from scipy.integrate import odeint

def odesys(y, t, m1, m2, R, l, M0, gama, k, g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = (2 * m1 + m2) * R ** 2
    a12 = m2 * R * (R ** 2 - l ** 2) ** 0.5 * np.cos(y[1] - y[0])
    a21 = (R ** 2 - l ** 2) ** 0.5 * R * np.cos(y[1] - y[0])
    a22 = R ** 2 - 2 / 3 * l ** 2

    b1 = M0 * np.sin(gama * t) - k * y[2] + m2 * R * (R ** 2 - l ** 2) ** 0.5 * y[3] ** 2 * np.sin(y[1] - y[0]) - (m1 + m2) * g * R * np.sin(y[0])
    b2 = -(R ** 2 - l ** 2) ** 0.5 * (g * np.sin(y[1]) + R * y[2] ** 2 * np.sin(y[1] - y[0]))

    dy[2] = (b1*a22-b2*a12)/(a11*a22-a12*a21)
    dy[3] = (b2*a11-b1*a21)/(a11*a22-a12*a21)

    return dy

m1 = 2
m2 = 1
R = 0.5
l = 0.25
M0 = 15
gama = 3 * np.pi / 2
k = 10
g = -9.8

steps = 1000
t_fin = 10
omega = 5      # скорость кольца
omega_2 = 4    # скорость колебаний стержня
delta = 2.6

t = np.linspace(0, 10, steps)

phi0 = -np.pi
alpha0 = np.pi / 6
dphi0 = 0
dalpha0 = 0
y0 = [phi0, alpha0, dphi0, dalpha0]

Y = odeint(odesys, y0, t, (m1, m2, R, l, M0, gama, k, g))
phi = Y[:, 0]
alpha = Y[:, 1]
dphi = Y[:, 2]
dalpha = Y[:, 3]
ddphi = [odesys(y, t, m1, m2, R, l, M0, gama, k, g)[2] for y, t in zip(Y, t)]
ddalpha = [odesys(y, t, m1, m2, R, l, M0, gama, k, g)[3] for y, t in zip(Y, t)]

fig_for_graphs = plt.figure(figsize=[13, 7])
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, phi, color='Blue')
ax_for_graphs.set_title("phi(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, alpha, color='Green')
ax_for_graphs.set_title("alpha(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

Nx = -(m1 + m2) * R * (ddphi * np.cos(phi) - dphi ** 2 * np.sin(phi)) - m2 * (R ** 2 - l ** 2) ** 0.5 * (ddalpha * np.cos(alpha) - dalpha ** 2 * np.sin(alpha))
Ny = - (m1 + m2) * (R * (ddphi * np.sin(phi) + dphi ** 2 * np.cos(phi)) + g) - m2 * (R ** 2 - l ** 2) ** 0.5 * (ddalpha * np.sin(alpha) + dalpha ** 2 * np.cos(alpha))
N = (Nx ** 2 + Ny ** 2) ** 0.5
ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, N, color='Black')
ax_for_graphs.set_title("N(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

Radius = 0.6 # радиус кольца
Rad_ax = 0.05 # радиус колечка к которому крепится
lenght = 0.6 # длина стержня

ax_X = Rad_ax * np.sin(phi)
ax_Y = Rad_ax * np.cos(phi) # колечко к которому крепится

X_trajectory = Radius * np.sin(phi)        #траектория кольца по X  и центр кольца
Y_trajectory = Radius * np.cos(phi)        #траектория кольца по Y  и центр кольца

psi = np.linspace(0, 2 * np.pi, steps + 1)
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