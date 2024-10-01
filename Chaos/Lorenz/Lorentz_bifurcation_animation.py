from Chaos.Lorenz.Lorentz_System import rk4_lorentz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

rho_list = np.linspace(0,100,100)
sig =  10
beta = 8/3

IC = [0.5,0.5,0.5]
x0,y0,z0 = IC[0], IC[1], IC[2]
h = 1e-2
t = np.arange(0,100,h)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
def update(frame):
    ax.cla()
    X, Y, Z = [x0], [y0], [z0]
    rho = rho_list[frame]
    for i in range(len(t)):
        xi, yi, zi = rk4_lorentz(h=h, ic=[X[i], Y[i], Z[i]], rho=rho, beta=beta, sig=sig)
        X.append(xi)
        Y.append(yi)
        Z.append(zi)

    ax.set_title(f'Rho = {rho:.2f}')
    ax.plot(X,Y,Z,lw=0.2)

anim = FuncAnimation(fig=fig, func=update, frames=len(rho_list), interval=100, repeat=False)
plt.show()