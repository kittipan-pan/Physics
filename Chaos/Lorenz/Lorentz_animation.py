from Chaos.Lorenz.Lorentz_System import rk4_lorentz
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

IC = [1,0,0]
x0,y0,z0 = IC[0], IC[1], IC[2]
X, Y, Z = [x0], [y0], [z0]

h = 1e-3
t = np.arange(0,180,h)
for i in range(len(t)):
    xi,yi,zi = rk4_lorentz(h,[X[i], Y[i], Z[i]],rho=250,sig=15,beta=8/3)
    X.append(xi)
    Y.append(yi)
    Z.append(zi)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# for view rough path
skip = 100
def update(frame):
    ax.cla()
    current_frame = frame*skip

    ax.plot(X[:current_frame], Y[:current_frame], Z[:current_frame], lw=1)
    ax.scatter3D(X[current_frame - 1], Y[current_frame - 1], Z[current_frame - 1], s=10, color='red')

    ax.set_title(f't = {t[current_frame]:.3f}')

anim = FuncAnimation(fig=fig, func=update, frames=int(len(t)/skip), interval=100)
plt.show()
