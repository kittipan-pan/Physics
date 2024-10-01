from Chaos.Multi_dimensions.Tools import RK4
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation

# Equations
flows = ['x*(1-x)*(1-x)', '1']

# Initial Conditions 1
IC1 = [0.1, 0]
x1, y1 = RK4.plot2d(IC1, flows, [0,20,0.1], ['cartesian', 'point'])

# Initial Conditions 2
IC2 = [0.5, np.pi/4]
x2, y2 = RK4.plot2d(IC2, flows, [0,20,0.1], ['cartesian', 'point'])

# Initial Conditions 3
IC3 = [1.2, np.pi/4]
x3, y3 = RK4.plot2d(IC3, flows, [0,5,0.1], ['cartesian', 'point'])

fig1 = plt.figure()
ax = fig1.add_subplot()
def update(frame):
    ax.cla()
    ax.plot(x1[:frame],y1[:frame])
    ax.plot(x2[:frame],y2[:frame])
    ax.plot(x3[:frame], y3[:frame])

    ax.scatter(x1[0], y1[0], color='red', s=10)
    ax.scatter(x2[0], y2[0], color='red', s=10)
    ax.scatter(x3[0], y3[0], color='red', s=10)

    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_aspect('equal')
    ax.set_title(r'$\dot{r} = -r \left ( 1-r \right ) \left ( 2-r \right ), \qquad \dot{\theta} = 1$')
    ax.grid()

ani = FuncAnimation(fig=fig1, func=update, frames=len(x1), interval=1)
plt.show()
