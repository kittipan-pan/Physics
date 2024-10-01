from Chaos.Multi_dimensions.Tools import RK4, Poincare_section, Cross_plane
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import  FuncAnimation

fig = plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

IC = [0.5,0.5,0.5]
# Equation
flows = ['y', '-x+y*z', '1-y**2']

cross_line = np.linspace(-2,2,100)
def update(frame):
    ax1.cla()
    ax2.cla()

    x, y, z = RK4.plot3d(IC, flows, [0, 200, 0.01], 'point')

    # Plot 3D graph
    ax1.plot(x,y,z, lw=0.2, alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Display Poincare section
    cross_section = ['x', cross_line[frame]]
    Poincare_section(data=[x, y, z], axis=cross_section, ax=ax2, opt='forward')
    point = Poincare_section(data=[x, y, z], axis=cross_section, opt=['data', 'forward'])
    Cross_plane(axis=cross_section, ax=ax1, data=point, size=2)

    ax2.set_xlim(-3,3)
    ax2.set_ylim(-3,3)

anim = FuncAnimation(fig=fig, func=update, frames=len(cross_line), interval=100, repeat=False)
plt.show()