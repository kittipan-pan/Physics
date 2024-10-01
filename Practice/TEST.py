import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
U = np.linspace(0, 2 * np.pi, 100)
V = np.linspace(0, np.pi, 100)

R = 1
THETA = np.pi/2
PHI = 0

X = R * np.outer(np.cos(U), np.sin(V))
Y = R * np.outer(np.sin(U), np.sin(V))
Z = R * np.outer(np.ones(np.size(U)), np.cos(V))

def init_point(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r*np.cos(theta)
    return x,y,z

def end_point(z, r, theta, phi):
    x = 0
    y = 0
    z = z
    return x,y,z

# Plot the surface
#ax.plot_surface(X, Y, Z, color='b')
ax.set_aspect('equal')
lim = 1.5
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

z = 2
ax.scatter3D(0,0,z, color='red')

p1 = init_point(r=R, theta=THETA, phi=PHI)
p2 = end_point(z=z, r=R, theta=THETA, phi=PHI)

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
print(p2)
plt.show()