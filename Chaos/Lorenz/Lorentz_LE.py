import numpy as np
import matplotlib.pyplot as plt
from random import uniform as rand
from Lorentz_System import rk4_lorentz

# Initial Condition.
IC = [2,0,10]

# Lorentz constant parameters.
RHO = 28
SIG = 10
BETA = 8/3

time_step = 1e-3
time_stop = 50
t = np.arange(0, time_stop + time_step, time_step)

# Initial point.
x0,y0,z0 = IC[0], IC[1], IC[2]
X, Y, Z = [x0], [y0], [z0]

# Make a slightly different initial point going parallel to the original point.
# distance far from the origin is 0.1.
dr = np.array([rand(0,1),rand(0,1),rand(0,1)])
dr_size = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)
dr_unit = dr / dr_size

# decreasing distance to 0.1
d0 = 0.1
dr = dr_unit * d0

# Parallel initial point.
xp0, yp0, zp0 = x0+dr[0], y0+dr[1], z0+dr[2]
XP, YP, ZP = [xp0], [yp0], [zp0]

dis = [d0]
preLE = []
LE = []
tLE = []

reset = 100
for i in range(len(t) - 1):
    # Approximate line
    x_next, y_next, z_next = rk4_lorentz(h=time_step,ic=[X[i], Y[i], Z[i]],rho=RHO,sig=SIG,beta=BETA)
    # Parallel line
    xp_next, yp_next, zp_next = rk4_lorentz(h=time_step,ic=[XP[i], YP[i], ZP[i]],rho=RHO,sig=SIG,beta=BETA)

    # distance
    # the parallel point will far away from the approximate point
    dr = np.array([xp_next - x_next, yp_next - y_next, zp_next - z_next])
    dr_size = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)

    # Reset scale of dr.
    if np.mod(i, reset) == 0 and i > 0:
        # Collecting LE before reset
        """
        lambda = 1/del_t * ln|delx_n/delx_0|.
        in this case, we consider xn as dn.
        """
        delta = np.log((dr_size/d0)) / (reset*time_step)
        preLE.append(delta)
        LE.append(np.mean(preLE))
        tLE.append(t[i])

        dr_unit = dr / dr_size # rescale to 1
        dr = dr_unit * d0 # reduce to 0.1
        dr_size = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)

        xp_next = x_next + dr[0]
        yp_next = y_next + dr[1]
        zp_next = z_next + dr[2]

    X.append(x_next)
    Y.append(y_next)
    Z.append(z_next)
    XP.append(xp_next)
    YP.append(yp_next)
    ZP.append(zp_next)
    dis.append(dr_size)

fig1 = plt.figure()
ax1 = fig1.add_subplot(1,2,1, projection='3d')

# Approximate
ax1.plot(X, Y, Z, lw=1)
ax1.scatter3D(x0, y0, z0, color='red', s=10)

# Perturb Orbit
ax1.plot(XP, YP, ZP, lw=.4, alpha=0.8)
ax1.set_box_aspect([1,1,1])

ax2 = fig1.add_subplot(2,2,2)
ax2.plot(t,dis)

ax3 = fig1.add_subplot(2,2,4)
ax3.plot(tLE, LE)
ax3.annotate(f'{LE[-1]}', (tLE[-1], LE[-1]))

plt.show()

print(f'L1 = {LE[-1]:.4f}, L2 = 0, L3 = {-1-SIG-BETA-LE[-1]:.4f}')