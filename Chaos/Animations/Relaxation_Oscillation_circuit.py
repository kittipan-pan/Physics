import matplotlib.pyplot as plt
from matplotlib.animation import  FuncAnimation
from Chaos.Multi_dimensions.Tools import RK4

IC = [1,0]
# I > neta : in this case I = 1, neta = 0.1
flows = ['x-1/3*x**3 - y', '0.4*x']
t, I, VC = RK4.plot2d(IC, flows,[0,50,0.1], ['point', 'time'])

fig = plt.figure(figsize=(10,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
def update(frame):
    ax1.cla()
    ax2.cla()

    # Plot VC versus I
    ax1.set_title(r'$\dot{I} = I - \frac{x^3}{3} - V_c,\qquad \dot{V_c} = 0.4I$')
    ax1.plot(I[:frame],VC[:frame])
    ax1.scatter(I[frame-1],VC[frame-1], s=10, color='red')

    # Plot V and I in time
    ax2.plot(t[:frame], I[:frame], label='I')
    ax2.plot(t[:frame], VC[:frame], label='VC')

    ax1.set_xlim(-2.5,2.5)
    ax1.set_ylim(-2.5,2.5)
    ax1.set_xlabel(r'$\dot{I}$')
    ax1.set_ylabel(r'$\dot{V_c}$')
    ax1.grid()

    ax2.set_title(f't = {t[frame]:.02f}')
    ax2.set_xlim(0,50)
    ax2.set_ylim(-2.5,2.5)
    ax2.legend()
    ax2.grid()

ani = FuncAnimation(fig=fig, func=update, frames=len(t), interval=1, repeat=False)
plt.show()
