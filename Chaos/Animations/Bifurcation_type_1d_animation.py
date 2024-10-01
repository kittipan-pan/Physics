import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

XR = []

def zero_horizontal_cross(x_data:list, y_data:list, r):
    for i in range(len(x_data)-2):
        # Upward trend (White Hollow)
        if y_data[i] < 0 < y_data[i + 1]:
            plt.scatter(x[i], 0, s=60, color='black', facecolors='none')
            data = [x[i], r, True]
            XR.append(data)
        # Downward trend (Black Hollow)
        elif y_data[i] > 0 > y_data[i + 1]:
            plt.scatter(x[i], 0, s=60, color='black')
            data = [x[i], r, False]
            XR.append(data)

a_list = np.linspace(-5,5,100)
x = np.linspace(-10,10,200)
y0_line = np.zeros_like(x)

def update(frame):
    plt.cla()
    a = a_list[frame]
    fx = x*(a-2*np.exp(x))
    plt.plot(x, fx) # Plot function f(x)
    zero_horizontal_cross(x, fx, a)
    plt.plot(x, y0_line, color='black')
    plt.title(f'a = {np.round(a_list[frame],2)}')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig=fig, func=update,frames=len(a_list), interval=1, repeat=False)
plt.show()

for x,r,fill in XR:
    if fill:
        plt.scatter(x,r, s=60, color='black', facecolors='none')
    else:
        plt.scatter(x, r, s=60, color='black')

plt.grid()
plt.xlabel('x')
plt.ylabel('a')
plt.show()