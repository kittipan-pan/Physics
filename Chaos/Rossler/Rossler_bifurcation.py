from Chaos.Rossler.Rossler_System import rk4_rossler
import numpy as np
import matplotlib.pyplot as plt
from time import time

a_list = np.arange(0,0.39, 1e-3)
b = 0.2
c = 4
IC = [0,1,0]
x0,y0,z0 = IC[0], IC[1], IC[2]

h = 1e-1
t = np.arange(0,200 + h , h)
fig = plt.figure()
ax1 = fig.add_subplot()

t1 = time()
for a in a_list:
    X, Y, Z = [x0], [y0], [z0]
    for i in range(len(t)):
        x,y,z = rk4_rossler(h=h, ic=[X[i], Y[i], Z[i]], a=a,b=b,c=c)
        X.append(x)
        Y.append(y)
        Z.append(z)

    index = np.where(t > 150)[0][0]
    X = X[index:]
    Y = Y[index:]
    u = []
    for i in range(len(X) - 1):
        if X[i-1] < 0 < X[i+1]:
            u.append(Y[i])

    ax1.scatter(np.zeros_like(u) + a, np.abs(u), s=0.5, color='black', alpha=0.5)

t2 = time()
print(f"Time taken: {(t2-t1):.03f}s")

plt.show()