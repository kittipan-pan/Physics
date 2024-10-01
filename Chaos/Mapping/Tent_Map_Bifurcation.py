import matplotlib.pyplot as plt
import numpy as np
from random import random as rand
import time

start = time.time()
"""
x_n+1 = r*x_n ; x_n < 1/2 , r*(1-x_n) ; x_n >= 1/2
X' = r ; X < 1/2, -r ; X > 1/2
"""

h = 1e-4 # Time Steps of r
r = np.arange(0,2,h)

def mapping(initial_condition,r):
    xi = initial_condition
    x_next = 0

    preLE = []
    for i in range(100):
        if xi < 0.5:
            x_next = r*xi
        elif xi >= 0.5:
            x_next = r*(1-xi)
        preLE.append(np.log(r))
        xi = x_next

    le = np.mean(preLE)

    return x_next, le

X = []
LE = []
for i in range(len(r)):
    x0 = rand()
    x, lyapunov = mapping(initial_condition=x0,r=r[i])
    X.append(x)
    LE.append(lyapunov)

end = time.time()
print(f"Time taken: {(end-start):.03f}s")

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.scatter(r,X,s=0.01,color='blue')
ax2.plot(r,LE)
ax2.plot(r,np.zeros_like(r), color='red')
ax2.set_ylim(-1,1)
plt.show()