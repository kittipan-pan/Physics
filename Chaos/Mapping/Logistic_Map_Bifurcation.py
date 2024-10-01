import matplotlib.pyplot as plt
import numpy as np
import random
import time

start = time.time()

"""
x_n+1 = r*x_n*(1-x_n)

X' = r - 2*r*X
"""

h = 1e-4 # Time Steps of r
r = np.arange(0,4,h)
def mapping(initial_condition:float, r:float):
    xi = initial_condition
    x_next = 0

    preLE = []
    for i in range(0, 100):
        x_next = r*xi*(1-xi)
        lyapunov = np.log(abs(r - 2*r*xi))
        preLE.append(lyapunov)
        xi = x_next

    le = np.mean(preLE)

    return x_next, le

X = []
LE = []
for i in range(len(r)):
    x0 = np.random.random()
    x, lyapunov = mapping(initial_condition=x0, r=r[i])
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
