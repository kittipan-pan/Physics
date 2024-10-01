import matplotlib.pyplot as plt
import numpy as np
"""
x_n+1 = r*x_n*(1-x_n)

X = RX(1-X) = RX - RX^2
X' = R - 2RX
"""
r = 1
n = 100
x0 = 0.3

# Plot Setting
fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_figheight(6)
fig.set_figwidth(14)

ax1.set_xlim(0,1)
ax1.set_ylim(0,1)
ax1.grid()
ax1.set_title(r"Logistic Map of $x_{n+1}$ = $r x_{n} (1-x_{n})$")

ax2.set_xlim(0,n)
ax2.grid()
ax2.set_title(f"r={r}")
ax2.set_xlabel(r'$x_{n}$')
ax2.set_ylabel(r'$x_{n+1}$')

# Logistic Map Trend line
t = np.arange(0,5,0.01)
f = r*t*(1-t)
ax1.plot(t,f,color='black')
# y = x line
ax1.plot(t,t,color='orange')

# Logistic Map Equation
X = [x0]
Y = []

for i in range(n):
    x_next = r*X[i]*(1-X[i]) # map i times
    X.append(x_next)
    Y.append(x_next)

Y.append(r*X[n]*(1-X[n]))
ax1.scatter(X,Y,marker='x',s=100)

# Cobweb line
horizontal = []
vertical = [0]
for i in range(len(X)):
    horizontal.append(X[i])
    horizontal.append(X[i])
    vertical.append(Y[i])
    if i < (len(X)-1):
        vertical.append(Y[i])

ax1.plot(horizontal,vertical,color='red')

# x_n graph
index_X = list(range(len(X)))
ax2.plot(index_X, X, '-s')

plt.show()
