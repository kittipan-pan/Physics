def rk4_lorentz(h:float, ic:list[float], rho:float, sig:float, beta:float):
    x,y,z = ic[0], ic[1], ic[2]

    k1x = sig * (y - x)
    k1y = x * (rho - z) - y
    k1z = x * y - beta * z

    x1 = x + k1x * h / 2
    y1 = y + k1y * h / 2
    z1 = z + k1z * h / 2

    k2x = sig * (y1 - x1)
    k2y = x1 * (rho - z1) - y1
    k2z = x1 * y1 - beta * z1

    x2 = x + k2x * h / 2
    y2 = y + k2y * h / 2
    z2 = z + k2z * h / 2

    k3x = sig * (y2 - x2)
    k3y = x2 * (rho - z2) - y2
    k3z = x2 * y2 - beta * z2

    x3 = x + k3x * h
    y3 = y + k3y * h
    z3 = z + k3z * h

    k4x = sig * (y3 - x3)
    k4y = x3 * (rho - z3) - y3
    k4z = x3 * y3 - beta * z3

    x4 = x + (k1x + 2 * k2x + 2 * k3x + k4x) * h / 6
    y4 = y + (k1y + 2 * k2y + 2 * k3y + k4y) * h / 6
    z4 = z + (k1z + 2 * k2z + 2 * k3z + k4z) * h / 6

    return x4,y4,z4
