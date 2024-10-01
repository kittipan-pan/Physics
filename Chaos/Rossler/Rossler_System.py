def rk4_rossler(h:float, ic:list[float], a,b,c):
    x, y, z = ic[0], ic[1], ic[2]

    k1x = -y - z
    k1y = x + a * y
    k1z = b + z * (x - c)

    x1 = x + k1x * h / 2
    y1 = y + k1y * h / 2
    z1 = z + k1z * h / 2

    k2x = -y1 - z1
    k2y = x1 + a * y1
    k2z = b + z1 * (x1 - c)

    x2 = x + k2x * h / 2
    y2 = y + k2y * h / 2
    z2 = z + k2z * h / 2

    k3x = -y2 - z2
    k3y = x2 + a * y2
    k3z = b + z2 * (x2 - c)

    x3 = x + k3x * h
    y3 = y + k3y * h
    z3 = z + k3z * h

    k4x = -y3 - z3
    k4y = x3 + a * y3
    k4z = b + z3 * (x3 - c)

    x4 = x + (k1x + 2 * k2x + 2 * k3x + k4x) * h / 6
    y4 = y + (k1y + 2 * k2y + 2 * k3y + k4y) * h / 6
    z4 = z + (k1z + 2 * k2z + 2 * k3z + k4z) * h / 6

    return x4,y4,z4