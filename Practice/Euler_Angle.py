import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

pi = np.pi
I3 = np.matrix(np.identity(3))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def euler_angle(phi: float, theta: float, psi: float, decimal_limit: int=3) -> np.ndarray:
    """
    :return: transformation array by 3x3.
    """
    a1, a2, a3= np.deg2rad(phi), np.deg2rad(theta), np.deg2rad(psi)

    transformation = [[
        cos(a3)*cos(a1) - cos(a2)*sin(a1)*sin(a3),
        cos(a3)*sin(a1) + cos(a2)*cos(a1)*sin(a3),
        sin(a3)*sin(a2)], [
        -sin(a3)*cos(a1) - cos(a2)*sin(a1)*cos(a3),
        -sin(a3)*sin(a1) + cos(a2)*cos(a1)*cos(a3),
        cos(a3)*sin(a2)], [
        sin(a2)*sin(a1), -sin(a2)*cos(a1), cos(a2)]]

    return np.round(transformation, decimal_limit)
class Coordinates:
    def __init__(self, size: float=10., origin: list=(0,0,0), euler_rotate=euler_angle(0,0,0)):
        self.s = size
        self.origin = np.array(origin)
        self.A = euler_rotate

        # i-j-k components of the coordinate.
        self.components = self.A

        # Collect xyz-axis position.
        # i.e. [[1,0,0], [0,1,0], [0,0,1]]
        axes_limits = self.components *self.s + self.origin

        # x-axis
        ax.plot3D(
            [origin[0], axes_limits[0][0]],
            [origin[1], axes_limits[0][1]],
            [origin[2], axes_limits[0][2]], 'blue')

        # y-axis
        ax.plot3D(
            [origin[0], axes_limits[1][0]],
            [origin[1], axes_limits[1][1]],
            [origin[2], axes_limits[1][2]], 'red')

        # z-axis
        ax.plot3D(
            [origin[0], axes_limits[2][0]],
            [origin[1], axes_limits[2][1]],
            [origin[2], axes_limits[2][2]], 'green')

class Vector:
    """
    Accepting points pos1=[x1,y1,z1] and pos2=[x2,y2,z2] to display as a vector.
    :return: display a vector.
    """
    def __init__(self, position1: list=None, position2: list=None, color: str='black', visible: bool=True):

        self.pos1 = np.array(position1)
        self.pos2 = np.array(position2)

        # get a scalar of the vector.
        self.scalar = np.sqrt(np.sum(np.square(self.pos2 - self.pos1)))

        self.direction = self.pos2 - self.pos1

        # decorating vector.
        self.color = color
        self.visible = visible
        if self.visible:
            Vector.vector_plot(self)

    def vector_plot(self) -> None:

        ax.quiver(
            X=self.pos1[0], Y=self.pos1[1], Z=self.pos1[2],
            U=self.direction[0], V=self.direction[1], W=self.direction[2],
            color=self.color)

    """
    There are no usages because we can't plot a vector from a reference coordinate.
    We only got their components about reference axes.
    """
    def get_point(self, reference_coordinate=None):
        if reference_coordinate is not None:
            return np.array([np.matmul(reference_coordinate.A, self.pos1.T),
                             np.matmul(reference_coordinate.A, self.pos2.T)])
        else:
            return np.array([self.pos1, self.pos2])

    # Return the i-j-k component of the vector depends on reference coordinate.
    def get_component(self, reference_coordinate=None) -> list:
        if reference_coordinate is not None:
            return np.matmul(reference_coordinate.A, self.direction.T)
        else:
            return self.direction

    # Short code for displaying vector if you are too lazy to set variable of self.visible .
    def enable(self) -> bool:
        return self.visible == True
    def disable(self) -> bool:
        return self.visible == False

    # Adding by tail to head vector algebra.
    # The resultant vector will point out from the first vector's origin.
    def __add__(self, other):
        new_direction = self.direction + other.direction
        return Vector(position1=self.pos1, position2=new_direction)

#################################################################
def main() -> None:
    size = 5

    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    A = euler_angle(45,0,0)
    Coordinates()
    Coordinates(euler_rotate=A)

    plt.show()

if __name__ == '__main__':
    main()