import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from setting import *

# Constant variables
PI = np.pi
MU0 = 12.57e-7

# Collection of positions data of coil's length
l_positions: list[tuple[float | int, float | int, float | int]]

# Magnetic field function
def magnetic_field(l_current: tuple[float, float, float], l_next: tuple[float, float, float], p: tuple[float, float, float]) -> np.array:
    """
    :param l_current: current position of length data.
    :param l_next: next position of length data.
    :param p: target position.
    :return: magnetic in vector form.
    """

    # Using Biot-Savart Law. We neglect the constant term, mu0*I/4*pi, for simplicity
    dl = [l_next[0] - l_current[0], l_next[1] - l_current[1], l_next[2] - l_current[2]]
    dr = np.array([p[0] - l_current[0], p[1] - l_current[1], p[2] - l_current[2]])
    dr_size = np.sqrt(dr[0]**2 + dr[1]**2 + dr[2]**2)

    if dr_size == 0:
        return np.zeros(3)

    unit_dr = dr / dr_size

    B = np.cross(dl, unit_dr) / dr_size ** 2

    return B


# Generating solenoid-coil positions data
center_z = l / 2
theta = np.linspace(0.0, 2 * np.pi * N, coil_res)
x = r * np.cos(theta)
y = r * np.sin(theta)
dz = l / coil_res
z = np.arange(0, l, dz) - center_z
l_positions = list(zip(x.tolist(),y.tolist(),z.tolist()))

# # Example: Straight coil positions data in z-direction as shown below
# l_positions = [
#     (0,0,-1),
#     (0,0,-0.5),
#     (0,0,0),
#     (0,0,0.5),
#     (0,0,1),
# ]

# Create a grid of positions
_x = np.linspace(xlim[0], xlim[1], grid_res)
_y = np.linspace(ylim[0], ylim[1], grid_res)
_z = np.linspace(zlim[0], zlim[1], grid_res)
x, y, z = np.meshgrid(_x, _y, _z)

# Flatten the grid into a list of positions
positions: list[tuple[float, float, float]] = list(zip(x.ravel().tolist(), y.ravel().tolist(), z.ravel().tolist()))


# List to store the magnetic field directions at each point
data = []

# Calculate magnetic field at each point in the grid
if DISPLAY_MAGNETIC_VECTOR: # Why do I calculate if I don't show the vectors?
    for p_current in positions:
        net_B = np.zeros(3)
        for i in range(0, len(l_positions) - 1):
            Bi = magnetic_field(l_positions[i], l_positions[i+1], p_current)
            net_B += Bi
        data.append((p_current, tuple(net_B.tolist())))


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Set axis limits
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_zlim(zlim)

# Set axis labels
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Display magnetic vectors
if DISPLAY_MAGNETIC_VECTOR:

    # Normalize the field magnitudes for color mapping
    magnitudes = np.array([np.linalg.norm(direction) for _, direction in data])
    norm = Normalize(vmin=magnitudes.min(), vmax=magnitudes.max())
    cmap = cm.viridis  # You can choose other colormaps (e.g., 'plasma', 'inferno', etc.)

    for position, direction in data:
        X, Y, Z = position
        U, V, W = direction

        # Normalize the vector magnitude for arrow scaling
        magnitude = np.linalg.norm([U, V, W])

        # Get the color based on the magnitude of the vector
        color = cmap(norm(magnitude))

        if FANCY_VECTOR:
            # Using Non-linear alpha makes graph better
            # Map the magnitude to alpha (transparency)
            alpha = max(0.3, norm(
                magnitude) ** 0.5)  # Adjust 0.5 (exponent) to control fading strength; 0.3 is the minimum alpha
        else:
            # Linear alpha
            # # Map the magnitude to alpha (transparency) - higher magnitude = more opaque
            alpha = norm(magnitude)  # Use the same normalization to map the magnitude to alpha (0-1)

        # Plot the quiver (arrows) with color and transparency
        if MAGNETIC_FADING_VECTOR:
            ax.quiver(X, Y, Z, U, V, W, color=color, length=VECTOR_SIZE*alpha, normalize=True, alpha=alpha)
        else:
            ax.quiver(X, Y, Z, U, V, W, color=color, length=VECTOR_SIZE*alpha, normalize=True, alpha=1)

# Display positions
if DISPLAY_POSITION:
    for position in positions:
        ax.scatter3D(position[0], position[1], position[2], s=position_size, color='gray')

# Display coil
if DISPLAY_COIL:
    lx = []
    ly = []
    lz = []

    for i in range(0, len(l_positions)):
        lx.append(l_positions[i][0])
        ly.append(l_positions[i][1])
        lz.append(l_positions[i][2])

    plt.plot(lx,ly,lz, color="black", lw=coil_thickness)

def specific_magnetic_field_position(p):
    net_B = np.zeros(3)
    for i in range(0, len(l_positions) - 1):
        Bi = magnetic_field(l_positions[i], l_positions[i + 1], p)
        net_B += Bi

    B_size = np.sqrt(net_B[0] ** 2 + net_B[1] ** 2 + net_B[2] ** 2)
    return B_size * I * MU0 / (4 * PI)

# Function to track the mouse's position in 3D as a list
if DISPLAY_MAGNETIC_IN_MOUSE_POSITION:
    def on_mouse_move(event):
        if event.inaxes == ax:
            # Retrieve the mouse's x, y position in data coordinates
            mouse_x, mouse_y = event.xdata, event.ydata
            if mouse_x is not None and mouse_y is not None:
                # Extract 3D coordinates using format_coord
                coords_str = ax.format_coord(mouse_x, mouse_y)

                # Parse the coordinates string (e.g., "x=1.0, y=2.0, z=3.0")
                coords_list = [
                    value.split('=')[1]
                    for value in coords_str.split(', ')
                ]

                if "°" in coords_list[0]:
                    return

                mouse_pos = []
                for coord in coords_list:
                    mouse_pos.append(float(coord.replace("−", "-")))

                # Update the figure's title
                fig.suptitle(f"B({mouse_pos[0]}, {mouse_pos[1]}, {mouse_pos[2]}) = {specific_magnetic_field_position(mouse_pos)} T", fontsize=10)
                fig.canvas.draw_idle()
    # Connect the event to the figure
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

print(f"B({SPECIFIC_MAGNETIC_FIELD_POSITION[0]},{SPECIFIC_MAGNETIC_FIELD_POSITION[1]},{SPECIFIC_MAGNETIC_FIELD_POSITION[2]}) = "
      f"{specific_magnetic_field_position(SPECIFIC_MAGNETIC_FIELD_POSITION)} T")

# Display the plot
if DISPLAY_GRAPH:
    plt.show()