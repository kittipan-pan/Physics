import numpy as np
import matplotlib.pyplot as plt

# Use this method with the class RK4 only. Convenience to convent string equation and execute the code
def _Convert_substrings(text: str, specific_substrings: list[str], new_substrings: list[str]) -> str:
    """
    Replace specific substrings in the input text with new substrings.

        For example,

        equation = 3*x + y**2 - z*x

        old_sub = [x, y, z]

        new_sub = [x1, y1, z1]

        Method return:

        '3*x + y**2 - z*x' ---> '3*x1 + y1**2 - z1*x1'.

    ------------------------------------
    :param text: The input equation text.
    :param specific_substrings: The specific substrings to replace.
    :param new_substrings: The new substrings to replace with.
    :return: The change substring text.
    """
    for i in range(len(specific_substrings)):
        text = text.replace(specific_substrings[i], new_substrings[i])

    return text

def _PolarToCartesian(r, theta) -> list[float]:
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y

def _options(requires:list[str] | str, user_input:list[str] | str =None) -> list[bool]:
    """
    To create a method require options list.

    Example of usage,

    options(requires=['data', 'text'], user_input=['text'])

    --> return [False, True]

    --------------------------------
    :param requires: Method requires.
    :param user_input: User requires.
    :raise TypeError: If the user requires don't match the method requires.
    :return: List of bools.
    """
    require_list = [[], []]
    option_list = ['point', 'time', 'text', 'forward', 'reverse', 'both', 'data', 'offplane',
                   'cartesian']

    if not isinstance(user_input, list):
        user_input = [user_input]

    for require in requires:
        if require not in option_list:
            raise NameError(f'The \'{require}\' is not in the \'option_list\'.'
                            f' Please define the \'{require}\' in the \'option_list\'.')
        if require in option_list:
            require_list[0].append(require)
            require_list[1].append(False)

    for option in user_input:
        # Return if user doesn't require any options
        if option is None:
            return require_list[1]

        # Raise TypeError if the user's requires don't match the method's requires
        if option not in require_list[0]:
            print(user_input)
            raise TypeError(f'Not recognized option \'{option}\'!\n'
                            f'This method is requires {require_list[0]}.')

        if option in require_list[0]:
            index = require_list[0].index(option)
            require_list[1][index] = True

    return require_list[1]


def _var_convert() -> list[str]:
    var_texts = ['x', 'y', 'z', '^', 'E', 'e', 'sin', 'cos', 'tan']

    var_formats = [['x[i]', 'y[i]', 'z[i]', '**', 'np.exp', 'np.e', 'np.sin', 'np.cos', 'np.tan'],
                   ['x1', 'y1', 'z1', '**', 'np.exp', 'np.e', 'np.sin', 'np.cos', 'np.tan'],
                   ['x2', 'y2', 'z2', '**', 'np.exp', 'np.e', 'np.sin', 'np.cos', 'np.tan'],
                   ['x3', 'y3', 'z3', '**', 'np.exp', 'np.e', 'np.sin', 'np.cos', 'np.tan']]

    return var_texts, var_formats

class RK4:
    @staticmethod
    def plot1d(ic:float, equation:str,
             t:list[float], opt:list[str]=None, ax=None):
        """
        Plot an approximate solution of the Non-autonomous (no variable t) 1d-flow equation.

        ------------------------------------------------
        :param ic: Initial condition x(0).
        :param equation: String equation.
        :param t: [start, stop, step].
        :param ax: Set default axes of the figure. If no it will automatically create a figure.
        :param opt: User-option requires.
        """

        # Convenience for user input a single option
        if type(opt) is not list:
            opt = [opt]
        get_point, get_time, get_text = _options(['point', 'time', 'text'], opt)

        # Because I created and support a list.
        ic = [ic]
        equation = [equation]

        x0 = ic[0]
        x = [x0]

        h = float(t[2])
        time = np.arange(t[0], t[1] + h, h)

        # Convenience for user input a single option
        RK4_exec = f"""for i in range(len(time)-1):
        k1x = {_Convert_substrings(equation[0], _var_convert()[0], _var_convert()[1][0])}

        x1 = x[i] + k1x * h/2

        k2x = {_Convert_substrings(equation[0], _var_convert()[0], _var_convert()[1][1])}

        x2 = x[i] + k2x * h / 2

        k3x = {_Convert_substrings(equation[0], _var_convert()[0], _var_convert()[1][2])}

        x3 = x[i] + k3x * h

        k4x = {_Convert_substrings(equation[0], _var_convert()[0], _var_convert()[1][3])}

        x4 = x[i] + (k1x+2*k2x+2*k3x+k4x)*h/6

        x.append(x4)"""

        if get_text:
            print(f'IC = {ic}')
            print(f'x0, y0 = IC')
            print('x, y = [x0], [y0]')
            print(f'h = {h}')
            print(f'time = np.arange({t[0]}, {t[1]} + h, h)')
            print(RK4_exec)
            return

        exec(RK4_exec)  # run RK4 text to code

        # Return plot points
        if get_point and not get_time:
            return x
        elif get_point and get_time: # If it requires time
            return time, x

        # Display the graph is there is no 'ax'
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        ax.plot(time, x, lw=0.2)
        ax.set_xlabel('time')
        ax.set_ylabel('x')
        ax.scatter(time[0], x0, color='red', s=10)

    @staticmethod
    def plot2d(ic:list[float], equations:list[str],
               t:list[float], opt:list[str]=None, ax=None):
        """
        Plot an approximate solution of the Non-autonomous (no variable t) 2d-flow equation.

        ------------------------------------------------
        :param ic: Initial conditions [x(0), y(0)].
        :param equations: List of string equations.
        :param t: [start, stop, step].
        :param ax: Set default axes of the figure. If no it will automatically create a figure.
        :param opt: User-option requires.
        """

        # Convenience for user input a single option
        if type(opt) is not list:
            opt = [opt]
        get_point, get_time, get_text, polar_to_cartesian = _options(['point', 'time', 'text', 'cartesian'], opt)

        x0, y0 = ic[0], ic[1]
        x, y = [x0], [y0]

        h = float(t[2])
        time = np.arange(t[0], t[1] + t[2], t[2])

        # Good for making the code run faster
        RK4_exec = f"""for i in range(len(time)-1):
        k1x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][0])}
        k1y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][0])}

        x1 = x[i] + k1x * h/2
        y1 = y[i] + k1y * h/2

        k2x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][1])}
        k2y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][1])}

        x2 = x[i] + k2x * h / 2
        y2 = y[i] + k2y * h / 2

        k3x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][2])}
        k3y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][2])}

        x3 = x[i] + k3x * h
        y3 = y[i] + k3y * h

        k4x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][3])}
        k4y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][3])}

        x4 = x[i] + (k1x+2*k2x+2*k3x+k4x)*h/6
        y4 = y[i] + (k1y+2*k2y+2*k3y+k4y)*h/6

        x.append(x4)
        y.append(y4)"""

        if get_text:
            print(f'IC = {ic}')
            print(f'x0, y0 = IC')
            print('x, y = [x0], [y0]')
            print(f'h = {h}')
            print(f'time = np.arange({t[0]}, {t[1]} + h, h)')
            print(RK4_exec)
            return

        exec(RK4_exec)

        # Change Polar to Cartesian coordinates
        if polar_to_cartesian:
            x, y = _PolarToCartesian(x, y)

        # Return plot points
        if get_point and not get_time:
            return x, y
        elif get_point and get_time: # If it requires time
            return time, x, y

        # Display the graph is there is no 'ax'
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()

        if polar_to_cartesian:
            # Label Polar coordinate
            ax.set_xlabel('r')
            ax.set_ylabel(r'$\theta$')
        else:
            # Label Cartesian coordinate
            ax.set_xlabel('x')
            ax.set_ylabel('y')

        # Good when we want to use for figure subplots
        ax.plot(x, y, lw=0.2)
        ax.scatter(x[0], y[0], color='red', s=10)

    @staticmethod
    def plot3d(ic:list[float], equations:list[str],
             t=list[float], opt:list[str]=None, ax=None):
        """
        Plot an approximate solution of the Non-autonomous (no variable t) 3d-flow equation.

        ------------------------------------------------
        :param ic: Initial conditions [x(0), y(0), z(0)].
        :param equations: List of string equations.
        :param t: [start, stop, step].
        :param ax: Set default axes of the figure. If no it will automatically create a figure.
        :param opt: User-option requires.
        """

        # Convenience for user input a single option
        if type(opt) is not list:
            opt = [opt]
        get_point, get_time, get_text = _options(['point', 'time', 'text'], opt)

        x0, y0, z0 = ic[0], ic[1], ic[2]
        x, y, z = [x0], [y0], [z0]

        h = float(t[2])
        time = np.arange(t[0], t[1] + h, h)

        # Good for making the code run faster
        RK4_exec = f"""for i in range(len(time)-1):
        k1x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][0])}
        k1y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][0])}
        k1z = {_Convert_substrings(equations[2], _var_convert()[0], _var_convert()[1][0])}
    
        x1 = x[i] + k1x * h/2
        y1 = y[i] + k1y * h/2
        z1 = z[i] + k1z * h/2
    
        k2x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][1])}
        k2y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][1])}
        k2z = {_Convert_substrings(equations[2], _var_convert()[0], _var_convert()[1][1])}
    
        x2 = x[i] + k2x * h / 2
        y2 = y[i] + k2y * h / 2
        z2 = z[i] + k2z * h / 2
    
        k3x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][2])}
        k3y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][2])}
        k3z = {_Convert_substrings(equations[2], _var_convert()[0], _var_convert()[1][2])}
    
        x3 = x[i] + k3x * h
        y3 = y[i] + k3y * h
        z3 = z[i] + k3z * h
    
        k4x = {_Convert_substrings(equations[0], _var_convert()[0], _var_convert()[1][3])}
        k4y = {_Convert_substrings(equations[1], _var_convert()[0], _var_convert()[1][3])}
        k4z = {_Convert_substrings(equations[2], _var_convert()[0], _var_convert()[1][3])}
        
        x4 = x[i] + (k1x+2*k2x+2*k3x+k4x)*h/6
        y4 = y[i] + (k1y+2*k2y+2*k3y+k4y)*h/6
        z4 = z[i] + (k1z+2*k2z+2*k3z+k4z)*h/6
        
        x.append(x4)
        y.append(y4)
        z.append(z4)"""

        if get_text:
            print(f'IC = {ic}')
            print('x0, y0, z0 = IC[0], IC[1], IC[2]')
            print('x, y, z = [x0], [y0], [z0]')
            print(f'h = {h}')
            print(f'time = np.arange({t[0]}, {t[1]} + h, h)')
            print(RK4_exec)
            return

        exec(RK4_exec)

        # Return plot points
        if get_point and not get_time:
            return x, y, z
        elif get_point and get_time: # If it requires time
            return time, x, y, z

        # Display the graph is there is no 'ax'
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

        # Good when we want to use for figure subplots
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot(x, y, z, lw=0.4)
        ax.scatter3D(x0, y0, z0, color='red', s=10)

    @staticmethod
    def display():
        plt.show()

def Poincare_section(data, axis:list[str, float],
                     ax=None, opt:list[str]=None):

    # Convenience for user input a single option
    if type(opt) is not list:
        opt = [opt]
    is_forward, is_reverse, get_point, get_data = _options(['forward', 'reverse', 'point', 'data'], opt)

    data_length = len(data[0])
    axes = ['x', 'y', 'z']
    chosen_axis = axes.index(axis[0])
    remaining_axes = [i for i in range(len(axes)) if i != chosen_axis]

    # Default at zero
    if len(axis) == 1:
        axis.append(0)

    critical_val = axis[1]

    # Lists for plotting Poincare section graph
    u = []
    v = []
    points = [[],[],[]]
    for i in range(1, data_length - 1):
        # Get data of the chosen axis
        num_prev = data[chosen_axis][i-1]
        num_next = data[chosen_axis][i+1]

        # Check if the data of the chosen axis is passing that critical value
        # Get forward trajectories
        # We defined if the next value is grater than the current value
        if is_forward and num_prev <= critical_val <= num_next:
            u.append(data[remaining_axes[0]][i])
            v.append(data[remaining_axes[1]][i])
            points[0].append(data[0][i])
            points[1].append(data[1][i])
            points[2].append(data[2][i])

        # Get reverse trajectories
        if is_reverse and num_prev >= critical_val >= num_next:
            u.append(data[remaining_axes[0]][i])
            v.append(data[remaining_axes[1]][i])
            points[0].append(data[0][i])
            points[1].append(data[1][i])
            points[2].append(data[2][i])

    # Return the 2D-Poincare section graph
    if get_point:
        return u,v

    # Return which point we used 3D-graph to plot 2D-Poincare section graph
    if get_data:
        return points

    # Label axes correspond to the chosen axis
    if axis[0] == 'x':
        ax.set_xlabel('y')
        ax.set_ylabel('z')
    elif axis[0] == 'y':
        ax.set_xlabel('x')
        ax.set_ylabel('z')
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    ax.scatter(u, v, s=5)
    ax.grid()
    ax.set_title(f'{axis[0].upper()}-axis at {axis[0]} = {axis[1]:.2f}')

def Cross_plane(axis:list[str, float], size:float = 5, center:float=None, ax=None, data=None, opt:list[str]=None):

    # Convenience for user input a single option
    if type(opt) is not list:
        opt = [opt]
    off_plane = _options(['offplane'], opt)[0] # Damm it return as a list [False]

    'Display plane correspond to the axis'
    # X-plane
    grid_div = np.linspace(-size, size, 2)
    if axis[0] == 'x':
        Y, Z = np.meshgrid(grid_div, grid_div)
        X = 0 * Y + axis[1]
    # Y-plane
    elif axis[0] == 'y':
        X, Z = np.meshgrid(grid_div, grid_div)
        Y = 0 * X + axis[1]
    # Z-plane
    else:
        X, Y = np.meshgrid(grid_div, grid_div)
        Z = 0 * X + axis[1]

    if center is None:
        center = [0,0,0]

        X = X + center[0]
        Y = Y + center[1]
        Z = Z + center[2]

    if off_plane is False:
        ax.plot_surface(X, Y, Z, alpha=0.5)

    if data is not None:
        ax.scatter(data[0], data[1], data[2], color='red', s=5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

# TEST
if __name__ == '__main__':
    IC = 1
    dx_dt = 'sin(x)'
    T = [0,1,0.1]
    RK4.plot1d(ic=IC, equation=dx_dt, t=T, opt=['no'])
    RK4.display()