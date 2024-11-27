# ------------------------------------------------------------------------------------------------------------------- #
#                                                 ADJUST VARIABLES                                                    #
# ------------------------------------------------------------------------------------------------------------------- #
I = 2 # Current

# Adjust graphic visualizations
xlim = [-1, 1]
ylim = [-1, 1]
zlim = [-1, 1]
DISPLAY_MAGNETIC_IN_MOUSE_POSITION = False
DISPLAY_GRAPH = True

# Adjust magnetic vector
VECTOR_SIZE: float = 0.5
DISPLAY_MAGNETIC_VECTOR: bool = True
MAGNETIC_FADING_VECTOR: bool = True
FANCY_VECTOR: bool = True

# Adjust positions
grid_res: int = 5
position_size: float = 5.0
DISPLAY_POSITION: bool = False

# Adjust solenoid coil
N: int = 10 # Number of turns
l: float = 1 # Length of coil
r: float = 0.4 # Coil radius
coil_res: int = 200 # Resolution of data
coil_thickness: float = 2.0
DISPLAY_COIL: bool = True

SPECIFIC_MAGNETIC_FIELD_POSITION = (0.5, 0.5, 0.5)
# ------------------------------------------------------------------------------------------------------------------- #
# TEST
# B(0.5,0.5,0.5) = 1.0293403582319423e-06 T, N = 10, l = 1, r = 0.4, I = 1
# B(0.5,0.5,0.5) = 5.227736913933672e-07 T, N = 5, l = 1, r = 0.4, I = 1
# B(0.5,0.5,0.5) = 4.734042715263149e-07 T, N = 10, l = 2, r = 0.4, I = 1
# B(0.5,0.5,0.5) = 2.674089284969609e-07 T, N = 10, l = 1, r = 0.2, I = 1
# B(0.5,0.5,0.5) = 2.0586807164638846e-06 T, N = 10, l = 1, r = 0.4, I = 2