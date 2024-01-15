"""
@author: Diego Ferruzzo Correa
@date: 14/01/2024
"""

import numpy as np
#import matplotlib.pyplot as plt
#import sys
#sys.path.append('../../Minhas-funcs/')
from drone import Drone, plot_trajectories
#from myfunctions import *

drone = Drone()
x0 = np.array([0,0,0,0,1,0,0,0,0,0,0,0])
U = np.array([1,0,0,0])
drone.set_x0(x0)
drone.set_U(U)
drone.compute_trajectories()
plot_trajectories(drone.compute_trajectories())
