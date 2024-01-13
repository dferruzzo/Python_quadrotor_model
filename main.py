"""
@author: Diego Ferruzzo Correa
@date: 13/01/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../../Minhas-funcs/')
from drone import Drone
from myfunctions import *


#test()

mydrone = Drone()
x0 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]])
print("x0=", x0)
print(mydrone.dynamics(x0))
