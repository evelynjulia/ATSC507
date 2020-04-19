# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 9
# NWP - ATSC 507


#%% 
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import math 
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D

current_dir = os.getcwd()
my_fig_dir = str(current_dir)+'/HW9/'

run_date = dt.datetime.now().strftime('%y%m%d')
myfigsize = (9,4)

# %% Question 4

from HW9.hw9_functions import gnomonic

gnomonic(radius=2, c=12, fig_dir=my_fig_dir, type='equidistant')
gnomonic(radius=2, c=12, fig_dir=my_fig_dir, type='equiangular')

gnomonic(radius=2, c=24, fig_dir=my_fig_dir, type='equidistant')
gnomonic(radius=2, c=24, fig_dir=my_fig_dir, type='equiangular')

gnomonic(radius=2, c=36, fig_dir=my_fig_dir, type='equidistant')
gnomonic(radius=2, c=36, fig_dir=my_fig_dir, type='equiangular')

gnomonic(radius=2, c=48, fig_dir=my_fig_dir, type='equidistant')
gnomonic(radius=2, c=48, fig_dir=my_fig_dir, type='equiangular')


# %% Question 1

x = np.arange(-0.9, 1, 0.1)
#x = np.arange(-1, 1.01, 0.1)

rho = 1
k = int(len(x) / 3)  # the number of points in each region
v1 = x[0:k+1]
v2 = x[k:2*k+1]
v3 = x[2*k:3*k+1]

gen_p1 = round(sum(v1 * rho) / (1*k), 3)
gen_p2 = round(sum(v2 * rho) / (1*k), 3)
gen_p3 = round(sum(v3 * rho) / (1*k), 3)
