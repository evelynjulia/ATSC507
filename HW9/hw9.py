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


# %% Question 1

