# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 7
# NWP - ATSC 507

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import interpolate
import pandas as pd
import math 

fig_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW7/'

# fig_dir = '/Users/catherinemathews/UBC/NWP'

run_date = dt.datetime.now().strftime('%y%m%d')

myfigsize = (15,9)

# %% Create the grid and initial conditions

max_x = 1000             # number of grid points in x-direction
dx = 100.               # horizontal grid spacing (m)
dt = 10.                # time increment (s)
u = 5.                  # horizontal wind speed (m/s)


# %% Question 1: calculate and display courant number

# (u*dt)/dx

Cr = (u*dt) / dx
print("the courant number is ", Cr)


# %% 2a) Create initial concentration anomaly distribution in the x-direction

conc = np.zeros(max_x) # initial concentration of background is zero

c_max = 10.0                      # max initial concentration

conc[100:151] = np.linspace(0, c_max, 51)   # insert left side of triangle
conc[150:201] = np.linspace(c_max, 0, 51)        # insert right side of triangle
conc[20:41] = np.linspace(0, -0.5*c_max, 21)    # insert left side of triangle
conc[40:61] = np.linspace(-0.5*c_max, 0, 21)    # insert right side of triangle


# %% 2b) Plot (using blue colour) the initial concentration distribution on a graph.

xvals = np.arange(0,1000,1)

plt.plot(xvals,conc)
plt.show()

fig, ax = plt.subplots(1,1, figsize=myfigsize)
plt.plot(xvals,conc)
ax.set_xlabel("Grid index (i)")
ax.set_ylabel("Quantity")
#plt.title("")
# ax.set_ylim(-5,95)
plt.show()
#plt.savefig(fig_dir+run_date+'_'+'example_plot'+'.png')

