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

# %% Create the grid and initial conditions

max_x = 1000             # number of grid points in x-direction
dx = 100.               # horizontal grid spacing (m)
dt = 10.                # time increment (s)
u = 5.                  # horizontal wind speed (m/s)


# %% Question 1: calculate and display courant number

# (u*dt)/dx

Cr = (u*dt) / dx
print("the courant number is ", Cr)

# %% 2a

# Create initial concentration anomaly distribution in the x-direction

np.
conc <- rep(0.0, imax)   # initial concentration of background is zero


cmax = 10.0                      # max initial concentration
conc[100:150] <- seq(0., cmax, len = 51)        # insert left side of triangle
conc[150:200] <- seq(cmax, 0., len = 51)        # insert right side of triangle
conc[20:40] <- seq(0., -0.5*cmax, len = 21)    # insert left side of triangle
conc[40:60] <- seq(-0.5*cmax, 0., len = 21)    # insert right side of triangle