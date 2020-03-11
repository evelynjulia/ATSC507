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
from pathlib import Path
import os

current_dir = os.getcwd()
fig_dir = str(current_dir)+'/HW7/'


#file_to_open = data_folder / "raw_data.txt"


#fig_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW7/'

# fig_dir = '/Users/catherinemathews/UBC/NWP'

run_date = dt.datetime.now().strftime('%y%m%d')

myfigsize = (9,4)

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

# %% 3) show (in red) the ideal exact final solution

#Also, on the same plot, show (in red) the ideal exact final solution,
# after the puff anomaly has been advected downwind, as given by

c_ideal = np.zeros(max_x) # initial concentration of ideal background is zero

#c_max = 10.0                      # max initial concentration

c_ideal[800:851] = np.linspace(0, c_max, 51)    # insert left side of triangle
c_ideal[850:901] = np.linspace(c_max, 0, 51)    # insert right side of triangle
c_ideal[720:741] = np.linspace(0, -0.5*c_max, 21)    # insert left side of triangle
c_ideal[740:761] = np.linspace(-0.5*c_max, 0, 21)    # insert right side of triangle




# %% 4) Advect the concentration puff anomaly 

# for the following number of time steps, plot (in green) the resulting concentration on the same graph, using FTBS
nsteps = (max_x - 300) / (u * dt / dx)
iters = np.arange(0,nsteps,1)
#  "forward in time, backward in space" (FTBS) .

# initial concentration (made longer so the diff can be the same length)
c_now = np.append(0, conc.copy())

for time_step in iters:
    #print(time_step)
    d_conc = np.diff(c_now)
    c_next = ( -Cr * d_conc ) + c_now[1:len(c_now)]
    c_now = np.append(0, c_next.copy())

c_FTBS = c_now.copy()[0:1000]

plt.plot(conc)
plt.plot(c_now)
plt.show()



# %% 2b) Plot (using blue colour) the initial concentration distribution on a graph.

xvals = np.arange(0,1000,1)

# plt.plot(xvals,conc)
# plt.show()

fig, ax = plt.subplots(1,1, figsize=myfigsize)
ax.plot(xvals,conc, color = 'b', label = 'original concentration')
ax.plot(xvals,c_ideal, color='r', label = 'exact final solution')
ax.plot(xvals, c_FTBS, color='g', label = 'FTBS solution')
ax.set_xlabel("Grid index (i)")
ax.set_ylabel("Quantity")
#plt.title("")
# ax.set_ylim(-5,95)
plt.legend()
#plt.show()
plt.savefig(fig_dir+'FTBS'+'_'+run_date+'.png')


# %%  5) 
#
#  Repeat steps (2-4) to re-initialize, but plotting (in green) on a new graph, and using
# RK3 for the advection.  Use same number of time steps.


