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
fig_dir = str(current_dir)+'/HW9/'

run_date = dt.datetime.now().strftime('%y%m%d')
myfigsize = (9,4)

# %%

# set a value for a and c:
a = 2
c = 12 # this is a c12 sphere


# set up x y and z values
x = np.linspace(-a,a,c)
y = np.linspace(-a,a,c)

# create the points along the edges
xv, yv = np.meshgrid(x, y)

# create the z values for cube
top =  np.ones((c,c)) *2
bottom =  np.ones((c,c)) *-2

# %%
# plot

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xv, yv, z)
ax.plot_wireframe(xv, yv, top, color = 'b')
ax.plot_wireframe(xv, yv, bottom, color = 'b')
ax.plot_wireframe(top, xv, yv, color = 'r')
ax.plot_wireframe(bottom, xv, yv, color = 'r')
ax.plot_wireframe(xv, top, yv, color = 'g')
ax.plot_wireframe(xv, bottom, yv, color = 'g')
#ax.view_init(60, 35)
#plt.show()
plt.savefig(fig_dir+'step1_box'+'_'+run_date+'.png')

# %%

# set the radius of the sphere
r = 2

x, y, z = (r) / np.sqrt() * ()