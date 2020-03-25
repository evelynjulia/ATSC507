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
z = np.linspace(-a,a,c)

x2d = np.ones((c,c)) *x
y2d = np.ones((c,c)) *y
z2d = np.ones((c,c)) *z

xv, yv = np.meshgrid(x, y)


# %%
# plot


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xv, yv, zv)
#ax.plot_wireframe(xv, yv, z2)
ax.contour3D(x2d, y2d, z2d, 50, cmap='binary')
#ax.plot_trisurf(x2d, y2d, z2d, linewidth=0, antialiased=False)
#ax.contour3D(x2d,y2d,z2d)
#ax.view_init(60, 35)
fig
plt.show()

# %%
# test plot

def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

