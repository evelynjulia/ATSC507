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

# %%

from HW9.hw9_functions import gnomonic

gnomonic(radius=2, c=12, fig_dir=my_fig_dir, type='equidistant')
gnomonic(radius=2, c=12, fig_dir=my_fig_dir, type='equiangular')


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

# %% Equidistant gnomonic plot

# set the radius of the sphere
r = 2

# set a value for a and c:
a = (np.sqrt(3)) / 3 *r
c = 12 # this is a c12 sphere


# set up x y and z values
x = np.linspace(-a,a,c)
y = np.linspace(-a,a,c)

# a* np.tan(x)
# ylocal_angle = a* np.tan(y)

# create the points along the edges
xlocal, ylocal = np.meshgrid(x, y)

# # create the z values for cube
# top =  np.ones((c,c)) *2
# bottom =  np.ones((c,c)) *-2

#xsphere = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) )* (a, xlocal, ylocal)
#ysphere = 
#zsphere = 

bigZ_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * a
bigZ_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -a

bigx_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * xlocal
bigx_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -xlocal

bigy_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * ylocal
bigy_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -ylocal

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xv, yv, z)
ax.plot_wireframe(bigx_top, bigy_top, bigZ_top, color = 'b')
ax.plot_wireframe(bigx_bottom, bigy_bottom, bigZ_bottom, color = 'b')
ax.plot_wireframe(bigZ_top, bigx_top, bigy_top, color = 'r')
ax.plot_wireframe(bigZ_bottom, bigx_bottom, bigy_bottom, color = 'r')
ax.plot_wireframe(bigx_top, bigZ_top, bigy_top, color = 'g')
ax.plot_wireframe(bigx_bottom, bigZ_bottom, bigy_bottom, color = 'g')
plt.title('equidistant gnomonic projection')
#ax.view_init(60, 35)
#plt.show()
#plt.savefig(fig_dir+'equidistant_gnomonic'+'_'+run_date+'.png')

#x, y, z = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) )* (a, xlocal, ylocal)


# %%
a = np.pi / 4
x = np.linspace(-a,a,c)
y = np.linspace(-a,a,c)
xlocal_angle = a* np.tan(x)
ylocal_angle = a* np.tan(y)


# create the points along the edges
xlocal, ylocal = np.meshgrid(xlocal_angle, ylocal_angle)

# create sides
bigZ_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * a
bigZ_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -a

bigx_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * xlocal
bigx_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -xlocal

bigy_top = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * ylocal
bigy_bottom = ( (r) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -ylocal

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(xv, yv, z)
ax.plot_wireframe(bigx_top, bigy_top, bigZ_top, color = 'b')
ax.plot_wireframe(bigx_bottom, bigy_bottom, bigZ_bottom, color = 'b')
ax.plot_wireframe(bigZ_top, bigx_top, bigy_top, color = 'r')
ax.plot_wireframe(bigZ_bottom, bigx_bottom, bigy_bottom, color = 'r')
ax.plot_wireframe(bigx_top, bigZ_top, bigy_top, color = 'g')
ax.plot_wireframe(bigx_bottom, bigZ_bottom, bigy_bottom, color = 'g')
plt.title('equiangular gnomonic projection')
#plt.show()
plt.savefig(fig_dir+'equiangular_gnomonic'+'_'+run_date+'.png')

# plt.plot(x, np.ones(12))
# plt.plot(xlocal_angle, np.zeros(12))
# plt.show()