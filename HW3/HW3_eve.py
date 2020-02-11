# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 3
# NWP - ATSC 507

'''
Homework #3, due in 1 week or so.

Coastline data: https://www.evl.uic.edu/pape/data/WDB/
Focus on coastlines, islands and lakes (-cil) for N. America, Europe, and Asia.

1) Then everyone use these files to create two maps, similar to the maps in the solved 
example on page 748 Stull 2017 Practical Meteorology.  Namely, one map is a plot of 
the coastlines on a lat/lon grid, and the other is a plot of the coastlines on a polar 
stereographic grid using 60°N as the reference latitude.

2) On page 748 Stull 2017 Practical Meteorology, in the INFO box, is eq. (F20.2), 
which gives an expression for "r".  This looks different than the expression I gave 
in class.  Start with the eq for "r" that Stull wrote on the blackboard during class, 
and show how you can manipulate that equation to get eq (F20.2). 

3) For the second advection term shown in class (term II), simplify it using 
equations such as M27 or other equations.  Show your work, and discuss the 
interpretation of your results.


'''


# %%

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import interpolate
import pandas as pd
import math 


fig_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW3/'
data_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW3/coastlines/'

# fig_dir = '/Users/catherinemathews/UBC/NWP'
# data_dir = '/Users/catherinemathews/UBC/NWP'

run_date = dt.datetime.now().strftime('%y%m%d')

myfigsize = (15,9)

# %% Load data

asia = pd.read_csv(data_dir+'asia-cil.csv', header=None, usecols=[0,1], names=("lat", "lon"))
namer =pd.read_csv(data_dir+'namer-cil.csv', header=None, usecols=[0,1], names=("lat", "lon"))
europe = pd.read_csv(data_dir+'europe-cil.csv', header=None, usecols=[0,1], names=("lat", "lon"))
samer = pd.read_csv(data_dir+'samer-cil.csv', header=None, usecols=[0,1], names=("lat", "lon"))
africa = pd.read_csv(data_dir+'africa-cil.csv', header=None, usecols=[0,1], names=("lat", "lon"))

# just get northern asia
asia_n = asia[asia['lat']>=0]

# combine dataset and get northern hemisphere:
all_data = asia.append(namer).append(europe).append(samer).append(africa)

data_n = all_data[all_data['lat']>=0] # & all_data['lon']>=-180]
n_hem = data_n[data_n['lon']>=-180]

# %% Transform to stereographic coords
'''
x = r* cos(lon)
y = r*sin(lon)
r = L * tan(0.5* (90-lat))
L = R0*(1+sin(ref_lat))

R0 = 6371
ref_lat = 60 # the latitude intersected by the projection plane
'''
to_rad = math.pi/180
R0 = 6371
ref_lat = 60 


# get lat and lon in radians
lat = n_hem['lat']*to_rad
lon = n_hem['lon']*to_rad

L = R0 * ( 1 + np.sin(ref_lat*to_rad) )
r = L * np.tan(0.5* (90-lat))

x = r * np.cos(lon)
y = r * np.sin(lon)

# %% Other lines to plot

# get equator line
eq_lat = np.zeros(len(lat))
eq_lat_rad = eq_lat*to_rad
eq_r = L * np.tan(0.5* (90-eq_lat_rad))

eq_x = eq_r * np.cos(lon)
eq_y = eq_r * np.sin(lon)

# get lon 0, 45, 90, 135, 180, -45, -90, -135

all_lat = np.arange(0,90,0.001)
all_lat_rad = all_lat*to_rad
all_r = r = L * np.tan(0.5* (90-all_lat_rad))

lon0 = np.zeros(len(all_lat))
lon45 = np.ones(len(lon)) * 45
lon90 = np.ones(len(lon)) * 90
lon135 = np.ones(len(lon)) * 135

x0 = all_r * np.cos(lon0)
y0 = all_r * np.sin(lon0)

# plt.plot(x0,y0,'.',markersize=0.1)
# plt.show()


# %% PLOTS

plt1 = "lat_lon_grid"# title for plt 1
plt2 = "stereo_grid"

fig, ax = plt.subplots(1,1, figsize=myfigsize)
# plt.plot(namer['lon'], namer['lat'],  '.', markersize=0.1)
# plt.plot(asia_n['lon'], asia_n['lat'],  '.', markersize=0.1)
# plt.plot(europe['lon'], europe['lat'],  '.', markersize=0.1)
plt.plot(n_hem['lon'], n_hem['lat'], '.', markersize=0.1)
plt.grid()
ax.set_xlabel("Longitude (\N{DEGREE SIGN})")
ax.set_ylabel("Latitude (\N{DEGREE SIGN})")
plt.title("Northern Hemisphere coastlines: lat-lon grid")
ax.set_ylim(-5,95)
plt.show()
#plt.savefig(fig_dir+run_date+'_'+plt1+'.png')

fig, ax = plt.subplots(1,1, figsize=(9,9))
ax.plot(x, y, '.', markersize=0.1)
ax.plot(eq_x, eq_y, '.', markersize=0.1)
#plt.plot(x0,y0,'.',markersize=0.1)
ax.set_xlabel("x (km)")
ax.set_ylabel("y (km)")
plt.title("Northern Hemisphere coastlines: polar stereographic grid")
plt.show()
#plt.savefig(fig_dir+run_date+'_'+plt2+'.png')






# %%
