# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 6
# NWP - ATSC 507


#%% 

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import interpolate
import pandas as pd
import math 
from pathlib import Path
import os
from scipy.ndimage.interpolation import shift

current_dir = os.getcwd()
fig_dir = str(current_dir)+'/HW6/'

run_date = dt.datetime.now().strftime('%y%m%d')
myfigsize = (9,4)

#%% Part A

m = np.linspace(0.1,10,1000)
cr = (np.sqrt(18)) / ( 8* np.sin( ( 2 * np.pi ) / m ) - np.sin( ( (4*np.pi ) / m ) ) )

constant = np.ones(m.shape)*0.729

# %% plot


fig, ax = plt.subplots(1,1, figsize=myfigsize)
ax.plot(m,cr, color = 'b', label = 'solution')
ax.plot(m,constant, color='r', label = 'Cr = 0.729')
ax.set_xlabel("m")
ax.set_ylabel("Cr")
ax.set_ylim(-6,6)
plt.title("A1 plot")
# ax.set_ylim(-5,95)
plt.legend()
#plt.show()
plt.savefig(fig_dir+'A1'+'_'+run_date+'.png')


# %%


# amplitude with cr

# try for m = 1
#m = np.array([2, 2.5, 3, 4, 8, 16])
cr = np.arange(0, 4, 0.1)
m=2
ak2 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )
m=2.5
ak25 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )
m=3
ak3 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )
m=4
ak4 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )
m=8
ak8 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )
m=16
ak16 = np.sqrt(  1 - (cr**2 / 18)*( 8*np.sin(2*np.pi / m) - np.sin(4*np.pi / m)  )  )



fig, ax = plt.subplots(1,1, figsize=myfigsize)
ax.plot(cr,ak2, label = 'm = 2')
ax.plot(cr,ak25, label = 'm = 2.5')
ax.plot(cr,ak3, label = 'm = 3')
ax.plot(cr,ak4, label = 'm = 4')
ax.plot(cr,ak8, label = 'm = 8')
ax.plot(cr,ak16, label = 'm = 16')
#ax.plot(xvals,c_ideal, color='r', label = 'exact final solution')
ax.set_xlabel("Cr")
ax.set_ylabel("Amplitude")
#plt.title("")
# ax.set_ylim(-5,95)
plt.legend()
#plt.show()
plt.savefig(fig_dir+'A1b'+'_'+run_date+'.png')

####################################################
# %% Part B

Cr = np.arange(0, 4, 0.1)

m=2
kdx = 2*np.pi / m
# calculate the amplutide
term1 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr2 = term1 + term2 - term3 - term4
ak2 = np.sqrt(ak_sqr2)


m=2.5
kdx = 2*np.pi / m
# calculate the amplutide
term1_25 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2_25 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3_25 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4_25 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr_25 = term1_25 + term2_25 - term3_25 - term4_25
ak25 = np.sqrt(ak_sqr_25)

m=3
kdx = 2*np.pi / m
# calculate the amplutide
term1_3 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2_3 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3_3 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4_3 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr_3 = term1_3 + term2_3 - term3_3 - term4_3
ak3 = np.sqrt(ak_sqr_3)


m=4
kdx = 2*np.pi / m
# calculate the amplutide
term1_4 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2_4 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3_4 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4_4 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr_4 = term1_4 + term2_4 - term3_4 - term4_4
ak4 = np.sqrt(ak_sqr_4)


m=8
kdx = 2*np.pi / m
# calculate the amplutide
term1_8 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2_8 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3_8 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4_8 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr_8 = term1_8 + term2_8 - term3_8 - term4_8
ak8 = np.sqrt(ak_sqr_8)



m=16
kdx = 2*np.pi / m
# calculate the amplutide
term1_16 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
term2_16 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
term3_16 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
term4_16 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
ak_sqr_16 = term1_16 + term2_16 - term3_16 - term4_16
ak16 = np.sqrt(ak_sqr_16)

# # %%
# ##### plot for different Cr values


# Cr = 0.9

# m = np.arange(1,20,0.01)
# kdx = 2*np.pi / m


# term1 = (1 - (Cr**2 /4) + (Cr**2 /4)*(1 - 2*(np.sin(kdx))**2 ) )**2
# term2 = ( -(Cr**4 / 6) + (Cr**6 / 48)  ) * (  3*(np.sin(kdx))**2   - 4*(np.sin(kdx))**4  )
# term3 = (Cr - (Cr**3 / 8)) * ((np.sin(kdx))**2) 
# term4 = (Cr **6 / 144) * ((3*(np.sin(kdx))) - (4*(np.sin(kdx))**3))**2
# ak_sqr2 = term1 + term2 - term3 - term4
# ak2 = np.sqrt(ak_sqr2)

# plt.plot(m, ak2)
# plt.show()


# %% Plot for B3


fig, ax = plt.subplots(1,1, figsize=myfigsize)
ax.plot(Cr,ak2, label = 'm = 2')
ax.plot(Cr,ak25, label = 'm = 2.5')
ax.plot(Cr,ak3, label = 'm = 3')
ax.plot(Cr,ak4, label = 'm = 4')
ax.plot(Cr,ak8, label = 'm = 8')
ax.plot(Cr,ak16, label = 'm = 16')
#ax.plot(xvals,c_ideal, color='r', label = 'exact final solution')
ax.set_xlabel("Cr")
ax.set_ylabel("Amplitude")
#plt.title("")
# ax.set_ylim(-5,95)
plt.legend()
#plt.show()
plt.savefig(fig_dir+'B3'+'_'+run_date+'.png')