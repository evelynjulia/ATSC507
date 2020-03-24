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

#%%

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
