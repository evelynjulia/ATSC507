# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 11
# NWP - ATSC 507


# %% load libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

current_dir = os.getcwd()
out_dir = str(current_dir)+'/HW10/'


myfigsize = (9,4)

# %%

x = np.linspace(0,20,100)
y = 0.5*x + x* np.sin((2*np.pi/10)*x)

y0 = np.zeros(x.shape)


fig, ax = plt.subplots(1,1, figsize=myfigsize)
ax.plot(x,y)
ax.plot(x,y0)
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.show()
plt.savefig(out_dir+'function'+'.png')

# %% 

# (c) use Gauss quadrature to numerically integrate it (using the eqs 
# and tables in the handout, not any built-in integration function) for the 
# following number of key points (m or n=):  (i) 2 ,  (ii) 4 , (iii) 6 , (iv) 8 , 
# and discuss how Gaussian quadrature converges to the exact solution.  Show your 
# work on your spreadsheet, or matlab, or your computer program.

# Lets define the function as a function:
def func(x):
    y = 0.5*x + x* np.sin((2*np.pi/10)*x)
    return y


# calculate xk values
def get_xk(a,b,ek):
    xk = (b+a)/2 + ((b-a)/2)*ek
    return xk


# then integrate with gaussian quadrature
def gaussInt(a,b,wk,fxk):
    I = ((b-a)/2) * sum(wk*fxk)
    return I


# integrate between 0 and 20
a = 0
b = 20


# Get the weights
#(i) 2 ,  (ii) 4 , (iii) 6 , (iv) 8 
# M2 weights:
# Ek = +- 0.5773502692
# Wk = 1
ek2 = np.array([0.5773502692, -0.5773502692])
w2 = np.array([1,1])

ek4 = np.array([0.3399810436, -0.3399810436, 0.8611363116, -0.8611363116])
w4 = np.array([0.6521451549, 0.6521451549, 0.3478548451, 0.3478548451])

ek6 = np.array([0.2386191861, -0.2386191861, 0.6612093865, -0.6612093865, 0.9324695142, -0.9324695142])
w6 = np.array([0.4679139346, 0.4679139346, 0.3607615730, 0.3607615730, 0.1713244924, 0.1713244924])

ek8 = np.array([0.1834346425, -0.1834346425, 0.5255324099, -0.5255324099, 0.7966664774, -0.7966664774, 0.9602898565, -0.9602898565])
w8 = np.array([0.3626837834, 0.3626837834, 0.3137066459, 0.3137066459, 0.2223810345, 0.2223810345, 0.1012285363, 0.1012285363])


# get xk values
xk_m2 = get_xk(a,b,ek2)
xk_m4 = get_xk(a,b,ek4)
xk_m6 = get_xk(a,b,ek6)
xk_m8 = get_xk(a,b,ek8)

# then get the function values, fxk:
fxk_m2 = func(xk_m2)
fxk_m4 = func(xk_m4)
fxk_m6 = func(xk_m6)
fxk_m8 = func(xk_m8)

# then integrate:
area_m2 = gaussInt(a,b,w2,fxk_m2)
area_m4 = gaussInt(a,b,w4,fxk_m4)
area_m6 = gaussInt(a,b,w6,fxk_m6)
area_m8 = gaussInt(a,b,w8,fxk_m8)


data = {"area, m=2": [area_m2],
'area, m=4': [area_m4],
'area, m=6': [area_m6],
'area, m=8': [area_m8]
}

df = pd.DataFrame(data) 
df.to_csv(out_dir+"area_from_gauss_quad.csv")