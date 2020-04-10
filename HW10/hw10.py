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
import math

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


def get_area(a,b,m):

    # define weights and zero values
    if m==2:
        ek = np.array([0.5773502692, -0.5773502692])
        w = np.array([1,1])
    elif m==4:
        ek = np.array([0.3399810436, -0.3399810436, 0.8611363116, -0.8611363116])
        w = np.array([0.6521451549, 0.6521451549, 0.3478548451, 0.3478548451])
    elif m==6:
        ek = np.array([0.2386191861, -0.2386191861, 0.6612093865, -0.6612093865, 0.9324695142, -0.9324695142])
        w = np.array([0.4679139346, 0.4679139346, 0.3607615730, 0.3607615730, 0.1713244924, 0.1713244924])
    elif m==8:
        ek = np.array([0.1834346425, -0.1834346425, 0.5255324099, -0.5255324099, 0.7966664774, -0.7966664774, 0.9602898565, -0.9602898565])
        w = np.array([0.3626837834, 0.3626837834, 0.3137066459, 0.3137066459, 0.2223810345, 0.2223810345, 0.1012285363, 0.1012285363])

    # get xk values
    xk = get_xk(a,b,ek)

    # then get the function values, fxk:
    fxk = func(xk)

    # then integrate to get area:
    area = gaussInt(a,b,w,fxk)

    return area




# integrate between 0 and 20
a = 0
b = 20

area_m2 = get_area(a,b,m=2)
area_m4 = get_area(a,b,m=4)
area_m6 = get_area(a,b,m=6)
area_m8 = get_area(a,b,m=8)


data = {"area, m=2": [area_m2],
'area, m=4': [area_m4],
'area, m=6': [area_m6],
'area, m=8': [area_m8]
}

df = pd.DataFrame(data) 
df.to_csv(out_dir+"area_from_gauss_quad.csv")


# %% Question 3

# Plot eq. (4.22) of Coiffier similar to his Fig 4.2, but for:  
# (a) m=0 with n=1 to 4, 
# (b) m=1 with n=1 to5, and 
# (c) m=2 with n=2 to 6. 


u = np.linspace(-1,1,100) # from -pi to pi

n = 1
m = 0




def get_deriv(m,n,u):
    if m == 0:
        if n == 1:
            ddu = 2*u
        elif n== 2:
            ddu = (12* u**2) -4
        elif n==3:
            ddu = 120*u**3 -72*u
        elif n ==4:
            ddu = 1680*u**4 - 1440*u**2 + 144
    
    return ddu


u = np.linspace(-1,1,100)

# start of for m = 0
m = 0
part_an = np.array([1,2,3,4])
part_a = np.empty((len(part_an), len(u)))


for n in part_an:
    # calculate the function
    t1 = np.sqrt((2*n +1)*(math.factorial(n-m)/math.factorial(n+m)))
    t2 = ((1-u**2)**(m/2) ) / ((2**n) * math.factorial(n))
    t3 = get_deriv(m,n,u)
    Pmn =  t1 * t2 * t3 
    # append answer to array
    part_a[n-1] = Pmn

# y0 = np.zeros(x.shape)
labels = ['n=1', 'n=2', 'n=3', 'n=4']

fig, ax = plt.subplots(1,1, figsize=myfigsize)
lines = ax.plot(u, part_a.T)
ax.legend(lines, labels)
ax.plot(u,y0, 'k')
ax.set_xlabel("u")
ax.set_ylabel("P(u)")
plt.title('m = 0')
#plt.show()
plt.savefig(out_dir+'q3a'+'.png')




t1 = np.sqrt((2*n +1)*(math.factorial(n-m)/math.factorial(n+m)))
t2 = ((1-u**2)**(m/2) ) / ((2**n) * math.factorial(n))
t3 = get_deriv(m,n,u)


Pmn =  t1 * t2 * t3 