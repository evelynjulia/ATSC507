# %%
# -*- coding: utf-8 -*-
# Eve Wicksteed
# HW 3
# NWP - ATSC 507

'''
HW 4 - (Due the week after Spring Break. ).  Given the following function

T(°C) =A * (c*m*∆t + Tref - Tref_o) * (Tref_o - c*m*∆t)

with Tref_o = 2,  A = 1,  c = 1.5,  ∆t = 1,  and  t = m*∆t.

This was the basis for the worksheet that I handed out today in class (the coloured 
curved lines in the fig below), where I used variable Tref in the range of 2 to 6, 
and m in the range of 0 to 1. 
Given the info above, the function that you should apply to the finite difference 
methods (a) - (d) below is:

∂T/∂t = f(t, T) = 1.5 * { 2 – 1.5*t – [ T / (2 – 1.5*t) ] }

[Hint: To get the eq above for f(t, T), I first solved the eq above for Tref(T, t).  
Then I  analytically found ∂T/∂t from the first equation above, and substituted in 
the expression for Tref.  This takes advantage of the fact that Tref is constant 
along any of the curves in the fig below.]


Please start from initial condition of T = 2 degC as we did in class, but compute 
using any method (excel, matlab, R, python, etc) the new T (degC) at 1 timestep 
(1∆t) ahead using:

a) Euler forward
b) RK2
c) RK3
d) RK4
e) Which one gave an answer closest to the actual analytical answer as 
given by the function above?   

[Note: do NOT use the 1-D model from the previous HW for this.]


'''

# %%


import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import interpolate
import pandas as pd
import math 


fig_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW4/'
#data_dir = '/Users/ewicksteed/Documents/Eve/NWP/HW4/'

run_date = dt.datetime.now().strftime('%y%m%d')

#myfigsize = (15,9)


# %% Set constants

#∂T/∂t 
#f = 1.5 * ( 2 – 1.5*t – ( T / (2 – 1.5*t) ) )
T_n = 2
del_t = 1
t = 0

f_tn = 1.5 * ( 2 -(1.5*t)-( T_n / (2-1.5*t) ) )

# %% a) Euler forward

T_np1 = (f_tn * del_t ) + T_n 

# gives T_np1 = 3.5


# %% b) RK2 

# f_tn = 1.5 * ( 2 -(1.5*tn)-( T_n / (2-1.5*tn) ) )
T_star_RK2 = T_n + ( (del_t/2) * f_tn )
f_tstar_RK2 = 1.5 * ( 2 -(1.5*(tn+(del_t/2)))-( T_star_RK2 / (2-1.5*(tn+(del_t/2))) ) )
T_np1_RK2 = T_n + ( del_t * f_tstar_RK2 )


# gives T_np1 = 0.5749999999999997



# %% c) RK3

# f_tn = 1.5 * ( 2 -(1.5*tn)-( T_n / (2-1.5*tn) ) )
T_star_RK3 = T_n + ( (del_t/3) * f_tn )
f_tstar_RK3 = 1.5 * ( 2 -(1.5*(tn+(del_t/3)))-( T_star_RK3 / (2-1.5*(tn+(del_t/3))) ) )
T_star2_RK3 = T_n + ( (del_t/2) * f_tstar_RK3 )
f_tstar2_RK3 = 1.5 * ( 2 -(1.5*(tn+(del_t/2)))-( T_star2_RK3 / (2-1.5*(tn+(del_t/2))) ) )
T_np1_RK3 = T_n + ( del_t * f_tstar2_RK3 )

# gives T_np1 = 1.625


# %% RK4

# f_tn = 1.5 * ( 2 -(1.5*tn)-( T_n / (2-1.5*tn) ) )

k1 = 
k2 = 
k3 = 
