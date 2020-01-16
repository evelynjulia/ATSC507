# Eve Wicksteed
# HW 1
# NWP - ASC 507

'''
Use any computing method you want.   This problem can be done on a spreadsheet (with
difficulty).  Or it might be easier to write your program in any of MatLab, R, python, or
fortran, etc.
============
Given:
2-D Domain:  xkm = 0 to 1000 km ,   zkm = 0 to 30 km
for your calculations, use dx = 20 km, and dz = 1 km or finer (I use dz=0.001 km).

Use the WRF definition of eta (eqs. 2.2 - 2.5 in WRF-ARW4 tech note 2019), where pi is dry
hydrostatic pressure (written as pd in the 2019 tech note).
Let:
pi_top = 2 kPa
eta_c = 0.3  . This is the eta value, above which eta becomes a pure pressure coordinate in
this hybrid system.

For any location x(km), vertical profiles of temperature are given by
T(degC) =   (40 - 0.08*xkm)  -  6.5*zkm     for 0 < zkm < 12
T(degC) =   (40 - 0.08*xkm)  -  6.5*12       for 12 < zkm   (i.e., isothermal above 12 km)
for zkm = height above sea level

Actual Mean Sea-Level (at z = 0) pressure is Pmsl = 95kPa + 0.01*xkm
Use the info above to determine P vs z, by iterating up from sea-level using hypsometric eq.
P2 = P1 * exp[ (z1-z2) / (a*Tkelvin) ]   where a = 0.0293 km/K.

This gives you the hydrostatic pressure (pi = P = pd) as a function of height.

NOTE: in WRF4 eq. (2.2), they use a REFERENCE sea-level pressure of Po = 100 kPa.  Yes, you
should use this reference value in eq. (2.2), even though you use the actual Pmsl pressure as
a basis for all your other calculations.

Intersecting the background pressure field that you just determined is the topography,
given in height above sera level by:

Zground_km = 1 km + (1km)*cos[ 2*(3.14159)*(xkm-500km) / 500km ]     for  250km < x < 750 km
and Zground = 0 elsewhere.
'''

import numpy as np
import matplotlib.pyplot as plt


#%% Part 1
'''
Find:  On an x-z graph, plot the altitudes (km) of the following isobaric surfaces:
100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2 kPa.
On the same plot, plot the altitude of Zground.
'''


# Use the hydrostatic equation:
# dP = rho * g * dz
# dz = dP/(rho*g)

# 2-D Domain:  xkm = 0 to 1000 km ,   zkm = 0 to 30 km
# for your calculations, use dx = 20 km
dx = 20
x = np.arange(0,2000,dx)


# dx = 20
# x = np.arange(0,2000,dx)

# zkm = 0 to 30 km
# for your calculations, use dx = 20 km, and dz = 1 km or finer (I use dz=0.001 km).


rho = 1.2754 # kg/m³ --> dry air density
g = -9.81
#dz = [] # need tp calculate this
pressure = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2]) # kPa

dp = np.diff(pressure)

calc_dz = dp/(rho*g)


# Zground_km = 1 km + (1km)*cos[ 2*(3.14159)*(xkm-500km) / 500km ]     for  250km < x < 750 km
# and Zground = 0 elsewhere.

z = np.array([0])

for i in range (0,10):
    z = np.append(z, z[i]+calc_dz[i+1])


################################################# attempt 2
dx = 20
x = np.arange(0,2000,dx)

dz = 0.001
z = np.arange(0,30,dz)

Pmsl = 95 + 0.01*x



T = np.empty((len(x),len(z)))

for i in range(len(x)):
    for j in range(len(z)):
        if j <12:
            print(T[i,j])
            print(i,j)
            T[i,j] = (40 - 0.08*x[i])  -  6.5*z[j]
            print(T[i,j])
        else:
            T[i,j] = (40 - 0.08*x[i])  -  6.5*12

Tkelvin = T + 273

'''
Actual Mean Sea-Level (at z = 0) pressure is Pmsl = 95kPa + 0.01*xkm
Use the info above to determine P vs z, by iterating up from sea-level using hypsometric eq.
P2 = P1 * exp[ (z1-z2) / (a*Tkelvin) ]   where a = 0.0293 km/K.
'''
a = 0.0293 # km/K.


list_of_pressure = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2]
P = np.empty((len(x),len(z)))
P[:,0] = Pmsl

for i in range(len(x)):
    for j in range(len(z)-1):
        P[:,j+1] = P[:,j] * [ np.exp(z[j] - z[(j+1)]) / (a*Tkelvin[:,j+1]) ]


z_at_P = np.empty((len(x),len(list_of_pressure)))
for i in range(len(x)):
    for j in range(len(list_of_pressure)-1):
        z_at_P[j+1]= (a*Tkelvin* np.log(list_of_pressure[i]/list_of_pressure[i+1]) ) +z_at_P[j]


#%% Part 2
'''
Interpolate to find the Psurface (kPa) pressure at Zground. Namely, it is the pressure
that corresponds to eta = 1.
This pressure that you use to find eta in exercises (3) & (4).
Present the results in a table, where:
row1 = x, (km)
row2 = Zground, (km)
row3=Psfc (kPa)
'''

#%% Part 3
'''
Create a new P-x graph, on which you plot lines of constant eta, for the eta values listed
below.  Namely, it should look something like WRF4 figure 2.1b, but with the more realistic
meteorology that I prescribed above.  Also, like that figure, plot pressure P on the vertical
axis in reversed order (highest pressure at the bottom of the figure), but don't use a log
https://www.eoas.ubc.ca/courses/atsc507/A507Assignments/HW1-hybrid-eta_exercise-v2.txt 1/2
1/15/2020 https://www.eoas.ubc.ca/courses/atsc507/A507Assignments/HW1-hybrid-eta_exercise-v2.txt
scale for P.
CAUTION: when calculating the values of B to use in WRF4 eq. (2.2), be advised that WRF4 eq.
(2.3) applies only for eta > eta_c.
eta =
0
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.85
0.9
0.95
1
'''


#%% Part 4
'''
Create a new z-x graph, on which you plot the z altitudes of the constant eta lines for
the same eta values as in part (3) above. Make use of the hypsometric eq to find the heights
z at the pressure levels that correspond to the requested eta values.
'''