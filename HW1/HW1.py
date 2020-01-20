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
given in height above sea level by:

Zground_km = 1 km + (1km)*cos[ 2*(3.14159)*(xkm-500km) / 500km ]     for  250km < x < 750 km
and Zground = 0 elsewhere.
'''

#%% Set some constants and directories

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import interpolate
import pandas as pd

fig_dir = '/Users/ewicksteed/Documents/Eve/507/HW1/'
data_dir = '/Users/ewicksteed/Documents/Eve/507/HW1/'

# fig_dir = '/Users/catherinemathews/UBC/NWP'
# data_dir = '/Users/catherinemathews/UBC/NWP'


run_date = dt.datetime.now().strftime('%y%m%d')

myfigsize = (15,9)


#%% Part 1
'''
Find:  On an x-z graph, plot the altitudes (km) of the following isobaric surfaces:
100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2 kPa.
On the same plot, plot the altitude of Zground.
'''
dx = 20
x = np.arange(0,1020,dx)

dz = 1
z = np.arange(0,31,dz)

Pmsl = 95 + 0.01*x

# set topography
'''
Zground_km = 1 km + (1km)*cos[ 2*(3.14159)*(xkm-500km) / 500km ]     for  250km < x < 750 km
and Zground = 0 elsewhere.
'''
zground = np.empty(len(x))
for i in range(len(x)):
    if x[i] >250 and x[i] < 750:
        zground[i] = 1+ (1 * np.cos( 2*(3.14159)*(x[i]-500) / 500 ))
    else:
        zground[i] = 0


T = np.empty((len(x),len(z)))

for j in range(len(z)):
    if z[j]<12:
        #print(z[j], "is less than 12")
        T[:,j] = (40 - 0.08*x)  -  6.5*z[j]
    else: 
        #print(z[j], "is more than 12")
        T[:,j] = (40 - 0.08*x)  -  6.5*12


Tkelvin = T + 273

'''
Actual Mean Sea-Level (at z = 0) pressure is Pmsl = 95kPa + 0.01*xkm
Use the info above to determine P vs z, by iterating up from sea-level using hypsometric eq.
P2 = P1 * exp[ (z1-z2) / (a*Tkelvin) ]   where a = 0.0293 km/K.
'''
a = 0.0293 # km/K.


# list_of_pressure = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 2]
P = np.empty((len(x),len(z)))
P[:,0] = Pmsl

for j in range(len(z)-1):
    P[:,j+1] = P[:,j] *  np.exp( (z[j] - z[(j+1)]) / (a*Tkelvin[:,j+1]) )

levs = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90,100]
lev_labels = ['2 kPa', '5 kPa', '10 kPa', '20 kPa', '30 kPa', '40 kPa', '50 kPa', '60 kPa', '70 kPa', '80 kPa', '90 kPa','100 kPa']

fig, ax = plt.subplots(1,1, figsize=myfigsize)
#P_plot = ax.contour(P.T, levs)
P_plot = ax.contour(x,z,P.T, levs)
ax.clabel(P_plot, fmt = '%1.0f kPa')
ax.fill(x,zground)
ax.set_xlabel('x-domain (km)')
ax.set_ylabel('Height (km)')
plt.title('Part 1: X-Z graph with altitudes of isobaric surfaces')
#plt.show()
plt.savefig(fig_dir+run_date+'part1_plot'+'.png')



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

p_sfc = []
interp_levs = zground.copy()
for i in range(len(x)):
    interp_eq = interpolate.interp1d(z, P[i])
    if interp_levs[i] > 0:
        print('need to interpolate at', i)
        p_sfc_i = interp_eq(interp_levs[i])
        print(p_sfc_i)
    else:
        p_sfc_i = Pmsl[i]
    p_sfc = np.append(p_sfc, p_sfc_i)

### could also do this by using hypsometric equation:
p_sfc_hyp = P[:,0] *  np.exp( (z[0] - zground) / (a*Tkelvin[:,1]) )

# print in table:
# (create pd dataframe)
part_2_table = pd.DataFrame()
part_2_table['x'] = x
part_2_table['z_ground'] = zground
part_2_table['p_sfc_(interpolation)'] = p_sfc
part_2_table['p_sfc_(hypsometric_eqn)'] = p_sfc_hyp

# save to csv
part_2_table.to_csv(data_dir+run_date+'Part_2_table.csv', sep=',')


part_2_table2 = pd.DataFrame()
part_2_table2['x'] = x
part_2_table2['z_ground'] = zground
part_2_table2['p_sfc_(hypsometric_eqn)'] = p_sfc_hyp
part_2_table2.to_csv(data_dir+run_date+'Part_2_table2.csv', sep=',')



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

# Given:
eta_c = 0.3 # above this value eta becomes a pure pressure coordinate
eta = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1])
pt = 2 # kPa p top

ps = p_sfc_hyp # p surface
p0 = Pmsl # reference sea-level pressure

# Equations:

# eqns 2.5
c1 = (2 * (eta_c**2)) / ((1-eta_c)**3)
c2 = ( (- eta_c) * (4 + eta_c + eta_c**2) ) / ((1-eta_c)**3)
c3 = (2 * (1 + eta_c + (eta_c**2) ) ) / ((1-eta_c)**3)
c4 = (- (1 + eta_c) ) / ((1-eta_c)**3)

# For all eta
B_eta = c1 + c2*(eta) + c3*(eta**2) +c4*(eta**3)  # eqn 2.3
# replace with B=0 for eta <= eta_c
ind1 = np.where(eta <= eta_c)
B_eta[ind1] = 0

# Hybrid sigma-pressure vertical coordinate
pd = np.empty((len(x), len(eta)))
for i in range(len(x)):
    pd[i] = B_eta*(ps[i]-pt) + (eta - B_eta)*(p0[i]-pt)+pt   # eqn 2.2

## Plot
labels = ['eta 0', 'eta 0.1', 'eta 0.2', 'eta 0.3', 'eta 0.4', 'eta 0.5', 'eta 0.6', 'eta 0.7', 'eta 0.8', 'eta 0.85', 'eta 0.9', 'eta 0.95', 'eta 1 = sfc pressure']

fig, ax = plt.subplots(1,1, figsize=myfigsize)
#P_plot = ax.contour(P.T, levs)
eta_plot = ax.plot(x,pd)
#ax.fill(x,p_sfc_hyp)
ax.invert_yaxis()
ax.set_xlabel('x-domain (km)')
ax.set_ylabel('Pressure (kPa)')
ax2 = ax.twinx()
ax2.set_ylabel('Height (km)')
ax2.fill(x,zground)
ax2.set_ylim(0,30)
plt.legend(eta_plot, labels)
#fig.tight_layout()
plt.title('Part 3: X-P graph with pressures of constant eta values')
plt.savefig(fig_dir+run_date+'part3_plot'+'.png')
#plt.show()

fig, ax = plt.subplots(1,1, figsize=myfigsize)
#P_plot = ax.contour(P.T, levs)
eta_plot = ax.plot(x,pd)
#ax.fill(x,p_sfc_hyp)
ax.invert_yaxis()
ax.set_xlabel('x-domain (km)')
ax.set_ylabel('Pressure (kPa)')
# ax2 = ax.twinx()
# ax2.set_ylabel('Height (km)')
# ax2.fill(x,zground)
# ax2.set_ylim(0,30)
# ax.plot(x,p_sfc_hyp, label='surface pressure')
plt.legend(eta_plot, labels)
#fig.tight_layout()
plt.title('Part 3: X-P graph with pressures of constant eta values')
plt.savefig(fig_dir+run_date+'part3_plot2'+'.png')
#plt.show()


#%% Part 4
'''
Create a new z-x graph, on which you plot the z altitudes of the constant eta lines for
the same eta values as in part (3) above. Make use of the hypsometric eq to find the heights
z at the pressure levels that correspond to the requested eta values.
'''

# so now we have Pd which is the pressure levels and we need to find z from pd:

z_eta = np.empty((len(x),len(eta)))
z_eta[:,0] = zground
for lev in range(pd.shape[1]-1):
    # z2 = (a*Tkelvin * ln( P1/P2)) + z1
    # p1 = pd[:,lev+1] because of the order of the pd array (has 2kpa first)
    # p2 = pd[:,lev]
    # because pd[0] = array([ 2.  , 11.3 , 20.6 , 29.9 , 39.2 , 48.5 , 57.8 , 67.1 , 76.4 , 81.05, 85.7 , 90.35, 95.  ])
    # z_eta[:,lev+1]= (a*Tkelvin[:,lev]* np.log(pd[:,lev+1]/pd[:,lev]) ) +z_eta[:,lev] 
    z_eta[:,lev+1]= (a*Tkelvin[:,lev]* np.log(pd[:,(12-lev)]/pd[:,(11-lev)]) ) +z_eta[:,lev]

labels2 = ['eta 1','eta 0.95','eta 0.9','eta 0.85','eta 0.8','eta 0.7','eta 0.6','eta 0.5','eta 0.4','eta 0.3','eta 0.2','eta 0.1','eta 0']

fig, ax = plt.subplots(1,1, figsize=myfigsize)
eta_plot2 = ax.plot(x,z_eta)
ax.fill(x,zground)
ax.set_xlabel('x-domain (km)')
ax.set_ylabel('Height (km)')
plt.legend(eta_plot2, labels2)
plt.title('Part 4: X-Z graph with altitudes of constant eta values')
#plt.show()
plt.savefig(fig_dir+run_date+'part4_plot'+'.png')


