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
my_out_dir = str(current_dir)+'/HW11/'


# %% Set up data

# Each field (i.e., each weather map) below covers an
# area from North to South and West to East.

anal = np.array([[5.2, 5.3, 5.4, 5.3],[5.3, 5.4, 5.5, 5.4],[5.4, 5.5, 5.6, 5.5],[5.5, 5.6, 5.7, 5.6],[5.6, 5.7, 5.8, 5.7]])

fcst = np.array([[5.3, 5.4, 5.5, 5.4],[5.5, 5.4, 5.5, 5.6],[5.6, 5.6, 5.6, 5.6],[5.8, 5.7, 5.6, 5.7],[5.9, 5.8, 5.7, 5.8]])

verif = np.array([[5.3, 5.3, 5.3, 5.4],[5.4, 5.3, 5.4, 5.5],[5.5, 5.4, 5.5, 5.5],[5.7, 5.5, 5.6, 5.6],[5.8, 5.7, 5.6, 5.6]])

clim = np.array([[5.4, 5.4, 5.4, 5.4],[5.4, 5.4, 5.4, 5.4],[5.5, 5.5, 5.5, 5.5],[5.6, 5.6, 5.6, 5.6],[5.7, 5.7, 5.7, 5.7]])

# %% A19



# %% a. Mean forecast error

ME_fcst = np.mean(fcst-verif)

print("the mean forecast error is", round(ME_fcst, 3))

# %% b. mean persistence error

ME_persist = np.mean(anal-verif)

print("the mean persistence error is", round(ME_persist, 3))

# %% c. mean absolute forecast error

MAE_fcst = np.mean(np.abs(fcst-verif))

print("the mean absolute forecast error is", round(MAE_fcst, 3))

# %% d. mean squared forecast error

MSE_fcst = np.mean((fcst-verif)**2)

print("the mean squared forecast error is", round(MSE_fcst, 3))

# %% e. mean squared climatology error

MSE_clim = np.mean((clim-verif)**2)

print("the mean squared forecast error is", round(MSE_clim, 3))


# %% f. mean squared forecast error skill score

MSE_ref = MSE_clim
MSESS_fcst = 1 - (MSE_fcst/MSE_ref)

print("the mean squared forecast error skill score is", round(MSESS_fcst, 3))


# %% g. RMS forecast error

RMSE_fcst = np.sqrt(MSE_fcst)

print("the root mean squared forecast error is", round(RMSE_fcst, 3))


# %% h. correlation coefficient between forecast and verification

fprime = fcst - np.mean(fcst)
vprime = verif - np.mean(verif)

PCC_fv = (np.mean(fprime*vprime)) / ( (np.sqrt(np.mean(fprime**2))) * np.sqrt(np.mean(vprime**2)) )

print("the Pearson correlation coefficient between forecast and verification is", round(PCC_fv, 3))

# %% i. forecast anomaly correlation

AC = np.mean( ((fcst - clim)- np.mean(fcst - clim)) * ((verif - clim)- np.mean(verif - clim)) )  / np.sqrt( np.mean(((fcst - clim) - np.mean(fcst - clim))**2) * np.mean(((verif - clim) - np.mean(verif - clim))**2) )

print("the forecast anomaly correlation is", round(AC,3))

# %% j. persistence anomaly correlation

AC_persist = np.mean( ((anal - clim)- np.mean(anal - clim)) * ((verif - clim)- np.mean(verif - clim)) )  / np.sqrt( np.mean(((anal - clim) - np.mean(anal - clim))**2) * np.mean(((verif - clim) - np.mean(verif - clim))**2) )

print("the persistance anomaly correlation is", round(AC_persist,3))


# %% k. Draw height contours by hand for each field, to show locations of ridges and troughs.

plt.contour(anal, linewidths=4)
#plt.show()
plt.savefig(my_out_dir+'anal'+'.png')

plt.contour(fcst, linewidths=4)
#plt.show()
plt.savefig(my_out_dir+'fcst'+'.png')

plt.contour(verif, linewidths=4)
#plt.show()
plt.savefig(my_out_dir+'verif'+'.png')

plt.contour(clim, linewidths=4)
#plt.show()
plt.savefig(my_out_dir+'clim'+'.png')

#np.savetxt(my_fig_dir"foo.csv", a, delimiter=",")



# %% save values to pandas df

df = pd.DataFrame()

df['ME_fcst']= ME_fcst

data = {"ME_fcst": [ME_fcst],
'ME_persist': [ME_persist],
'MAE_fcst': [MAE_fcst],
'MSE_fcst': [MSE_fcst],
'MSE_clim': [MSE_clim],
'MSESS_fcst': [MSESS_fcst],
'RMSE_fcst': [RMSE_fcst],
'PCC_fv': [PCC_fv],
'AC': [AC],
'AC_persist': [AC_persist]
}

df = pd.DataFrame(data) 

df.to_csv(my_out_dir+"error_metrics.csv")



# %% A 20

# have the following contingincy table:

a = 150
b = 65
c = 50
d = 100

n = a + b + c + d

E = ((a+b)/n)* ((a+c)/n) + ((d+b)/n)*((d+c)/n)

a_random = ((a+b)*(a+c)) / (n)

# %%
# So now calculate all binary verification stats

bias = (a+b) / (a+c)

portion_correct = (a + d) / n

HSS = (portion_correct - E) / (1-E)

hit_rate = a / (a + c)

FA_rate =  b/ (b+d)# false alarm rate

FA_ratio = b / (a+b) # false alarm ratio

TSS = hit_rate - FA_rate # true skill score

CSI = a / (a+b+c) # critical success index

GSS = (a - a_random) / (a - a_random + b + c) # gilber skill score

# %% add to df

a20df = pd.DataFrame()


a20data = {"a": [a],
'b': [b],
'c': [c],
'd': [d],
'n': [n],
'E': [E],
'a_random': [a_random],
'bias': [bias],
'portion_correct': [portion_correct],
'HSS': [HSS],
'hit_rate': [hit_rate],
'FA_rate': [FA_rate],
'FA_ratio': [FA_ratio],
'TSS': [TSS],
'CSI': [CSI],
'GSS': [GSS]
}

a20df = pd.DataFrame(a20data) 

a20df.to_csv(my_out_dir+"binary_verif_stats.csv")


# %% A 21

# a, b, c, and d are the same

o = 0.5 # %

cost = 5000  # protective cost
loss = 50000


e_clim = min(cost, o*loss)
e_forcast = (a/n)*cost + (b/n)*cost + (c/n)*loss
e_perfect = o*cost

Value = (e_clim - e_forcast) / (e_clim - e_perfect)

# or alternatively calculate like this:

r_cl = cost / loss
v = (min(r_cl, o) - (FA_rate * r_cl*(1-o)) + (hit_rate*(1-r_cl)*o) - o ) / (min(r_cl, o) - o*r_cl)


a21df = pd.DataFrame()

a21data = {"cost": [cost],
'loss': [loss],
'clim_freq': [o],
'e_clim': [e_clim],
'e_forcast': [e_forcast],
'e_perfect': [e_perfect],
'Value': [Value], 
'r_cl': [r_cl], 
'v_eq_20_55': [v]}

a21df = pd.DataFrame(a21data) 
a21df.to_csv(my_out_dir+"a21_v2.csv")


# %% a 22

k = np.arange(0,20,1)
pk = np.append(np.arange(0.9, 0, -0.05), np.array([0.02, 0])) 
ok = np.array([1,1,0,1,1,1,0,1,0,1,0,0,1,0,0,1,0,0,0,0])

N = 20

# %% A

# Brier skill score:

BSS = 1 - ( (sum((pk-ok)**2)) /  ( (sum(ok)) * (N - sum(ok))  ) ) 

# %% B Reliabilty diagram

# get bins:
dp = 0.2
bins = np.arange(0, 1.2, dp)

# find which obs goes in which bin... 
j = np.ceil((pk / dp))

# and the number of obs in each bin (nj)
unique_j, nj = np.unique(j, return_counts=True)

# get the observed probabilities that happened:
oj = j[ok==1]
unique_oj, n_oj = np.unique(oj, return_counts=True)

# in this case we need to include in the array that 0 observations fall in bin 0
final_noj = np.append(0, n_oj)

# n_oj / nj
portion_verified = final_noj / nj


# %% create dataframes


big_tbl_data = {"k": k,
'pk': pk,
'ok': ok,
'bin': j}

big_tbl = pd.DataFrame(big_tbl_data) 
big_tbl.to_csv(my_out_dir+"a22_big_tbl.csv")


sml_tbl_data = {"j": unique_j,
'pj': bins,
'nj': nj,
'noj': final_noj,
'noj/nj': portion_verified}

sml_tbl = pd.DataFrame(sml_tbl_data) 
sml_tbl.to_csv(my_out_dir+"a22_sml_tbl.csv")

# %% plot

# 1-1 line
x = np.linspace(0,1,100)

plt.plot(x, x, 'r--')
plt.plot(bins, portion_verified, ".-")
plt.xlabel("pj")
plt.ylabel("noj / nj")
plt.title("Reliability diagram")
#plt.show()
plt.savefig(my_out_dir+'a22_reliability'+'.png')


# %% C find reliability brier skill score

# pj (bin centre) = bins
# nj = nj
# noj = final_noj

BSS_r = sum(((nj*bins)-final_noj)**2)  / (sum(ok)) * (N - sum(ok))

results_a22_data = {"BSS": [BSS],
'BSS_r': [BSS_r]}

results_a22 = pd.DataFrame(results_a22_data) 
results_a22.to_csv(my_out_dir+"a22_results.csv")

# %% A 23 ROC diagram

# part c
# a23_hit_rate = [1.00,1.00,1.00,1.00,1.00,1.00,0.77,0.54,0.38,0.23,0.08]
# a23_false_alarm_rate = [0.71,0.71,0.53,0.35,0.18,0.00,0.00,0.00,0.00,0.00,0.00]

# part a
a23_hit_rate = [1.00,1.00,1.00,0.86,0.85,0.85,0.77,0.69,0.62,0.38,0.15]
a23_false_alarm_rate = [1.00,0.71,0.41,0.25,0.24,0.18,0.12,0.06,0.00,0.00,0.00]


plt.plot(x,x)
plt.plot(a23_false_alarm_rate, a23_hit_rate, ".-")
plt.xlabel("False alarm rate")
plt.ylabel("Hit rate")
plt.title("ROC diagram")
#plt.show()
plt.savefig(my_out_dir+'a23_ROC_a'+'.png')


# %% area under curve:

import numpy as np
from scipy.integrate import simps
from numpy import trapz


# The y values.  A numpy array is used here,
# but a python list could also be used.
y = a23_hit_rate

# Compute the area using the composite trapezoidal rule.
area = trapz(np.flip(a23_hit_rate), x = np.flip(a23_false_alarm_rate))
print("area =", area)

# Compute the area using the composite Simpson's rule.
area = simps(a23_hit_rate, x=np.flip(a23_false_alarm_rate))
print("area =", area)


# from sklearn import metrics
# y = np.array([1, 1, 2, 2])
# pred = np.array([0.1, 0.4, 0.35, 0.8])
# fpr, tpr, thresholds = metrics.roc_curve(a23_hit_rate, a23_false_alarm_rate, pos_label=2)
# metrics.auc(fpr, tpr)

# obs_y = [1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,0,1,0,1]

# probs = [50,20,20,60,50,20,30,90,40,30,100,10,0,10,80,60,70,90,80,70,10,10,0,0,80,0,0,100,10,90]
# calculate roc curve
# fpr, tpr, thresholds = metrics.roc_curve(obs_y, probs) # false alarms, # hit rate

# plt.plot(fpr, tpr)
# plt.show()

# calculate AUC
# auc = metrics.roc_auc_score(obs_y, probs)
# print('AUC: %.3f' % auc)