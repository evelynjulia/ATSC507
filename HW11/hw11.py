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

# %% create error functions



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