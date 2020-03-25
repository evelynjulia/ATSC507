
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import math 
from pathlib import Path
import os
from mpl_toolkits.mplot3d import Axes3D

def gnomonic(radius, c, fig_dir, type='equidistant'):
    
    run_date = dt.datetime.now().strftime('%y%m%d')

    if type == 'equidistant':
        a = (np.sqrt(3)) / 3 *radius
        # set up x y and z values
        x = np.linspace(-a,a,c)
        y = np.linspace(-a,a,c)

        # create the points along the edges
        xlocal, ylocal = np.meshgrid(x, y)


    elif type == 'equiangular':
        a = np.pi / 4
        x = np.linspace(-a,a,c)
        y = np.linspace(-a,a,c)
        xlocal_angle = a* np.tan(x)
        ylocal_angle = a* np.tan(y)

        # create the points along the edges
        xlocal, ylocal = np.meshgrid(xlocal_angle, ylocal_angle)



    bigZ_top = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * a
    bigZ_bottom = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -a

    bigx_top = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * xlocal
    bigx_bottom = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -xlocal

    bigy_top = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * ylocal
    bigy_bottom = ( (radius) / np.sqrt( a**2 +  xlocal**2 +ylocal**2) ) * -ylocal

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(xv, yv, z)
    ax.plot_wireframe(bigx_top, bigy_top, bigZ_top, color = 'b')
    ax.plot_wireframe(bigx_bottom, bigy_bottom, bigZ_bottom, color = 'b')
    ax.plot_wireframe(bigZ_top, bigx_top, bigy_top, color = 'r')
    ax.plot_wireframe(bigZ_bottom, bigx_bottom, bigy_bottom, color = 'r')
    ax.plot_wireframe(bigx_top, bigZ_top, bigy_top, color = 'g')
    ax.plot_wireframe(bigx_bottom, bigZ_bottom, bigy_bottom, color = 'g')
    plt.title(type+' gnomonic projection')
    plt.savefig(fig_dir+'test_'+type+'_gnomonic'+'_'+run_date+'.png')
