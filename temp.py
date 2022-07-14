# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import rh15d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from helita.sim import rh15d_vis
#import os
#os.system('cd /data/home/chae/rhex/rh/rh15d/run')
#os.system('mpirun -np 2 rh15d_ray_pool')
#os.system('cd output')
rh15d_vis.InputAtmosphere('/data/home/chae/rhex/rh/Atmos/FALC_82_5x5.hdf5')
fig = plt.figure()
gs = GridSpec(2,2,fig)
ax0 = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[1,1])

rr=rh15d.Rh15dout()
z=rr.atmos.height_scale[0,0,:]*1.e-3
temp=rr.atmos.temperature[0,0,:]
ax0.set_ylim(0,15000)
ax0.set_xlim(-500,2500)
p0 = ax0.plot(z,temp, color='r')
wvnm=rr.ray.wavelength
inten=rr.ray.intensity[0,0,:]
ax1.set_ylim(0,1.*inten.max())
ax2.set_ylim(0,1.*inten.max())
ax1.set_xlim(854.2-0.3, 854.2+0.3)
ax2.set_xlim(656.3-0.3, 656.3+0.3)
p1 = ax1.plot(wvnm,inten,  color='r')
p2 = ax2.plot(wvnm,inten,  color='b')