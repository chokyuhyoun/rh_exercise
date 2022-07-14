# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from helita.sim import rh15d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from astropy import units
from astropy import constants
import astropy.modeling.blackbody as bb
from fisspy.read import FISS
import xdrlib
def Radtemp(intensity, wvnm):
    wv=(wvnm.values*units.nm).to(units.m)
    nu=constants.c/wv
    hnu=nu*constants.h
    T=(hnu/constants.k_B).value/np.log(1+(2*hnu*nu**2/constants.c**2).value/intensity)
    return T


region='QS'

if region == 'QS':
    Afile='/data/home/chae/FISS/FISS_20170614_165710_A1_c.fts'
    Bfile='/data/home/chae/FISS/FISS_20170614_165710_B1_c.fts'
    fissA=FISS(Afile)
    fissB=FISS(Bfile)
#Aprof=fissA.data[int(fissA.nx/2)-10, int(fissB.ny/2)-20, :]
#Bprof=fissB.data[int(fissB.nx/2)-10, int(fissB.ny/2)-20, :]
    Aprof=fissA.data.mean((0,1))
    Bprof=fissB.data.mean((0,1))

if region == 'sunspot':
    if 1<0 :
        Afile='/data/home/chae/FISS/FISS_20150615_171619_A1_c.fts'
        Bfile='/data/home/chae/FISS/FISS_20150615_171620_B1_c.fts'
    else :    
        Afile='/data/home/chae/FISS/FISS_20180620_162619_A1_c.fts'
        Bfile='/data/home/chae/FISS/FISS_20180620_162620_B1_c.fts'
    
    fissA=FISS(Afile)
    fissB=FISS(Bfile)
    contA=fissA.data[:,:,0:50].mean(2)
    contB=fissB.data[:,:,0:50].mean(2)
    Aprof=fissA.data[np.logical_and(contA < 0.5*contA.max(), contA > 0.02*contA.max()), :].mean(0)
    Bprof=fissB.data[np.logical_and(contB < 0.5*contB.max(), contB > 0.02*contB.max()), :].mean(0)
    Aprof0=fissA.data.mean((0,1))
    #fissA.data[np.logical_and(contA < 1.0*contA.max(), contA > 0.8*contA.max()), :].mean(0)
    Bprof0=fissB.data.mean((0,1))
    #[np.logical_and(contB < 1.0*contB.max(), contB > 0.8*contB.max()), :].mean(0)
    eps=0.15
    Aprof=(Aprof-eps*Aprof0)/(1-eps)
    Bprof=(Bprof-eps*Bprof0)/(1-eps)

#rh15d.make_wave_file('/data/home/chae/rh/Atoms/wave_files/tmp.wave', 542.95, 544.96, 0.002)
#new_wave=np.arange(543.3, 534.8, 0.01)  
#p = xdrlib.Packer()
#nw = len(new_wave)
#p.pack_int(nw)
#p.pack_farray(nw, new_wave.astype('d'), p.pack_double)
#f=open('/data/home/chae/rh/Atoms/wave_files/chae.wave', 'w')  
#f.write(nw)
#f.write(new_wave.astype('d'))
#f.close()
#stop
os.chdir('/data/home/chae/rh/rh15d/run/')
os.system('rm output/*.hdf5')
os.system('mpirun -np 2 rh15d_ray_pool')   
os.chdir('output') 
fig = plt.figure(figsize=[8., 10])
gs = GridSpec(4,2,fig)
ax0 = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[2,1])
ax5 = fig.add_subplot(gs[3,0])
ax6 = fig.add_subplot(gs[3,1])
rr=rh15d.Rh15dout()
z=rr.atmos.height_scale[0,0,:]*1.e-3
temp=rr.atmos.temperature[0,0,:]
ax0.set_ylim(2000,8000)
ax0.set_xlim(-500,2500)
p0 = ax0.plot(z,temp)
wvnm=rr.ray.wavelength[()]
inten=rr.ray.intensity[0,0,:]
tau_one_height=rr.ray.tau_one_height[0,0,:]
sourceha=rr.ray.source_function[0,0,:,0]
chiha=rr.ray.chi[0,0,:,0]
Tha=Radtemp(sourceha,rr.ray.wavelength_selected[0])
sourceca=rr.ray.source_function[0,0,:,1]
Tca=Radtemp(sourceca,rr.ray.wavelength_selected[1])
sourcecaK=rr.ray.source_function[0,0,:,2]
TcaK=Radtemp(sourcecaK,rr.ray.wavelength_selected[2])

ax0.plot(z, Tca, color='r')
ax0.plot(z, Tha, color='b')
ax0.plot(z, TcaK, color='g')
wvca=wvnm-854.21
ca=(wvca-1.)*(wvca+1.) <= 0
caK=(wvnm-393.35-2)*(wvnm-393.35+2) <= 0
wvha=wvnm-656.285
ha=(wvha-2.)*(wvha+2.) <= 0
Lya=(wvnm-121.6-0.15)*(wvnm-121.6+0.15) <= 0

#wvna=wvnm-589.0
#Na=(wvna-1)*(wvna+1) <= 0
#wvfe=wvnm-6302.1
Fe=(wvnm-543.45-2)*(wvnm-543.45+2) <= 0

ax1.set_ylim(0.1,1.2)
ax1.set_yscale('log')
ax2.set_ylim(-100,2000)
ax3.set_ylim(0.1,1.2)
ax3.set_yscale('log')
ax4.set_ylim(-100,2000)
ax1.set_xlim(-0.5+656.285, 0.5+656.285)
ax2.set_xlim(-0.5+656.285, 0.5+656.285)
ax3.set_xlim(-0.5+854.21, 0.5+854.21)
ax4.set_xlim(-0.5+854.21, 0.5+854.21)
ax5.set_xlim(-0.3+543.45, 0.3+543.45)
ax6.set_xlim(-0.3+543.45, 0.3+543.45)
ax5.set_ylim(0.1,1.2)
ax5.set_yscale('log')
ax6.set_ylim(-100,1000)


p1 = ax1.plot(wvnm[ha],(inten[ha]/(inten[ha].max())),  color='b')
#p1a= ax1.plot(fissA.wave*0.1, (Aprof/Aprof.max()-0.065)/0.935*0.96)
p2 = ax2.plot(wvnm[ha],tau_one_height[ha]/1.E3,  color='b')
p3 = ax3.plot(wvnm[ca], (inten[ca]/(inten[ca].max())),  color='r')
#p3a= ax3.plot(fissB.wave*0.1, (Bprof/Bprof.max()-0.065)/0.935*0.97)

p4 = ax4.plot(wvnm[ca],tau_one_height[ca]/1.E3,  color='r')
p5 = ax5.plot(wvnm[Fe], inten[Fe]/inten[Fe].max(),  color='g')
p6 = ax6.plot(wvnm[Fe],tau_one_height[Fe]/1.E3,  color='g')

