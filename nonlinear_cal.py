# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helita.sim import rh15d
import os
import xarray
from fisspy.read import FISS
from mylib import misc
from mylib.sub import ReadFAL, Radtemp, Turbspeed, writerayinput, IntensityIntegral
from mylib.sub import TauIntegral
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import simpson
import time
import pickle

#%%
def f(x, S):
    return x-np.sin(S*x + theta1)

S = 0.5
phi = 0
P = 2.7
omega = 2*np.pi/(P*60)
t = 0
theta = omega*t - phi
theta1 = theta // 2*np.pi
v1 = 3.86

#%%
if 0 :
    Afile='/data/home/chokh2/rh/FISS/FISS_20170614_165710_A1_c.fts'
    Bfile='/data/home/chokh2/rh/FISS/FISS_20170614_165710_B1_c.fts'
    fissA=FISS(Afile)
    fissB=FISS(Bfile)
    Aprof=fissA.refProfile/fissA.refProfile.max()
    zeta=0.055 #0.065
    Aprof = (Aprof -zeta*Aprof.max())/(1.-zeta) 
    wvA = fissA.wave*0.1
    Bprof=fissB.refProfile/fissB.refProfile.max()
    Bprof = (Bprof -zeta*Bprof.max())/(1.-zeta) 
    wvB = fissB.wave*0.1

directory=r'/data/home/chokh2/rh/Atmos'
f=open(directory+r'/Maltby86E.tab', 'r')
data=(f.read()).split('\n')
f.close()
Ndep=int(data[2])
data1=data[5:Ndep+5]
z1=np.arange(Ndep, dtype='float')
T1=np.arange(Ndep, dtype='float')
nH1=np.arange(Ndep, dtype='float')
ne1=np.arange(Ndep, dtype='float')
vturb1=np.arange(Ndep, dtype='float')
# vz1=np.zeros(Ndep, dtype='float')
for i in range(Ndep):
    z1[i],T1[i],nH1[i], ne1[i], vturb1[i]=data1[i].split()      

n=301
delta=(z1[0]-z1.min())/(n-1)
z=z1[0]-np.arange(n)*delta
cs = CubicSpline(-z1,np.log10(T1) )
T=10.**cs(-z)
cs = CubicSpline(-z1, np.log10(ne1))
ne=10.**cs(-z)
cs = CubicSpline(-z1, np.log10(nH1))
nH=10.**cs(-z)
cs = CubicSpline(-z1, np.log10(vturb1+1e-10))
vturb=10.**cs(-z)

vz=z*0.0

outfile='/data/home/chokh2/rh/Atmos/model.hdf5'

s1 = abs(T-T.min()).argmin()
z_tm = z[s1]
nz=len(z)
zmax=z.max()
zmin=z.min()
nHp=np.zeros(shape=(1,2,1,1,nz), dtype='float')
nHp[0,0,0,0,:]=nH*1.e6
z=z.reshape(1, 1, 1,nz)*1.e3 
T=T.reshape(1, 1, 1, nz) 
vz=vz.reshape(1, 1, 1, nz) 
ne=ne.reshape(1, 1, 1, nz)*1.e6 
vturb=vturb.reshape(1, 1, 1, nz)*1.e3 

rh15d.make_xarray_atmos(outfile, z=z,T=T, nH=nHp, ne=ne,
                        vturb=vturb, vz=vz) 

wv0 = [1336, 1394, 2796, 5434, 5890, 6563, 8542]
wvarr = []
for i in range(len(wv0)):
    wvarr1 = np.arange(wv0[i]-0.3, wv0[i]+0.3, 0.02)
    wvarr = np.concatenate((wvarr, wvarr1))

rh15d.make_wave_file('/data/home/chokh2/rh/Atoms/wave_files/tmp.wave', new_wave=wvarr)

#%%
os.chdir(r'/data/home/chokh2/rh/rh15d/run/')
if 1 :
    os.system('rm output/*.hdf5')
    t_start = time.time()
    os.system('mpirun -np 2 rh15d_ray_pool')   
    t_end = time.time()
    os.chdir('output') 
    rr = rh15d.Rh15dout()
    ray = xarray.open_dataset(r'output_ray.hdf5')
    print(f'Calculation time = {t_end-t_start} s')
    result = {}
    result['z'] = rr.atmos.height_scale[0,0,:].data*1.e-3  # in km
    result['temp'] = rr.atmos.temperature[0,0,:].data
    result['inten'] = ray.intensity[0,0,ray.wavelength_indices].values
    result['chi'] = ray.chi[0,0,:,:].data
    result['sf'] = ray.source_function[0,0,:,:].data
    result['wv'] = ray.wavelength_selected.data
    result['tau_one_height'] = ray.tau_one_height[0,0,ray.wavelength_indices].data
    tau = result['chi'] * 0
    z_mid = np.zeros(len(result['z'])+1)
    z_mid[1:-1] = 0.5*(result['z'][1:] + result['z'][:-1])
    z_mid[0] = 2.*result['z'][0] - z_mid[1]
    z_mid[-1] = 2.*result['z'][-1] - z_mid[-2] 
    for j in range(len(result['z'])):
        dtau = result['chi'][j]*(z_mid[j]-z_mid[j+1])*1e3
        tau[j] = tau[j-1] + dtau if j != 0 else dtau
    result['tau'] = tau
    with open('output.p', 'wb') as file:
        pickle.dump(result, file)

#%%
os.chdir(r'/data/home/chokh2/rh/rh15d/run/output')
with open('output.p', 'rb') as file:
    res = pickle.load(file)
    
fig00 = plt.figure(1)
gs = GridSpec(2,2,fig00)
ax00 = fig00.add_subplot(gs[0,0])
ax01 = fig00.add_subplot(gs[0,1])
ax02 = fig00.add_subplot(gs[1,0])
ax03 = fig00.add_subplot(gs[1,1])

ax00.set_xlim(zmin, zmax)
ax00.set_ylim(3000, 14000)
ax00.set_ylabel('T (K)')
p00 = ax00.plot(z[0,0,0,:]/1.e3,T[0,0,0,:])
p001 = ax00.plot([z_tm, z_tm], ax00.get_ylim(), '--', color='gray')
ax01.set_xlim(zmin, zmax)
ax01.set_ylim(6, 16)
ax01.set_ylabel('log $n_e$')
p01 = ax01.plot(z[0,0,0,:]/1.e3,np.log10(ne[0,0,0,:]/1.E6))
p011 = ax01.plot([z_tm, z_tm], ax01.get_ylim(), '--', color='gray')
ax02.set_xlim(zmin,zmax)
ax02.set_ylim(10, 18)
ax02.set_xlabel('z (km)')
ax02.set_ylabel('log N$_H$')
p02 = ax02.plot(z[0,0,0,:]/1.e3,np.log10(nHp[0,0,0,0,:]/1.E6))
p021 = ax02.plot([z_tm, z_tm], ax02.get_ylim(), '--', color='gray')
ax03.sharex(ax00)
ax03.set_xlabel('z (km)')
ax03.set_ylabel('$v_{turb}$ (km/s)')
p03 = ax03.plot(z[0,0,0,:]/1e3, vturb[0,0,0,:]/1.e3)
p031 = ax03.plot([z_tm, z_tm], ax03.get_ylim(), '--', color='gray')
fig00.tight_layout()
fig00.savefig('fig00.png', dpi=300)    