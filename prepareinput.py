# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.constants as const
from astropy import units as u
from scipy import integrate 
from helita.sim import rh15d
#from helita.sim import rh15d_vis
directory='/data/home/chae/rh/Atmos/'
file='FALF_80'
f=open(directory+file+'.atmos', 'r')
data=(f.read()).split('\n')
f.close()
Ndep=data[9].split()
Ndep=int(Ndep[0])
data1=data[12:Ndep+12]
logm=np.arange(Ndep, dtype=float)
T=np.arange(Ndep, dtype=float)
ne=np.arange(Ndep, dtype=float)
vz=np.arange(Ndep, dtype=float)
vturb=np.arange(Ndep, dtype=float)
data2=data[Ndep+12+3:Ndep+12+3+Ndep]
nH1=np.arange(Ndep, dtype=float)
nH2=np.arange(Ndep, dtype=float)
nH3=np.arange(Ndep, dtype=float)
nH4=np.arange(Ndep, dtype=float)
nH5=np.arange(Ndep, dtype=float)
nHp=np.arange(Ndep, dtype=float)
for i in range(Ndep):
    logm[i],T[i],ne[i],vz[i],vturb[i]=data1[i].split()
    nH1[i],nH2[i],nH3[i],nH4[i],nH5[i],nHp[i]=data2[i].split()

nHt=nH1+nH2+nH3+nH4+nH5+nHp
rho=nHt*1.4*(const.m_p.value*1.e3)
z = integrate.cumtrapz(-10**logm/rho*np.log(10.), logm, initial=0.)+2.3e8
#plt.plot(z, np.log10(T))
plt.plot(z/1.e5, np.log10(T))
nH=np.zeros(shape=(1,6,1,1,T.shape[-1]), dtype='float')
nH[0,0,0,0,:]=nH1
nH[0,1,0,0,:]=nH2
nH[0,2,0,0,:]=nH3
nH[0,3,0,0,:]=nH4
nH[0,4,0,0,:]=nH5
nH[0,5,0,0,:]=nHp


z=z[None,None,None,:]
T=T.reshape(1,1,1,T.shape[-1]) #T[None,None,None,:]
vz=vz[None,None,None,:]
ne=ne[None,None,None,:]
vturb=vturb[None,None,None,:]
z=(z*u.cm).to(u.m)
T=T*u.K
vz=vz*u.m*u.s**(-1)
ne=(ne*u.cm**(-3)).to(u.m**(-3))
vturb=(vturb*u.km*u.s**(-1)).to(u.m*u.s**(-1))
nH=(nH*u.cm**(-3)).to(u.m**(-3))
rho=rho[None,None,None,:]
rho=(rho*u.g*u.cm**(-3)).to(u.kg*u.m**(-3))
outfile=directory+'test.hdf5'
#from helita.sim import rhd
#
#rh15d.make_xarray_atmos(outfile = outfile, T = T, vz = vz, z = z,                       nH = nh, ne = elec_den, vturb = vturb)
rh15d.make_xarray_atmos(outfile=outfile,T=T,vz=vz,z=z,nH=nH,ne=ne,vturb=vturb) 
#rh15d.make_xarray_atmos(outfile,T,vz,z, rho=rho, ne=ne,vturb=vturb)  
#atmos1 = xarray.open_dataset('/data/home/chae/rhex/rh/Atmos/FALC_82_5x5.hdf5')
#rh15d_vis.InputAtmosphere(outfile)
    
