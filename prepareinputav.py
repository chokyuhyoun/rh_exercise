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
#import numpy 
##from helita.sim import rh15d_vis
#import matplotlib.pyplot as plt
#from matplotlib.gridspec import GridSpec
##from helita.sim import rh15d_vis
#
import os
directory='/data/home/chae/rh/Atmos/'
model='M'
file='Maltby86'+model
#file='Av2015qs'
file='FALC'
#file='Av2015sunspot'

f=open(directory+'FALC.tab', 'r')
data=(f.read()).split('\n')
f.close()
Ndep=int(data[2])
data1=data[5:Ndep+5]
zQ=np.arange(Ndep, dtype='float')
TQ=np.arange(Ndep, dtype='float')
vturbQ=np.arange(Ndep, dtype='float')
nHQ=np.arange(Ndep, dtype='float')
neQ=np.arange(Ndep, dtype='float')
vzQ=np.zeros(Ndep, dtype='float')
for i in range(Ndep):
    zQ[i],TQ[i],nHQ[i], neQ[i],vturbQ[i]=data1[i].split()    

f=open(directory+'Maltby86M.tab', 'r')
data=(f.read()).split('\n')
f.close()
Ndep=int(data[2])
data1=data[5:Ndep+5]
zM=np.arange(Ndep, dtype='float')
TM=np.arange(Ndep, dtype='float')
vturbM=np.arange(Ndep, dtype='float')
nHM=np.arange(Ndep, dtype='float')
neM=np.arange(Ndep, dtype='float')
vzM=np.zeros(Ndep, dtype='float')
for i in range(Ndep):
    zM[i],TM[i],nHM[i], neM[i],vturbM[i]=data1[i].split()    
#
alpha=0.

z=zM*alpha+zQ*(1-alpha)
T=np.exp(np.log(TM)*alpha+np.log(TQ)*(1-alpha))
nH=np.exp(np.log(nHM)*alpha+np.log(nHQ)*(1-alpha))
ne=np.exp(np.log(neM)*alpha+np.log(neQ)*(1-alpha))
vturb=vturbM*alpha+vturbQ*(1-alpha)
vz=vzM*alpha+vzQ*(1-alpha)

#T=T*0+0.6e4
#nH=nH*0+5.e13
#ne=ne*0+1.e9
#vturb=vturb*0+10.
#vz=vz*0+0.

directory='/data/home/chae/rh/Atmos/'
outfile=directory+'test.hdf5'
nHp=np.zeros(shape=(1,2,1,1,T.shape[-1]), dtype='float')
nHp[0,0,0,0,:]=nH*1.e6
z=z.reshape(1, 1, 1, z.shape[-1])*1.e3 
T=T.reshape(1, 1, 1, T.shape[-1]) 
vz=vz.reshape(1, 1, 1, vz.shape[-1]) 
ne=ne.reshape(1, 1, 1, ne.shape[-1])*1.e6 
vturb=vturb.reshape(1, 1, 1, vturb.shape[-1])*1.e3 
rh15d.make_xarray_atmos(outfile,z=z,T=T, nH=nHp,ne=ne,vturb=vturb,vz=vz) 
for i in range(10):
    plt.close()

fig = plt.figure(1)
gs = GridSpec(2,2,fig)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

ax0.set_xlim(-500, 2500)
ax0.set_ylim(3000, 14000)
#ax0.set_xlabel('z (kim)')
ax0.set_ylabel('T')
p0 = ax0.plot(z[0,0,0,:]/1.e3,T[0,0,0,:])
ax1.set_xlim(-500, 2500)
ax1.set_ylim(6, 16)
#ax1.set_xlabel('z (kim)')
ax1.set_ylabel('log ne')
p1 = ax1.plot(z[0,0,0,:]/1.e3,np.log10(ne[0,0,0,:]/1.E6))
ax2.set_xlim(-500, 2500)
ax2.set_ylim(10, 18)
ax2.set_xlabel('z (kim)')
ax2.set_ylabel('log NH')
p2 = ax2.plot(z[0,0,0,:]/1.e3,np.log10(nHp[0,0,0,0,:]/1.E6))
ax3.set_xlim(-500, 2500)
ax3.set_xlabel('z (kim)')
ax3.set_ylabel('vturb')
p3 = ax3.plot(z[0,0,0,:]/1.e3, vturb[0,0,0,:]/1.e3)



#z=(z*u.km).to(u.m)
#T=T*u.K
#vz=vz*u.m*u.s**(-1)
#ne=(ne*u.cm**(-3)).to(u.m**(-3))
#vturb=(vturb*u.km*u.s**(-1)).to(u.m*u.s**(-1))
#nH=(nH*u.cm**(-3)).to(u.m**(-3))
#rho=rho[None,None,None,:]
#rho=(rho*u.g*u.cm**(-3)).to(u.kg*u.m**(-3))
#outfile=directory+'test.hdf5'
#rh15d.make_xarray_atmos(outfile,T,vz,z,nH=nH,ne=ne,vturb=vturb) 
#os.system('cp '+directory+'/FALC_82_test.hdf5 '+directory+'/test.hdf5')
