# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 18:10:39 2019

@author: 채종철
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import astropy.constants as const
from astropy import units as u
from scipy import integrate 
from scipy import signal
from scipy.interpolate import interp1d
from scipy.integrate import odeint
g0=27400. # cm/s^2
kb=1.38e-16  # cgsxsq
mH=1.67e-24  # cgs
AHe=0.1

directory='/data/home/chae/rhex/rh/Atmos/'
file='Maltby'
#file='Av2015sunspot'
f=open(directory+file+'.atmos', 'r')
data=(f.read()).split('\n')
f.close()
Ndep=int(data[2])
data1=data[5:Ndep+5]
seq=np.arange(Ndep, dtype='int')
depth=np.arange(Ndep, dtype='float')
mc=np.arange(Ndep, dtype='float')
T=np.arange(Ndep, dtype='float')
vturb=np.arange(Ndep, dtype='float')
Pg=np.arange(Ndep, dtype='float')
Pt=np.arange(Ndep, dtype='float')
Nt=np.arange(Ndep, dtype='float')
Nn=np.arange(Ndep, dtype='float')
ne=np.arange(Ndep, dtype='float')
vz=np.arange(Ndep, dtype='float')
vz=vz*0
rho=np.arange(Ndep, dtype='float')

for i in range(Ndep):
    seq[i],depth[i],mc[i],T[i],vturb[i],Pg[i],Pt[i],Nt[i],Nn[i],ne[i],rho[i]=data1[i].split()    
zdata = (-depth)[::-1]
tdata=T[::-1]
vtdata=vturb[::-1]
if file =='Maltby':
    vtdata=vturb[::-1]*1.5+0.3

Pt=Pt[::-1]
nedata=ne[::-1]
rhodata=rho[::-1]
Nndata=Nn[::-1]
tinterp=interp1d(zdata*1.e5,tdata)
vtinterp=interp1d(zdata*1.e5,vtdata)
logPtinterp=interp1d(zdata*1.e5,np.log(Pt))
Ptot0=np.exp(logPtinterp(0.))
if file =='Maltby':
    Ptot0=Ptot0


def Modelinterp(z0):
    t0=tinterp(z0)
    vt0=vtinterp(z0)
    return t0, vt0
def nel_LTE(Ptot, T, Vturb):
    eps=1.e-5
    
    uT=0.
    for i in np.arange(1,8,1):
        tmp=5040/T*13.6*(1.-1./(i*i))
        uT=uT+2*(i*i)*np.exp(-tmp)
    xH     = Ptot*0.+0.
    Theta  = 5040./T
    logUAl = 0.77*(Theta-0.5)/0.5+0.81*(1-Theta)/0.5
    logUNa = 0.31*(Theta-0.5)/0.5+0.60*(1-Theta)/0.5
    logUNa = 0.31*(Theta-0.5)/0.5+0.60*(1-Theta)/0.5
    logUK  = 0.34*(Theta-0.5)/0.5+0.60*(1-Theta)/0.5
    logUCa = 0.07*(Theta-0.5)/0.5+0.55*(1-Theta)/0.5
    
    PhiH  = 10.**(-13.6*Theta+2.5*np.log10(T)-np.log10(uT)-0.1762)
    PhiAl = 10.**(-5.985*Theta+2.5*np.log10(T)-logUAl-0.1762)
    PhiNa = 10.**(-5.13*Theta+2.5*np.log10(T)-logUNa-0.1762)
    PhiK  = 10.**(-4.34*Theta+2.5*np.log10(T)-logUK-0.1762)
    PhiCa = 10.**(-6.11*Theta+2.5*np.log10(T)-logUCa-0.1762)
    AK  = 1.31E-7
    AAl = 2.34E-6
    ANa = 2.13E-6
    ACa = 2.04E-6
    rr=-3.
    for rep in np.arange(10):
        r=10**rr
        mu    = (1+4.*AHe)/(1+r)
        rho   = Ptot/(kb*T/(mu*mH)+0.5*(Vturb*1.e5)**2)
        nHtot = rho/((1+4.*AHe)*mH)      
        PhiH1  = PhiH/(nHtot*kb*T)
        PhiAl1 = PhiAl/(nHtot*kb*T)
        PhiNa1 = PhiNa/(nHtot*kb*T)
        PhiK1  = PhiK/(nHtot*kb*T)
        PhiCa1 = PhiCa/(nHtot*kb*T)          
        xH    = PhiH1/(r+PhiH1) 
        xAl   = PhiAl1/(r+PhiAl1)
        xNa   = PhiNa1/(r+PhiNa1)
        xK    = PhiK1/(r+PhiK1)
        xCa   = PhiCa1/(r+PhiCa1)
        rr   = 0.5*(rr+np.log10(AK*xK+AAl*xAl+ANa*xNa+ACa*xCa+xH))
        
    nel   = nHtot*r    
    Pg    =  rho* kb*T/(mu*mH)  
    nHn   =  nHtot*(1-xH)
    nH    =  np.reshape(np.zeros(6*len(nHn)), (6, len(nHn)))
    nH[5,:] = nHtot*xH
    for i in np.arange(1,6,1):
        tmp  =  5040/T*13.6*(1.-1./(i*i))
        nH[i-1,:] = 2*(i*i)/uT*np.exp(-tmp)*nHn
    return nel, xH, nH,  rho, Pg
def HSE(lnPtot, z):
    Ptot = np.exp(lnPtot)
    T, Vturb = Modelinterp(z)
    nel, xH, nH, rho, Pg = nel_LTE(Ptot, T, Vturb)
    dlnPtotdz = -rho*g0/Ptot
    return dlnPtotdz
#Ptot0 = Pt[6]#np.array([5.436E3])
#T0 = np.array([3517.])
#V0 = np.array([0.48])
#nel, xH, nH, rho, Pg = nel_LTE(Ptot0, T0,V0)
#print(nel)
#z1=zdata[6:-2] 
atmos=tdata < 2.e4
ztmp=zdata[atmos]
Ntmp=len(ztmp)    
print(0.5*(ztmp[0]+ztmp[1]), ztmp[Ntmp-1] )
z1=np.linspace(0,ztmp[Ntmp-2],150)
z2=np.linspace(0.5*(ztmp[0]+ztmp[1]),-5., 20)
Ptot1=np.exp(odeint(HSE, np.log(Ptot0),z1*1.e5))
Ptot1=Ptot1.reshape(len(z1))
z2=z2[::-1]
Ptot2=np.exp(odeint(HSE, np.log(Ptot0),z2*1.e5))
Ptot2=Ptot2.reshape(len(z2))
z=np.concatenate((z2[::-1], z1))
Ptot=np.concatenate((Ptot2[::-1],Ptot1))

z=z[::-1]
Ptot=Ptot[::-1]
T, vturb=Modelinterp(z*1.e5)   
ne, xH, nH1, rho, Pg=nel_LTE(Ptot, T, vturb) #, xH=xH, nHtot=nHtot, mu=mu ) 
vz=z*0

if 1<0 :
    factor=1.
    T=tdata[::-1]
    z=zdata[::-1]
    ne=nedata[::-1]*factor
    rho=rhodata[::-1]*factor
    vturb=vtdata[::-1]
    nH=np.zeros(shape=(1,6,1,1,T.shape[-1]), dtype='float')  
    nH[0,0,0,0,:] = Nndata[::-1]*factor
    nH[0,5,0,0,:] = nedata[::-1]
    vz=z*0
else:
    nH=np.zeros(shape=(1,6,1,1,T.shape[-1]), dtype='float')
    nH[0,0,0,0,:]=nH1.sum(0)
 #   for i in np.arange(6):
 #       nH[0,i,0,0,:] = nH1[i,:]
    
    
    

from helita.sim import rh15d

z=z[None,None,None,:]
T=T[None,None,None,:]
vz=vz[None,None,None,:]
ne=ne[None,None,None,:]
vturb=vturb[None,None,None,:]
z=z*1.e3 #(z*u.km).to(u.m)
T=T #*u.K
vz=vz # *u.m*u.s**(-1)
ne=ne*1.e6 # (ne*u.cm**(-3)).to(u.m**(-3))
vturb=vturb*1.e3 #(vturb*u.km*u.s**(-1)).to(u.m*u.s**(-1))
nH=nH*1.e6 #(nH*u.cm**(-3)).to(u.m**(-3))
directory='/data/home/chae/rhex/rh/Atmos/'
outfile=directory+'test.hdf5'
rh15d.make_xarray_atmos(outfile,T,vz,z,nH=nH,ne=ne,vturb=vturb) 

fig = plt.figure()
gs = GridSpec(2,2,fig)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

ax0.set_xlim(-500, 2500)
ax0.set_ylim(3000, 15000)
#ax0.set_xlabel('z (kim)')
ax0.set_ylabel('T')
p0 = ax0.plot(z[0,0,0,:]/1.e3,T[0,0,0,:])
ax1.set_xlim(-500, 2500)
ax1.set_ylim(6, 16)
#ax1.set_xlabel('z (kim)')
ax1.set_ylabel('log ne')
p1 = ax1.plot(z[0,0,0,:]/1.e3,np.log10(ne[0,0,0,:]/1.E6))
ax2.set_xlim(-500, 2500)
ax2.set_ylim(-5, 5)
ax2.set_xlabel('z (kim)')
ax2.set_ylabel('log Pt')
p2 = ax2.plot(z[0,0,0,:]/1.e3,np.log10(Ptot/1.E1))
ax3.set_xlim(-500, 2500)
ax3.set_xlabel('z (kim)')
ax3.set_ylabel('vturb')
p3 = ax3.plot(z[0,0,0,:]/1.e3, vturb[0,0,0,:]/1.e3)

