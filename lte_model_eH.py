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
infile=directory+'test.hdf5'
info=xarray.open_dataset(infile)
zinput=info.z[0,0,0,:].data*1.e-3
tinput=info.temperature[0,0,0,:].data
vtinput=info.velocity_turbulent[0,0,0,:].data*1.e-3
nHinput=info.hydrogen_populations[0,:,0,0,:].data*1.e-6
nHtinput=nHinput.sum(0)
neinput=info.electron_density[0,0,0,:].data*1.e-6


os.chdir('/data/home/chae/rhex/rh/rh15d/run/output') 
import rh15d
rr=rh15d.Rh15dout()
zout=rr.atmos.height_scale[0,0,:]*1.e-3
tout=rr.atmos.temperature[0,0,:]
nHout=rr.atom_H.populations[:,0,0,:]*1.e-6
nHLout=rr.atom_H.populations_LTE[:,0,0,:]*1.e-6
nHtout=nHout.sum(0)
nHLtout=nHLout.sum(0)
xHout=nHout[5,:]/nHtout

tinterp=interp1d(zinput*1.e5,tinput, fill_value="extrapolate")
vtinterp=interp1d(zinput*1.e5,vtinput,fill_value="extrapolate")
logxHinterp=interp1d(zout*1.e5, np.log(xHout),fill_value="extrapolate")
lognHtinterp=interp1d(zinput*1.e5, np.log(nHtinput),fill_value="extrapolate")

t0=tinterp(0.)
vt0=vtinterp(0)
xH0=np.exp(logxHinterp(0))
nHt0=np.exp(lognHtinterp(0))
mu0=(1+4*AHe)/(1+xH0)
rho0=nHt0*(1+4*AHe)*mH
Ptot0=rho0*((kb*t0)/(mu0*mH)+0.5*(vt0*1.e5)**2)


def Modelinterp(z0):
    t0=tinterp(z0)
    vt0=vtinterp(z0)
    xH0=np.exp(logxHinterp(z0))  
    return t0, vt0, xH0
def nel_LTE(Ptot, T, Vturb, xH):
    eps=1.e-5
    
    uT=0.
    for i in np.arange(1,8,1):
        tmp=5040/T*13.6*(1.-1./(i*i))
        uT=uT+2*(i*i)*np.exp(-tmp)
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
        xH1    = 0.5*(xH+PhiH1/(r+PhiH1)) 
        xAl   = PhiAl1/(r+PhiAl1)
        xNa   = PhiNa1/(r+PhiNa1)
        xK    = PhiK1/(r+PhiK1)
        xCa   = PhiCa1/(r+PhiCa1)
        rr   = 0.5*(rr+np.log10(AK*xK+AAl*xAl+ANa*xNa+ACa*xCa+xH))
        
    nel   = nHtot*r    
    Pg    =  rho* kb*T/(mu*mH)  
    return nel, nHtot,  rho, Pg

def HSE(lnPtot, z):
    Ptot = np.exp(lnPtot)
    T, Vturb, xH = Modelinterp(z)
    nel, nHt, rho, Pg = nel_LTE(Ptot, T, Vturb, xH)
    dlnPtotdz = -rho*g0/Ptot
    return dlnPtotdz
atmos = tout < 2.e4
ztmp=zout[atmos]
z1=np.linspace(0,ztmp.max(),150)
z2=np.linspace(ztmp.min(),-5., 20)
Ptot1=np.exp(odeint(HSE, np.log(Ptot0),z1*1.e5))
Ptot1=Ptot1.reshape(len(z1))
z2=z2[::-1]
Ptot2=np.exp(odeint(HSE, np.log(Ptot0),z2*1.e5))
Ptot2=Ptot2.reshape(len(z2))
z=np.concatenate((z2[::-1], z1))
Ptot=np.concatenate((Ptot2[::-1],Ptot1))

z=z[::-1]
Ptot=Ptot[::-1]
#plt.plot(zarray/1.e5, np.log10(Ptotarray))
T, vturb, xH=Modelinterp(z*1.e5)   
ne, nHt, rho, Pg=nel_LTE(Ptot, T, vturb, xH) #, xH=xH, nHtot=nHtot, mu=mu ) 
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
    nH[0,5,0,0,:] = nedata[::-1]*0
    vz=z*0
else:
    nH=np.zeros(shape=(1,6,1,1,T.shape[-1]), dtype='float')
    nH[0,0,0,0,:] = nHt
    
    

from helita.sim import rh15d

z=z[None,None,None,:]
T=T[None,None,None,:]
vz=vz[None,None,None,:]
ne=ne[None,None,None,:]
vturb=vturb[None,None,None,:]
z=z*1.e3 #*u.m
T=T #*u.K
vz=vz #*u.m*u.s**(-1)
ne=ne*1.e6  #*u.m**(-3)
vturb=vturb*1.e3  #*u.m*u.s**(-1)
nH=nH*1.e6 # *u.m**(-3)
#rho=rho[None,None,None,:]
#rho=rho*1.e3 #(rho*u.g*u.cm**(-3)).to(u.kg*u.m**(-3))

#z=(z*u.km).to(u.m)
#T=T*u.K
#vz=vz*u.m*u.s**(-1)
#ne=(ne*u.cm**(-3)).to(u.m**(-3))
#vturb=(vturb*u.km*u.s**(-1)).to(u.m*u.s**(-1))
#nH=(nH*u.cm**(-3)).to(u.m**(-3))
#rho=rho[None,None,None,:]
#rho=(rho*u.g*u.cm**(-3)).to(u.kg*u.m**(-3))

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
ax1.set_ylim(8, 16)
#ax1.set_xlabel('z (kim)')
ax1.set_ylabel('log ne')
p1 = ax1.plot(z[0,0,0,:]/1.e3,np.log10(ne[0,0,0,:]/1.e6))
ax2.set_xlim(-500, 2500)
ax2.set_ylim(0, 20)
ax2.set_xlabel('z (kim)')
ax2.set_ylabel('log Nh0')
p2 = ax2.plot(z[0,0,0,:]/1.e3,np.log10(nH[0,0,0,0,:]/1.e6))
ax3.set_xlim(-500, 2500)
ax3.set_xlabel('z (kim)')
ax3.set_ylabel('vturb')
p3 = ax3.plot(z[0,0,0,:]/1.e3, vturb[0,0,0,:]/1.e3)

