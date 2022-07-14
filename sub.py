#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:18:11 2020

@author: chae
"""
import numpy as np
from astropy import units
from astropy import constants
from scipy.integrate import romb,simps, fixed_quad
from scipy.interpolate import CubicSpline
def ReadFAL(file):
    f=open(file, 'r')
    data=(f.read()).split('\n')
    f.close()
    Ndep=int(data[2][8:11])
    data1=data[4:Ndep+4]
    z=np.arange(Ndep, dtype='float')
    T=np.arange(Ndep, dtype='float')
    nH=np.arange(Ndep, dtype='float')
    ne=np.arange(Ndep, dtype='float')
    for i in range(Ndep):
        aa=data1[i].split()
        z[i]=float(aa[0])
        T[i]=float(aa[1])
        nH[i] = float(aa[3])+float(aa[4])+float(aa[5])
        ne[i] = float(aa[2])
    return z, T, ne, nH
def Radtemp(intensity, wvnm):
    wv=(wvnm.values*units.nm).to(units.m)
    nu=constants.c/wv
    hnu=nu*constants.h
    T=(hnu/constants.k_B).value/np.log(1+(2*hnu*nu**2/constants.c**2).value/intensity)
    return T
def Turbspeed(log10Nh):
#    coeffs=np.array([ 0.284, -0.0154,  0.640])
    coeffs=np.array([-0.0237111 ,  0.039041  , -0.63064979,  0.42720716])
    coeffb=np.array([ 1.02285984, -0.07933307,  0.50])
#    if len(log10Nh) == 1: log10Nh=np.array([log10Nh])
    log10Nh=np.atleast_1d(log10Nh)
    x = log10Nh - 16.
    y = x*0
    small = x <= 0
    if small.sum() > 0:
        y[small] = np.polyval(coeffs, x[small] )
    big = x >  0
    if big.sum() >0 :
        y[big] = np.polyval(coeffb, x[big])     
    if y.size == 1 :  y=y.item()
    return y

def writerayinput(file, mu, waveindex):
    f=open(file, 'wt')
    f.write(str(mu)+'\n')
    n=waveindex.size
    tmp=str(n)+'   '+str(waveindex[0])
    for i in range(1,n):
        tmp=tmp+' '+ str(waveindex[i])
    f.write(tmp+'\n')
    f.close()


def TauIntegral(z, chi):
    n=z.size
    cs = CubicSpline(-z, np.log(chi))
    tau = z*0.
    for i in range(n-1):
        n1=2**3+1
        delta = (z[i+1]-z[i])/(n1-1)
        z1=z[i]+np.arange(n1)*delta 
        chi1=np.exp(cs(-z1))
        tau[i+1]=tau[i]+romb(chi1, abs(delta))
    tau=tau+(tau[1]-tau[0])*0.5    
    return tau
def IntensityIntegrand(tau, cs):    
    return np.exp(cs(np.log(tau)))
 
def IntensityIntegral(tau,source, tau1):    

    integrand = np.log(tau*source)-(tau-tau1)    
    cs = CubicSpline(np.log(tau), integrand)
  
    
#    s= tau >= tau1
    xmin = np.log(tau1)
    xmax = np.log(tau.max())
    nnew = 2**12 +1
    delta=(xmax-xmin)/(nnew-1)
    xnew = np.arange(nnew)*delta+xmin

#    integral, err = fixed_quad(IntensityIntegrand, tau1, tau.max(), args=(cs))
    integrandnew = np.exp(cs(xnew))
    integral =  romb(integrandnew, delta )
#    integral =  simps(integrandnew, dx=delta )
    
    
#    for i in np.arange(top, n-1):
#        delta = np.log(tau[i+1])-np.log(tau[i]) 
#        integral = integral + 0.5*(integrand[i+1]+integrand[i])*delta
    return integral        