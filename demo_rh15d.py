# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec 
from helita.sim import rh15d
import os
import xarray
from fisspy.read import FISS
from sub import ReadFAL, Radtemp, Turbspeed, writerayinput, IntensityIntegral
from sub import TauIntegral
from scipy.interpolate import CubicSpline
data_kind =  'FALC Model' #'Model'

if data_kind == 'uniform' :
    zmax=5000.
    zmin=0.
    dz=-20.
    z=np.arange(zmax, zmin+dz, dz, dtype='float')
    z1=z
    T=(z-zmin)/(zmax-zmin)*0 +0.6e4
    nH=z*0+10.e13
    ne=z*0+1.e9
    vturb=z*0+1.
    vz=z*0+0.
#    outfile='/data/home/chae/rh/Atmos/test.hdf5'
if data_kind == 'FAL':    
    directory='/data/home/chae/rh/Atmos/'
    f=open(directory+'FALC.tab', 'r')
    data=(f.read()).split('\n')
    f.close()
    Ndep=int(data[2])
    data1=data[5:Ndep+5]
    z=np.arange(Ndep, dtype='float')
    T=np.arange(Ndep, dtype='float')
    vturb=np.arange(Ndep, dtype='float')
    nH=np.arange(Ndep, dtype='float')
    ne=np.arange(Ndep, dtype='float')
    vz=np.zeros(Ndep, dtype='float')
    for i in range(Ndep):
        z[i],T[i],nH[i], ne[i],vturb[i]=data1[i].split()    

if data_kind =='Model':
    file='ModelD1002.tab'
    directory='/data/home/chae/rh/Atmos/'
    f=open(directory+file, 'r')
    data=(f.read()).split('\n')
    f.close()
    Ndep=int(data[2][8:11])
    data1=data[4:Ndep+4]
    z=np.arange(Ndep, dtype='float')
    T=np.arange(Ndep, dtype='float')
    vturb=np.arange(Ndep, dtype='float')
    nH=np.arange(Ndep, dtype='float')
    ne=np.arange(Ndep, dtype='float')
    vz=np.zeros(Ndep, dtype='float')
    for i in range(Ndep):
        aa=data1[i].split()
        z[i]=float(aa[1])/1.e5
        T[i]=float(aa[2])
        nH[i] = float(aa[6])
        ne[i] = float(aa[3])
        vturb[i] =float(aa[7])/1.e5
#        z[i],T[i],nH[i], ne[i],vturb[i]=data1[i].split()    

if data_kind =='FALC Model':
    file='FALC.model'
    directory='/data/home/chae/rh/Atmos/'
    z,T,ne,nH = ReadFAL(directory+file)
    s=abs(T-1.5e4).argmin()
    z1=z[s:]
    T1=T[s:]
    ne1=ne[s:]
    nH1=nH[s:]
    n=301
    delta=(z1[0]-z1.min())/(n-1)
    z=z1[0]-np.arange(n)*delta
    cs = CubicSpline(-z1,np.log10(T1) )
    T=10.**cs(-z)
    cs = CubicSpline( -z1, np.log10(ne1))
    ne=10.**cs(-z)
    cs = CubicSpline(-z1, np.log10(nH1))
    nH=10.**cs(-z)
    vz=z*0.0
    vturb = 1.2*Turbspeed(np.log10(nH))    

outfile='/data/home/chae/rh/Atmos/model.hdf5'

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

#rh15d.make_xarray_atmos(outfile, z=z,T=T, nH=nHp,ne=ne,vturb=vturb,vz=vz) 

wv0= 656.28
wvarr0=np.arange(wv0-2, wv0+2, 0.002)
wv1= 854.21
wvarr1=np.arange(wv1-2, wv1+2, 0.002)

wvarr= np.concatenate((wvarr0,wvarr1))

#wvarr = np.concatenate((wvarr, np.arange(300., 900., 1.)))
wavefile='/data/home/chae/rh/Atoms/wave_files/tmp.wave'
os.system('rm '+ wavefile)
rh15d.make_wave_file(wavefile, new_wave=wvarr)
#for i in range(10):
plt.close('all')

fig = plt.figure(1)
gs = GridSpec(2,2,fig)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

ax0.set_xlim(zmin, zmax)
ax0.set_ylim(3000, 14000)
#ax0.set_xlabel('z (kim)')
ax0.set_ylabel('T')
p0 = ax0.plot(z[0,0,0,:]/1.e3,T[0,0,0,:])
ax1.set_xlim(zmin, zmax)
ax1.set_ylim(6, 16)
#ax1.set_xlabel('z (kim)')
ax1.set_ylabel('log ne')
p1 = ax1.plot(z[0,0,0,:]/1.e3,np.log10(ne[0,0,0,:]/1.E6))
ax2.set_xlim(zmin,zmax)
ax2.set_ylim(10, 18)
ax2.set_xlabel('z (kim)')
ax2.set_ylabel('log NH')
p2 = ax2.plot(z[0,0,0,:]/1.e3,np.log10(nHp[0,0,0,0,:]/1.E6))
ax3.set_xlim(9, 18)
ax3.set_xlabel('log NH')
ax3.set_ylabel('vturb')
p3 = ax3.plot(np.log10(nHp[0,0,0,0,:]/1.E6), vturb[0,0,0,:]/1.e3)
#xx=np.log10(nHp[0,0,0,0,:]/1.E6)
#xx=np.append(9., xx)
#small= xx < 16
#yy=(vturb[0,0,0,:]/1.e3)
##yy=np.append(15., yy)
#
#coeffs=np.polyfit(xx[small]-16., yy[small], 2)
#big = xx>=16
#coeffb=np.polyfit(xx[big]-16., yy[big], 2)
##coeff=np.array([0.28, 0.01, 0.83])
#xx=np.arange(9, 18, 0.1)
#small= xx< 16
#big=xx>=16
#yy=xx*0
#yy[small]=np.polyval(coeffs,xx[small]-16.)
#yy[big]=np.polyval(coeffb,xx[big]-16.)
#ax3.plot(xx, Turbspeed(xx) )



if 1 :
    Afile='/data/home/chae/FISS/FISS_20170614_165710_A1_c.fts'
    Bfile='/data/home/chae/FISS/FISS_20170614_165710_B1_c.fts'
    fissA=FISS(Afile)
    fissB=FISS(Bfile)
    Aprof=fissA.refProfile/fissA.refProfile.max()
    zeta=0.055 #0.065
    Aprof = (Aprof -zeta*Aprof.max())/(1.-zeta) 
    wvA = fissA.wave*0.1
    Bprof=fissB.refProfile/fissB.refProfile.max()
    Bprof = (Bprof -zeta*Bprof.max())/(1.-zeta) 
    wvB = fissB.wave*0.1


if 1:
    os.chdir('/data/home/chae/rh/rh15d/run/')
    os.system('rm output/*.hdf5')
    os.system('mpirun -np 3 rh15d_ray_pool')   
os.chdir('/data/home/chae/rh/rh15d/run/output') 
fig = plt.figure(figsize=[12., 16])
gs = GridSpec(3,2,fig)
ax0 = fig.add_subplot(gs[0,:])
ax1 = fig.add_subplot(gs[1,0])
ax2 = fig.add_subplot(gs[1,1])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[2,1])
rr=rh15d.Rh15dout()
atmos = xarray.open_dataset('/data/home/chae/rh/Atmos/model.hdf5')
z=rr.atmos.height_scale[0,0,:].data*1.e-3
temp=rr.atmos.temperature[0,0,:]
ray=rr.ray #xarray.open_dataset('output_ray.hdf5')
wvnm=ray.wavelength.values
inten=ray.intensity[0,0,:].values
chi=ray.chi[0,0,:,:].data
sf = ray.source_function[0,0,:,:].data
wvsel=ray.wavelength_selected
i0=abs(wvsel-wv0).argmin()
i1=abs(wvsel-wv1).argmin()
tau0=TauIntegral(z*1.e3, chi[:,i0])
tau1=TauIntegral(z*1.e3, chi[:,i1])

#wv0= 854.21 #656.285
#wv1= 854.21
#wvindex0=abs(wvnm-wv0).argmin()
#wvindex1=abs(wvnm-wv1).argmin()
#print('Line 0=', wv0, ' at ', wvindex0)
#print('Line 1=', wv1, ' at ', wvindex1)
wvs0=np.atleast_1d(wvarr0)
wvs1=np.atleast_1d(wvarr1)
n0=wvs0.size
index0=np.zeros(n0, dtype=int)
for i in range(n0):
    index0[i]= abs(wvnm-wvs0[i]).argmin()
n1=wvs1.size
index1=np.zeros(n1, dtype=int)
for i in range(n1):
    index1[i]= abs(wvnm-wvs1[i]).argmin()

index =np.append(index0, index1)    
writerayinput('../ray.input', 1.00, index)

#print('Line 2=', 393.37, ' at ', abs(wvnm-393.37).argmin())



print('sel wavelngth 0 =', wvsel[i0])
print('sel wavelngth 1 =', wvsel[i1])

tau_one_height=ray.tau_one_height[0,0,:]
Ts0=Radtemp(sf[:,i0], wvsel[i0])
Ts1=Radtemp(sf[:,i1], wvsel[i1])

ax0.set_ylim(2000,10000)
#ax0.plot(np.log10(tau0),temp, label='LTE')
#ax0.plot(np.log10(tau1),temp, label='LTE')
ax0.plot(np.log10(tau0), Ts0, color='b', label='Ha')
ax0.plot(np.log10(tau1), Ts1, color='r', label='Ca II 8542')
ax0.set_ylabel('Source Temperature (T)')
ax0.set_xlabel(rf'log $\tau_0$')
ax0.set_xlim(6, -3)

line1=(wvnm-wv1-1.)*(wvnm-wv1+1.) <= 0
line0=(wvnm-wv0-1.)*(wvnm-wv0+1.) <= 0
zc = np.interp(0., wvnm[line1]-wv1, tau_one_height[line1]/1.e3)
sc = np.interp(-zc, -z, ray.source_function[0,0,:,1])
Tc = np.interp(-zc, -z, temp)
ax1.set_ylim(0.1, 2.)
ax1.set_yscale('log')
ax1.set_xlabel(rf'$\lambda - \lambda_0$ (nm)')
ax2.set_xlabel(rf'$\lambda - \lambda_0$ (nm)')
ax3.set_xlabel(rf'$\lambda - \lambda_0$ (nm)')
ax4.set_xlabel(rf'$\lambda - \lambda_0$ (nm)')
ax1.set_ylabel('Intensity')
ax3.set_ylabel('Intensity')
#ax2.set_ylabel(rf'Height of $\tau=1$ (km)')
#ax4.set_ylabel(rf'Height of $\tau=1$ (km)')
   
#ax2.set_ylim(zmin,zmax)
ax3.set_ylim(0.1, 2.)
ax3.set_yscale('log')
#ax4.set_ylim(zmin,zmax)
ax1.set_xlim(-0.15*2, 0.15*2)
#ax2.set_xlim(-0.15, 0.15)
ax3.set_xlim(-0.15*2, 0.15*2)
ax4.set_xlim(-0.15, 0.15)

wv0ref=wvA.min()-wv0
ic0 = np.interp(wv0ref, wvnm[line0]-wv0, inten[line0])
aprof0=np.interp(wv0ref,wvA-wv0, Aprof)
wv1ref=wvB.max()-wv1
ic1 = np.interp(wv1ref, wvnm[line1]-wv1, inten[line1])
bprof0=np.interp(wv1ref,wvB-wv0, Bprof)


p1 = ax1.plot(wvnm[line0]-wv0-0.005, inten[line0]/ic0,  color='r', label='Model')
ax1.plot(wvA-wv0, Aprof/aprof0, color='k', label='Obs')
#p2 = ax2.plot(wvnm[line0]-wv0-0.005, tau_one_height[line0]/1.E3,  color='b')

p3 = ax3.plot(wvnm[line1]-wv1+0.005, inten[line1]/ic1,  color='r', label='Model')
ax3.plot(wvB-wv1, Bprof/bprof0, color='k', label='Obs')
#p4 = ax4.plot(wvnm[line1]-wv1+0.005, tau_one_height[line1]/1.E3,  color='r')




#print('zc=', zc, ', Tc=', Tc, ', sc/ic=', sc/3.81e-8)



# intenp=wvsel*0
# ns=len(wvsel)
# zmin=z[abs(z-1000).argmin()]
# for i in range(ns):
#     tau=TauIntegral(z*1.e3, chi[:,i ])    
#     intenp[i] = IntensityIntegral(tau, sf[:, i], 10.**np.interp(-zmin, -z, np.log10(tau)))
    
# ax1.plot(wvsel[0:n0].values-wv0-0.005, intenp[0:n0]/ic0)
# ax3.plot(wvsel[n0:].values-wv1+0.005, intenp[n0:]/ic1)    

# inten0a= np.interp(wvsel[0:n0], wvnm[line0], inten[line0])

# inten1a= np.interp(wvsel[n0:], wvnm[line1], inten[line1])

# p2 = ax2.plot(wvsel[0:n0]-wv0-0.005, inten0a/intenp[0:n0]-1.,  color='b')
# p4 = ax4.plot(wvsel[n0:]-wv1+0.005,  inten1a/intenp[n0:]-1.,  color='r')
tau0=z*0
tau1=z*0
for i in range(len(z)-1):
    tau1[i+1]=tau1[i]+0.5*(chi[i+1,i1]+chi[i,i1])*(z[i]-z[i+1])*1.e3
    tau0[i+1]=tau0[i]+0.5*(chi[i+1,i0]+chi[i,i0])*(z[i]-z[i+1])*1.e3

fg=plt.figure()

ax=fg.add_subplot(1,1,1)
ax.plot(z[1:], tau0[1:], label="Ha")
ax.plot(z[1:], tau1[1:], label="Ca")
ax.plot(z, z*0+1.)
ax.plot(z, z*0+10.)

ax.set_ylim(0.01, 1000.)
ax.set_yscale('log')