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


if 1 :
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

data_kind =  'FAL Model' #'Model'

if data_kind =='FAL Model':
    file='FALF.model'
    directory='/data/home/chokh2/rh/Atmos/'
    z,T,ne,nH = ReadFAL(directory+file)
    # s=abs(T-1.5e4).argmin()
    # s = abs(T-T.min()).argmin()
    s = 0
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

rh15d.make_xarray_atmos(outfile, z=z,T=T, nH=nHp,ne=ne,vturb=vturb,vz=vz) 

wv0= 656.282
wvarr0=np.arange(wv0-0.3, wv0+0.3, 0.005)
wv1= 854.209
wvarr1=np.arange(wv1-0.15, wv1+0.15, 0.0025)
wvarr= np.concatenate((wvarr0,wvarr1))
n0 = len(wvarr0)

rh15d.make_wave_file('/data/home/chokh2/rh/Atoms/wave_files/tmp.wave', new_wave=wvarr)
#for i in range(10):
plt.close('all')
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

os.chdir(r'/data/home/chokh2/rh/rh15d/run/')
if 0 :
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
with open('output_tm.p', 'rb') as file:
    res_tm = pickle.load(file)

Ha_cen_ind = res['inten'][:n0].argmin()
Ha_cen_wv = res['wv'][Ha_cen_ind]
Ca_cen_ind = res['inten'][n0:].argmin()+n0
Ca_cen_wv = res['wv'][Ca_cen_ind]

ind_tm = res['temp'].argmin()
z_tm = res['z'][ind_tm]

cont_f = res['sf']*np.exp(-res['tau'])*res['chi']
cont_f_tm = res['sf'][ind_tm:] \
            *np.exp(-(res['tau'][ind_tm:]-res['tau'][ind_tm])) \
            *res['chi'][ind_tm:]

dz = np.mean(res['z'][:-1] - res['z'][1:]) # in km 
# int_obs = np.sum(cont_f, axis=0)*dz*1e3
# int_tm = np.sum(cont_f_tm, axis=0)*dz*1e3
int_obs = res['wv']*0.
int_tm = res['wv']*0.
for i in range(len(int_obs)):
    int_obs[i] = IntensityIntegral(res['tau'][:, i], res['sf'][:, i], 0)
    int_tm[i] = IntensityIntegral(res['tau'][:, i], res['sf'][:, i], 
                                      res['tau'][ind_tm, i])

tau_one_cal = np.zeros(len(res['wv']))
tau_tm_one_cal = np.zeros(len(res['wv']))
for i in range(len(res['wv'])):
    tau_int = CubicSpline(res['tau'][:, i], res['z'])
    tau_tm_int = CubicSpline(res['tau'][ind_tm:, i]-res['tau'][ind_tm, i], 
                             res['z'][ind_tm:])
    tau_one_cal[i] = tau_int(1)
    tau_tm_one_cal[i] = tau_tm_int(1)

fig01, ((ax10, ax11), (ax12, ax13)) = plt.subplots(2, 2, figsize=[8, 6])

ax10.set_ylim(0.1, 2.)
ax10.set_yscale('log')
ax10.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
ax11.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
ax12.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
ax13.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
ax10.set_ylabel('Intensity')
# ax11.set_ylabel(r'$\dfrac{[I_{\nu}(h_{top}) - I_{\nu}(h_{tm})]} {I_\nu(h_{tm})}$')
ax11.set_ylabel('Contrast')
ax12.set_ylabel('Intensity')
ax13.set_ylabel('Contrast')
ax12.set_ylim(0.1, 2.)
ax12.set_yscale('log')
ax10.set_xlim(-0.15, 0.15)
ax11.set_xlim(-0.15, 0.15)
ax12.set_xlim(-0.15, 0.15)
ax13.set_xlim(-0.15, 0.15)

p101 = ax10.plot(res['wv'][:n0]-wv0, res['inten'][:n0]/res['inten'][0],
                 color='r', label=r'$I_\nu(h_{top})$')
p121 = ax12.plot(res['wv'][n0:]-wv1, res['inten'][n0:]/res['inten'][n0],
                 color='r', label=r'$I_\nu(h_{top})$')

t101 = ax10.text(0.49, 0.9, f'$\lambda (I_{{min}})$ = {Ha_cen_wv:.3f} nm', 
                 ha='right', transform=ax10.transAxes)
t102 = ax10.text(0.49, 0.82, f'$\lambda_0$ = {wv0:.3f} nm', 
                 ha='right', transform=ax10.transAxes)
Ha_lab_cen_wv = 1./(97491.219-82258.211)*1e7
t103 = ax10.text(0.99, 0.9, f'$\lambda_{{0, RH}}$ = {Ha_lab_cen_wv:.3f} nm', 
                 ha='right', transform=ax10.transAxes)

p103 = ax10.plot(res['wv'][:n0]-wv0, int_tm[:n0]/res['inten'][0], 'b', 
                 label=r'$I_\nu(h_{tm})$')
p123 = ax12.plot(res['wv'][n0:]-wv1, int_tm[n0:]/res['inten'][n0], 'b', 
                 label=r'$I_\nu(h_{tm})$')
aprof0 = Aprof[abs(res['wv'][0]-wvA).argmin()]
bprof0 = Bprof[abs(res['wv'][n0]-wvB).argmin()]
p102 = ax10.plot(wvA-wv0, Aprof/aprof0, color='k', label=r'$I_\nu(obs)$')
p122 = ax12.plot(wvB-wv1, Bprof/bprof0, color='k', label=r'$I_\nu(obs)$')

p104 = ax10.plot(res_tm['wv'][:n0]-wv0, res_tm['inten'][:n0]/res_tm['inten'][0], 
                 'b--', label=r'$I_\nu(h_{tm}, RH)$', alpha=0.3)
p124 = ax12.plot(res_tm['wv'][n0:]-wv1, res_tm['inten'][n0:]/res_tm['inten'][n0], 
                 'b--', label=r'$I_\nu(h_{tm}, RH)$', alpha=0.3)

p11 = ax11.plot(res['wv'][:n0]-wv0, 
                (res['inten'][:n0]-int_tm[:n0])/int_tm[:n0], 
                color='k')
p13 = ax13.plot(res['wv'][n0:]-wv1, 
                (res['inten'][n0:]-int_tm[n0:])/int_tm[n0:], 
                color='k')
t13 = ax13.text(0.03, 0.15, 
                r'$Cont_{\lambda}$='+'\n'+r'$\dfrac{I_{\nu}(h_{top}) - I_{\nu}(h_{tm})} {I_\nu(h_{tm})}$', 
                transform=ax13.transAxes)
p104 = ax10.plot([0, 0], ax10.get_ylim(), '--k', alpha=0.5)
p124 = ax12.plot([0, 0], ax12.get_ylim(), '--k', alpha=0.5)
leg10 = ax10.legend(frameon=False)
leg12 = ax12.legend(frameon=False)
fig01.tight_layout()
fig01.savefig('fig01.png', dpi=300)
#%% contribution function at the central wavelength

ind = np.arange(len(res['wv']))
ind0 = (ind < n0)
ind1 = (ind >= n0)

fig02 = plt.figure(figsize=(12, 8))
ax20 = fig02.subplots(2, 3)
cax20 = []
cb20 = []
im20 = []
p20 = []
p21 = []

for i in range(2):
    ind = [ind0, ind1][i]
    wv00 = [wv0, wv1][i]
    line = [r'H$\alpha$', 'Ca II 8542'][i]
    cen_ind = [Ha_cen_ind, Ca_cen_ind][i]
    cen_wv = [Ha_cen_wv, Ca_cen_wv][i]

    axp = ax20[i, 1]
    axp.set_title(r'C$_{\lambda}$ at the '+line+' center')
    axp.set_xlabel(r'log C$_{\lambda}$')
    axp.set_ylabel(r'height (km)')
    axp.set_ylim(-100, 2000)
    axp.set_xlim(-20, -12)

   
    for j in range(2):
        cont_dum = [cont_f, cont_f_tm][j]
        height = [r'h$_{top}$', r'h$_{tm}$'][j]
        z_range = [res['z'][0], res['z'][ind_tm]][j]
        tau_one = [tau_one_cal, tau_tm_one_cal][j]
        color = ['r', 'b'][j]
        ax = ax20[i, 2*j]
        ax.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
        ax.set_ylabel(r'height (km)')
        ax.set_title(r'C$_{\lambda}$('+height+')')
        ax.set_ylim(-100, 2000)
        ax.set_xlim(res['wv'][ind][0]-wv00, res['wv'][ind][-1]-wv00)
        im = ax.imshow(np.log10(cont_dum[::-1, ind]+1e-20), 
                       cmap='gray', origin='lower',
                       extent=[res['wv'][ind][0]-wv00, res['wv'][ind][-1]-wv00, 
                               res['z'][-1], z_range])
        im.set_clim(-15, -12)
        ax.set_aspect('auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="0%")
        cb = fig02.colorbar(im, cax=cax)
        cb.ax.set_ylabel(r'log C$_{\nu}$')
        im20.append(im)
        cb20.append(cb)
        cax20.append(cax)
        p21 = ax.plot([cen_wv-wv00]*2, ax.get_ylim(), '--w')
        p22 = ax.plot(res['wv'][ind]-wv00, tau_one[ind], 'c')
        t22 = ax.text(0.05, 0.92, r'$\mathbf{\tau_{\lambda} = 1}$', 
                      color='c', transform=ax.transAxes)
        if j == 0:
            p23 = ax.plot(res['wv'][ind]-wv00, 
                          res['tau_one_height'][ind]*1e-3, '--y', alpha=0.5)
            t23 = ax.text(0.05, 0.85, r'$\mathbf{\tau_{\lambda, RH} = 1}$', 
                          color='y', alpha=0.5,
                          transform=ax.transAxes)

        ind_z0 = [0, ind_tm][j]
        p_dum2 = axp.plot(np.log10(cont_dum[:, cen_ind]+1e-20), 
                         res['z'][ind_z0:], 
                         color=color)
        p_dum3 = axp.plot(axp.get_xlim(), [tau_one[cen_ind]]*2, '--'
                         'c')
        cont_max_z = res['z'][ind_z0:][cont_dum[:, cen_ind].argmax()]
        p_dum4 = axp.plot(axp.get_xlim(), [cont_max_z]*2, 
                         '--', color=color)
        # t_xpos = 0.45 if (i == 1 and j == 0) else 0.97
        t_ypos = [0.72-0.15*i, 0.37][j]
        axp.text(0.97, t_ypos, r'z($\tau_c$ = 1) = '+f'{tau_one[cen_ind]:.0f} km',
                 transform=axp.transAxes, color='c', ha='right')
        axp.text(0.97, t_ypos-0.06, 
                 r'z($C_{\lambda, max}$) = '+f'{cont_max_z:.0f} km',
                 transform=axp.transAxes, color=color, ha='right')

fig02.tight_layout()
fig02.savefig('fig02.png', dpi=300)
#%%

fig03 = plt.figure(figsize=(10, 7))
ax30 = fig03.subplots(2, 4)
im30 = []
for i in range(2):
    for j in range(2):
        k = i*2 + j
        obj = [res['chi'], res['tau'], res['sf'], res['sf']*res['chi']][k]
        title = [r'$\chi_{\nu}$', r'$\tau_{\nu}$', r'$S_{\nu}$', r'$j_{\nu}$'][k]
        cmap = ['gray', 'Blues_r', 'hot', 'YlGn_r'][k]
        for ii in range(2): 
            ind = [ind0, ind1][ii]
            wv00 = [wv0, wv1][ii]
            line = [r'H$\alpha$', 'Ca II 8542'][ii]
            ax = ax30[i, j*2+ii]
            ax.set_xlabel(r'$\lambda - \lambda_0$ (nm)')
            if ii == 0 : 
                ax.set_ylabel(r'height (km)')
                dum = ax.get_position()
            else : 
                ax.set_yticklabels('')
            ax.set_title(line+'  '+title+'  map')
            ax.set_ylim(-100, 2000)
            ax.set_xlim(res['wv'][ind][0]-wv00, res['wv'][ind][-1]-wv00)
            im = ax.imshow(np.log10(obj[::-1, ind]+1e-20), 
                           cmap=cmap, origin='lower',
                           extent=[res['wv'][ind][0]-wv00, res['wv'][ind][-1]-wv00, 
                                   res['z'][-1], res['z'][0]])
            im.autoscale()
            if k == 1: 
                con = ax.contour(res['wv'][ind]-wv00, res['z'][::-1], 
                                 obj[::-1, ind], [1], colors=['k'], 
                                 origin='lower', linestyles=['dotted'])
                im.set_clim(-2, 4)
            if k == 0: im.set_clim(-7, 0)
            if k == 3: im.set_clim(-15, -8)
            ax.set_aspect('auto')
            im30.append(im)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "5%", pad="0%")
        cb = fig03.colorbar(im, cax=cax)
        cb.ax.set_ylabel(r'log '+title)

    
fig03.tight_layout()
exc = 0.06
for i in range(2):
    for j in range(2):
        for k in range(2):
            dum = ax30[i, 2*j+k].get_position()
            if k == 0 :
                ax30[i, 2*j+k].set_position([dum.x0, dum.y0, dum.width+exc, dum.height])
            else :
                ax30[i, 2*j+k].set_position([dum.x0-exc, dum.y0, dum.width+exc, dum.height])
fig03.savefig('fig03.png', dpi=300)

tm_tau1_sf = res['wv']*0.
for i in range(len(res['wv'])):
    sf_int = interp1d(res['z'], res['sf'][:, i])
    tm_tau1_sf[i] = sf_int(tau_tm_one_cal[i])

    
taup = np.arange(0, 50, 1e-2)    
sf_tau = (res['wv']*0)[None, :] + taup[:, None]*0
for i in range(len(res['wv'])):
    sf_int = interp1d(res['tau'][:, i] - res['tau'][ind_tm, i], 
                      res['sf'][:, i])
    sf_tau[:, i] = sf_int(taup)    
exp_factor = np.zeros(len(res['wv']))[None, :] + np.exp(-taup)[:, None]

tau_from_tm = res['tau'] - res['tau'][ind_tm, :]
tau_from_tm[:ind_tm+1] = 1e-40    