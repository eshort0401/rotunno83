# System
import datetime
import warnings
import math
import string

# Performance
from numba import jit, prange

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

# Analysis
import numpy as np
import xarray as xr

# Helpers
from qian_helpers import integrate_qian, integrate_qian_U0
from qian_helpers import calc_theta, calc_k_2, calc_k_3
from rotunno_helpers import integrate_case_two, integrate_case_one
from channel_helpers import integrate_channel, integrate_channel_U0
from piecewise_N_helpers import integrate_piecewise_N
from piecewise_N_convective_helpers import integrate_piecewise_N_convective
from convective_helpers import integrate_convective
from continuous_N_helpers import integrate_continuous_N
from continuous_N_convective_helpers import integrate_continuous_N_convective


def calc_rotunno_parameters(theta0=300, delTheta=6, N=0.035,
                            latitude=-10, h=500):

    # Define constants
    omega = 2*np.pi/(24*3600)
    g = 9.80665
    # Calculate nondimensional model parameters from inputs
    f = 2.0*omega*np.sin(latitude * np.pi / 180.0)
    # N = np.sqrt((g/theta0)*(theta1-theta0)/tropHeight) # Brunt Vaisala Frequency
    if np.abs(latitude) < 30:
        beta = omega**2/(N*np.sqrt(omega**2-f**2))
    else:
        beta = omega**2/(N*np.sqrt(f**2-omega**2))
    Atilde = .5*h**(-1)*omega**(-3)/(12*60*60)

    return beta, Atilde, f, h


def solve_rotunno_case_one(xiN=41, zetaN=41, tauN=32, xipN=1000, zetapN=1000,
                           xi0=0.2, beta=7.27*10**(-3), Atilde=10**3):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN

    # Initialise solution arrays
    psi = np.zeros((tauN,zetaN,xiN))
    u = np.zeros((tauN,zetaN,xiN))
    w = np.zeros((tauN,zetaN,xiN))

    # Initialise domains
    xi = np.linspace(-3,3,xiN)
    zeta = np.linspace(0,6,zetaN)

    # This is so ugly fix it!
    # Make sure xip, zetap have even number of points to deal with annoying
    # log singularity. Need a better solution to this but struggling so hard...
    if (xipN % 2 == 0):
        xip = np.linspace(-10,10,xipN)
    else:
        xip = np.linspace(-10,10,xipN+1)

    if (zetaN % 2 == 0):
        zetap = np.linspace(0,6,xipN)
    else:
        zetap = np.linspace(0,6,xipN+1)

    tau = np.arange(0, 2*np.pi, delTau)

    print('Integrating')
    psi, u, w = integrate_case_one(xi,zeta,tau,xip,zetap,xi0,beta,Atilde)

    ds = xr.Dataset({'psi':(('tau','zeta','xi'),psi),
                     'u':(('tau','zeta','xi'),u),
                     'w':(('tau','zeta','xi'),w)},
                    {'tau': tau, 'zeta':zeta, 'xi':xi},
                    {'xi0':xi0, 'beta':beta, 'Atilde':Atilde})
    for var in ['psi', 'u', 'w', 'xi', 'zeta', 'tau']:
        ds[var].attrs['units'] = '-'

    print('Saving')
    ds.to_netcdf('../datasets/rotunno_case_1_{}.nc'.format(get_current_dt_str()),
                 encoding={'psi':{'zlib':True, 'complevel':9},
                           'u':{'zlib':True, 'complevel':9},
                           'w':{'zlib':True, 'complevel':9}})
    return ds


def solve_rotunno_case_two(xiN=161,zetaN=81,tauN=32,kN=2001,
                           xi0=0.2, beta=7.27*10**(-3), Atilde=10**3):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN

    # Initialise solution arrays
    psi = np.zeros((xiN,zetaN,tauN))
    u = np.zeros((xiN,zetaN,tauN))
    w = np.zeros((xiN,zetaN,tauN))

    # Initialise domains
    xi = np.linspace(-7,7,xiN)
    zeta = np.linspace(0,4,zetaN)
    tau = np.arange(0,2*np.pi,delTau)

    delS=1/kN
    s = np.arange(delS,1,delS, dtype=np.float64)
    theta = calc_theta(s,alpha=3)
    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta,1/(2*np.pi))
    # Get wavenumers less than 2*pi
    k_2 = calc_k_2(theta,1/(2*np.pi))
    k=np.concatenate([k_2[-1::-1], np.array([1/1/(2*np.pi)]), k_3])

    print('Integrating')
    psi, u, w = integrate_case_two(xi,zeta,tau,k,xi0,beta,Atilde)

    ds = xr.Dataset({'psi':(('tau','zeta','xi'),psi),
                     'u':(('tau','zeta','xi'),u),
                     'w':(('tau','zeta','xi'),w)},
                    {'tau': tau, 'zeta':zeta, 'xi':xi},
                    {'xi0':xi0, 'beta':beta, 'Atilde':Atilde})
    for var in ['psi', 'u', 'w', 'xi', 'zeta', 'tau']:
        ds[var].attrs['units'] = '-'

    print('Saving')
    ds.to_netcdf('../datasets/rotunno_case_2_{}.nc'.format(get_current_dt_str()),
                 encoding={'psi':{'zlib':True, 'complevel':9},
                           'u':{'zlib':True, 'complevel':9},
                           'w':{'zlib':True, 'complevel':9}})
    return ds


def solve_qian(xiN=241, zetaN=121, tauN=32, sN=2000, alpha=3,
               U=0.625, L=0.1, heat_right=True, save=True):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN
    delS = 1/sN
    delZeta = 4/zetaN

    # Initialise domains
    xi = np.linspace(-3,3,xiN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    zeta = np.linspace(0,6,zetaN, dtype=np.float64)
    tau = np.arange(0,2*np.pi,delTau, dtype=np.float64)
    s = np.arange(delS,1,delS, dtype=np.float64)

    print('Integrating')
    if U==0:
        psi, u, w, bq, bw = integrate_qian_U0(xi,zeta,tau,s,alpha,L,
                                              heat_right=heat_right)
        modes=2
    else:
        psi, u, w, bq, bw = integrate_qian(xi,zeta,tau,s,alpha,U,L,
                                           heat_right=heat_right)
        modes=3

    ds = xr.Dataset({'psi':(('mode','tau','zeta','xi'),psi),
                     'u':(('mode','tau','zeta','xi'),u),
                     'w':(('mode','tau','zeta','xi'),w),
                     'bq':(('tau','zeta','xi'),bq),
                     'bw':(('mode','tau','zeta','xi'),bw)},
                    {'mode': np.arange(1,modes+1), 'tau': tau,
                    'zeta':zeta, 'xi':xi},
                    {'U':U, 'L':L})

    print('Saving')
    for var in ['psi', 'u', 'w', 'xi', 'zeta', 'tau', 'bq', 'bw']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf('../datasets/qian_{}.nc'.format(get_current_dt_str()),
                     encoding={'psi':{'zlib':True, 'complevel':9},
                               'u':{'zlib':True, 'complevel':9},
                               'w':{'zlib':True, 'complevel':9}})

    return ds


def solve_piecewise_N(
        xN=241, zN=121, tN=32, sN=2000, alpha=2,
        L=0.1, N=2, H1=4, A=1, heat_right=True, save=True):

    print('Initialising.')

    # Time interval
    del_t = 2*np.pi/tN
    delS = 1/sN

    # Initialise domains
    # Normal
    x = np.linspace(-7, 7, xN, dtype=np.float64)
    # YMC
    # x = np.linspace(-15, 15, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there

    # Normal
    z1 = np.linspace(0, H1, zN, dtype=np.float64)
    z2 = np.linspace(H1, 2*H1, zN, dtype=np.float64)
    z = np.concatenate([z1, z2[1:]])

    # YMC
    # z1 = np.linspace(0, H1, zN, dtype=np.float64)
    # z2 = np.linspace(H1, H1+H1/4, int(zN/4), dtype=np.float64)
    # z = np.concatenate([z1, z2[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w, xi, bw, bq = integrate_piecewise_N(
        x, z, t, s, alpha, L, N, H1, zN, A)
    modes = 2

    # import pdb; pdb.set_trace()

    ds = xr.Dataset({
        'psi': (('mode', 'forcing', 't', 'z', 'x'), psi),
        'u': (('mode', 'forcing', 't', 'z', 'x'), u),
        'w': (('mode', 'forcing', 't', 'z', 'x'), w),
        'bq': (('t', 'z', 'x'), bq),
        'bw': (('mode', 'forcing', 't', 'z', 'x'), bw),
        'xi': (('mode', 'forcing', 't', 'z', 'x'), xi),
        },
        {
            'mode': np.arange(1, modes+1), 'forcing': np.arange(1, 3), 't': t,
            'z': z, 'x': x},
        {'L': L})

    print('Saving.')
    for var in ['psi', 'u', 'w', 'x', 'z', 't', 'xi', 'bw', 'bq']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf(
            '../datasets/qian_{}.nc'.format(get_current_dt_str()),
            encoding={
                'psi': {'zlib': True, 'complevel': 9},
                'u': {'zlib': True, 'complevel': 9},
                'w': {'zlib': True, 'complevel': 9},
                'xi': {'zlib': True, 'complevel': 9},
                'bw': {'zlib': True, 'complevel': 9},
                'bq': {'zlib': True, 'complevel': 9}
                })
    return ds


def solve_piecewise_N_convective(
        xN=241, zN=121, tN=32, sN=2000, alpha=2,
        L=1, D=1, N=2, H1=2, A=1, heat_right=True, save=True, z_top=None):

    if z_top is None:
        z_top = 2*H1

    print('Initialising.')

    # Time interval
    del_t = 2*np.pi/tN
    delS = 1/sN

    # Initialise domains
    # Normal
    x = np.linspace(-1, 1, xN, dtype=np.float64)
    # YMC
    # x = np.linspace(-15, 15, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there

    # Normal
    z1 = np.linspace(0, H1, zN, dtype=np.float64)

    zN2 = np.floor((z_top-H1)*zN/H1).astype(int)

    z2 = np.linspace(H1, z_top, zN2, dtype=np.float64)
    z = np.concatenate([z1, z2[1:]])

    # YMC
    # z1 = np.linspace(0, H1, zN, dtype=np.float64)
    # z2 = np.linspace(H1, H1+H1/4, int(zN/4), dtype=np.float64)
    # z = np.concatenate([z1, z2[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w = integrate_piecewise_N_convective(
        x, z, t, s, alpha, L, D, N, H1, zN, A)
    modes = 2

    ds = xr.Dataset({
        'psi': (('mode', 'forcing', 't', 'z', 'x'), psi),
        'u': (('mode', 'forcing', 't', 'z', 'x'), u),
        'w': (('mode', 'forcing', 't', 'z', 'x'), w),
        # 'bq': (('t', 'z', 'x'), bq),
        # 'bw': (('mode', 't', 'z', 'x'), bw)
        },
        {
            'mode': np.arange(1, modes+1), 'forcing': np.arange(1, 3), 't': t,
            'z': z, 'x': x},
        {'L': L})

    print('Saving.')
    for var in ['psi', 'u', 'w', 'x', 'z', 't']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf(
            '../datasets/qian_{}.nc'.format(get_current_dt_str()),
            encoding={
                'psi': {'zlib': True, 'complevel': 9},
                'u': {'zlib': True, 'complevel': 9},
                'w': {'zlib': True, 'complevel': 9},
                })
    return ds


def solve_convective(
        xN=241, zN=121, tN=32, sN=2000, alpha=2,
        L=1, D=1, A=1, heat_right=True, save=True):

    print('Initialising.')

    # Time interval
    del_t = 2*np.pi/tN
    delS = 1/sN

    # Initialise domains
    # Normal
    x = np.linspace(-3, 3, xN, dtype=np.float64)
    # YMC
    # x = np.linspace(-15, 15, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there

    # Normal
    z = np.linspace(0, 2, zN, dtype=np.float64)

    # YMC
    # z1 = np.linspace(0, H1, zN, dtype=np.float64)
    # z2 = np.linspace(H1, H1+H1/4, int(zN/4), dtype=np.float64)
    # z = np.concatenate([z1, z2[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w = integrate_convective(
        x, z, t, s, alpha, L, D, A)
    modes = 2

    ds = xr.Dataset({
        'psi': (('mode', 't', 'z', 'x'), psi),
        'u': (('mode', 't', 'z', 'x'), u),
        'w': (('mode', 't', 'z', 'x'), w),
        # 'bq': (('t', 'z', 'x'), bq),
        # 'bw': (('mode', 't', 'z', 'x'), bw)
        },
        {
            'mode': np.arange(1, modes+1), 't': t,
            'z': z, 'x': x},
        {'L': L})

    print('Saving.')
    for var in ['psi', 'u', 'w', 'x', 'z', 't']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf(
            '../datasets/convective_{}.nc'.format(get_current_dt_str()),
            encoding={
                'psi': {'zlib': True, 'complevel': 9},
                'u': {'zlib': True, 'complevel': 9},
                'w': {'zlib': True, 'complevel': 9}
                })
    return ds


def solve_continuous_N(
        xN=241, zN=121, tN=32, sN=2000, alpha=2,
        L=1, N=2, H1=5, H2=6, A=1, heat_right=True, save=True):

    print('Initialising')

    # Time interval
    del_t = 2*np.pi/tN
    delS = 1/sN

    # Initialise domains
    x = np.linspace(-7, 7, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    z1 = np.linspace(0, H1, zN, dtype=np.float64)
    zN_scaled = int(np.floor(zN*(H2-H1)/H1))
    z2 = np.linspace(H1, H2, zN_scaled+1, dtype=np.float64)
    z3 = np.linspace(H2, H2+H1, zN)
    z = np.concatenate([z1, z2[1:], z3[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w, xi, bw, bq = integrate_continuous_N(
        x, z, t, s, alpha, L, N, H1, H2, zN, zN_scaled, A)
    modes = 2

    ds = xr.Dataset({
        'psi': (('mode', 'forcing', 't', 'z', 'x'), psi),
        'u': (('mode', 'forcing', 't', 'z', 'x'), u),
        'w': (('mode', 'forcing', 't', 'z', 'x'), w),
        'bw': (('mode', 'forcing', 't', 'z', 'x'), bw),
        'xi': (('mode', 'forcing', 't', 'z', 'x'), xi),
        'bq': (('t', 'z', 'x'), bq),
        },
        {
            'mode': np.arange(1, modes+1), 'forcing': np.arange(1, 4), 't': t,
            'z': z, 'x': x},
        {'L': L})

    print('Saving.')
    for var in ['psi', 'u', 'w', 'xi', 'bw', 'bq', 'x', 'z', 't']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf(
            '../datasets/qian_{}.nc'.format(get_current_dt_str()),
            encoding={
                'psi': {'zlib': True, 'complevel': 9},
                'u': {'zlib': True, 'complevel': 9},
                'w': {'zlib': True, 'complevel': 9},
                'xi': {'zlib': True, 'complevel': 9},
                'bw': {'zlib': True, 'complevel': 9},
                'bq': {'zlib': True, 'complevel': 9}
                })
    return ds


def solve_continuous_N_convective(
        xN=241, zN=121, tN=32, sN=2000, alpha=2,
        L=1, N=2, H1=5, H2=6, A=1, D=1, heat_right=True,
        save=True, z_top=None):

    print('Initialising')

    # Time interval
    del_t = 2*np.pi/tN
    delS = 1/sN

    # Initialise domains
    x = np.linspace(-1, 1, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    z1 = np.linspace(0, H1, zN, dtype=np.float64)
    zN_scaled = int(np.floor(zN*(H2-H1)/H1))
    z2 = np.linspace(H1, H2, zN_scaled+1, dtype=np.float64)
    zN3 = int(np.floor((z_top-H2)*zN/H1).astype(int))
    z3 = np.linspace(H2, z_top, zN3, dtype=np.float64)
    z = np.concatenate([z1, z2[1:], z3[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w = integrate_continuous_N_convective(
        x, z, t, s, alpha, L, N, H1, H2, zN, zN_scaled, A, D)
    modes = 2

    ds = xr.Dataset({
        'psi': (('mode', 'forcing', 't', 'z', 'x'), psi),
        'u': (('mode', 'forcing', 't', 'z', 'x'), u),
        'w': (('mode', 'forcing', 't', 'z', 'x'), w),
        },
        {
            'mode': np.arange(1, modes+1), 'forcing': np.arange(1, 4), 't': t,
            'z': z, 'x': x},
        {'L': L})

    print('Saving.')
    for var in ['psi', 'u', 'w', 'x', 'z', 't']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf(
            '../datasets/qian_{}.nc'.format(get_current_dt_str()),
            encoding={
                'psi': {'zlib': True, 'complevel': 9},
                'u': {'zlib': True, 'complevel': 9},
                'w': {'zlib': True, 'complevel': 9}
                })
    return ds


def solve_channel(xiN=41, zetaN=21, tauN=32, sN=1000, alpha=3,
                  U=0, d=3, sigma=4, heat_island=True, save=False):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN
    delS = 1/sN
    delZeta = 4/zetaN

    # Initialise domains
    xi = np.linspace(-8,8,xiN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    zeta = np.linspace(0,8,zetaN, dtype=np.float64)
    tau = np.arange(0,2*np.pi,delTau, dtype=np.float64)
    s = np.arange(delS,1,delS, dtype=np.float64)

    print('Integrating')
    if U==0:
        psi, u, w, bq, bw = integrate_channel_U0(xi,zeta,tau,s,alpha,sigma,d,
                                                 heat_island=heat_island)
        modes=2
    else:
        psi, u, w, bq, bw = integrate_channel(xi,zeta,tau,s,alpha,U,sigma,d,
                                              heat_island=heat_island)
        modes=3

    ds = xr.Dataset({'psi':(('mode','tau','zeta','xi'),psi),
                     'u':(('mode','tau','zeta','xi'),u),
                     'w':(('mode','tau','zeta','xi'),w),
                     'bq':(('tau','zeta','xi'),bq),
                     'bw':(('mode','tau','zeta','xi'),bw)},
                    {'mode': np.arange(1,modes+1), 'tau': tau,
                    'zeta':zeta, 'xi':xi},
                    {'U':U, 'd':d, 'sigma':sigma})

    print('Saving')
    for var in ['psi', 'u', 'w', 'xi', 'zeta', 'tau', 'bq', 'bw']:
        ds[var].attrs['units'] = '-'

    if save:
        ds.to_netcdf('../datasets/qian_{}.nc'.format(get_current_dt_str()),
                     encoding={'psi':{'zlib':True, 'complevel':9},
                               'u':{'zlib':True, 'complevel':9},
                               'w':{'zlib':True, 'complevel':9}})

    return ds


def assign_units(ds):

    ds.x.attrs['units'] = 'm'
    ds.z.attrs['units'] = 'm'
    ds.t.attrs['units'] = 's'
    ds.u.attrs['units'] = 'm/s'
    ds.w.attrs['units'] = 'm/s'
    ds.psi.attrs['units'] = 'm^2/s'
    try:
        ds.xi.attrs['units'] = 'm'
        ds.bq.attrs['units'] = 'm/s^3'
        ds.bw.attrs['units'] = 'm/s^3'
    except:
        warnings.warn('Buoyancy not present.')

    return ds

def redimensionalise_rotunno(ds, h=500,
                             f=2*np.pi/(24*3600)*np.sin(np.deg2rad(-10)),
                             N=0.035):

    omega = 2*np.pi/(24*3600)
    ds = ds.assign_coords(zeta = ds.zeta * h)
    ds = ds.assign_coords(tau = ds.tau / omega)
    # Note this is different from the paper!
    ds['u'] = ds.u * h * omega
    ds['psi'] = ds.psi * h**2 * omega
    if np.abs(f) < 2*omega*np.sin(np.deg2rad(30)):
        ds = ds.assign_coords(xi = ds.xi*N*h*(omega**2-f**2)**(-1/2))
        ds['w'] = ds.w * h * omega * (omega**2 - f**2)**(1/2)/N
    elif np.abs(f) > 2*omega*np.sin(np.deg2rad(30)):
        ds = ds.assign_coords(xi = ds.xi*N*h*((f**2-omega**2)**(-1/2)))
        ds['w'] = ds.w * h * omega * (f**2-omega**2)**(1/2)/N
    ds.attrs['h'] = h
    ds.attrs['f'] = f
    ds.attrs['N'] = N
    ds = assign_units(ds)

    return ds


def redimensionalise_qian(
        ds, h=500, N=0.035, Q0=9.807*(3/300)/(12*3600)):

    omega = 2*np.pi/(24*3600)
    ds = ds.assign_coords(x=ds.x*N*h/omega)
    ds = ds.assign_coords(z=ds.z*h)
    ds = ds.assign_coords(t=ds.t/omega)
    ds['u'] = ds.u*Q0/(N*omega)
    ds['w'] = ds.w*Q0/(N**2)
    ds['psi'] = ds.psi*h*Q0/(N*omega)
    try:
        ds['bq'] = ds.bq*Q0/omega
        ds['bw'] = ds.bw*Q0/omega
        ds['xi'] = ds.xi*Q0/(N**2)/omega
    except KeyError:
        print('bq, bw or xi not present. Skipping.')
    ds.attrs['L'] = ds.attrs['L']*N*h/omega
    # ds.attrs['U'] = ds.attrs['U']*N*h
    ds.attrs['Q0'] = Q0
    ds.attrs['h'] = h
    ds.attrs['N'] = N
    ds.attrs['omega'] = omega
    ds = assign_units(ds)

    return ds


def redimensionalise_channel(ds,h=500,N=0.035,Q0=9.807*(3/300)/(12*3600)):

    omega = 2*np.pi/(24*3600)
    ds = ds.assign_coords(xi = ds.xi*N*h/omega)
    ds = ds.assign_coords(zeta = ds.zeta*h)
    ds = ds.assign_coords(tau = ds.tau/omega)
    ds['u'] = ds.u*Q0/(N*omega)
    ds['w'] = ds.w*Q0/(N**2)
    ds['psi'] = ds.psi*h*Q0/(N*omega)
    ds['bq'] = ds.bq*Q0/omega
    ds['bw'] = ds.bw*Q0/omega
    ds.attrs['d'] = ds.attrs['d']*N*h/omega
    ds.attrs['U'] = ds.attrs['U']*N*h
    ds.attrs['Q0'] = Q0
    ds.attrs['h'] = h
    ds.attrs['N'] = N
    ds.attrs['omega'] = omega
    ds = assign_units(ds)

    return ds


# Plotting functions
def init_fonts():
    plt.rc('text', usetex=False)

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'FreeSerif'})
    rcParams.update({'mathtext.fontset': 'dejavuserif'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})
    rcParams.update({'mathtext.fontset': 'cm'})


def get_current_dt_str():
    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    return dt


def plotDisp(
        ds, t=0, save=False,
        fig=None, ax=None, dlev=125, soltype='pwc'):

    init_fonts()

    dz = ds.z[1].values-ds.z[0].values
    zskip = int(np.ceil(dlev/dz))

    print('Plotting xi.')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # import pdb; pdb.set_trace()

    # psi plot
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()

    plt.gca()

    try:
        if ds.x.attrs['units'] == 'm':
            x = ds.x/1000
            z = ds.z/1000
            xi = ds['xi']/1000
            ax.set_xlabel('Distance [km]')
            ax.set_ylabel('Height [km]')
        else:
            x = ds.x
            z = ds.z
            xi = ds['xi']
            ax.set_xlabel('Distance [' + ds.x.attrs['units'] + ']')
            ax.set_ylabel('Height [' + ds.z.attrs['units'] + ']')
    except:
        for var in ds.keys():
            ds['xi'].attrs['units'] = '?'

    lines = []

    for k in range(zskip, len(z), zskip):
        # import pdb; pdb.set_trace()
        xik = xi.isel(t=t).isel(z=k) + z[k]
        line = ax.plot(x, xik, color=colors[0], linewidth=.75)
        lines.append(line[0])

    # x_max = np.max(x)
    # x_start, x_end, x_step = nice_bounds(-x_max, x_max, num_ticks=10)

    # plt.xticks(np.arange(-1500, 2000, 500))
    ax.set_xticks(np.arange(-750, 800, 250))
    # ax.set_yticks(np.arange(0, 30, 5))
    # ax.set_yticks(np.arange(0, 25, 2.5), minor=True)
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 4.5, .5), minor=True)

    # import pdb; pdb.set_trace()
    # ax.plot([x[0], x[-1]], [15, 15], 'k', dashes=(1, 5), zorder=4)
    # ax.plot([x[0], x[-1]], [19, 19], 'k', dashes=(1, 5), zorder=4)

    # ax.plot([x[0], x[-1]], [17, 17], 'k', dashes=(1, 5), zorder=4)

    # ax.plot([x[0], x[-1]], [2, 2], 'k', dashes=(1, 5), zorder=4)

    if soltype == 'pwc':
        ax.plot([x[0], x[-1]], [2, 2], 'k', dashes=(1, 5), zorder=4)
    else:
        ax.plot([x[0], x[-1]], [1.5, 1.5], 'k', dashes=(1, 5), zorder=4)
        ax.plot([x[0], x[-1]], [2.5, 2.5], 'k', dashes=(1, 5), zorder=4)

    # plt.title(var + ' [' + ds[var].attrs['units'] + ']')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.3)

    # z_start, z_end, z_step = nice_bounds(0, z[-1], 5)
    # plt.yticks(np.arange(z_start, z_end+z_step, z_step))

    t_s = int(ds.t.isel(t=t).values + 3600*12)
    h = int(t_s / 3600)
    m = int((t_s % 3600) / 60)
    ax.set_title('{:02d}:{:02d} LST'.format(h, m))

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(z[0], z[-1])

    if save:
        outFile = './../figures/{}_'.format(var)
        outFile += get_current_dt_str() + '.png'
        fig.savefig(outFile, dpi=100, writer='imagemagick')

    return fig, ax, lines, x, z, xi, zskip


def plotCont(
        ds, var='psi', cmap='RdBu_r', signed=True, t=0, save=False,
        fig=None, ax=None, cbar_steps=10, local_max=False, power_limits=False,
        abs_max=None, soltype='pwc'):

    init_fonts()
    # plt.close('all')

    print('Plotting {}.'.format(var))

    # import pdb; pdb.set_trace()

    # psi plot
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()

    if local_max:
        varMin = np.min(ds[var].isel(t=t))
        varMax = np.max(ds[var].isel(t=t))
    else:
        varMin = np.min(ds[var])
        varMax = np.max(ds[var])

    if abs_max is None:
        abs_max = np.max([np.abs(varMin), np.abs(varMax)])

    if signed:
        start, end, step = nice_bounds(-abs_max, abs_max, cbar_steps)
        levels = np.arange(start, end+step/2-1e-5, step/2)
    else:
        start, end, step = nice_bounds(0, abs_max, 5)
        levels = np.arange(0, end+step/2, step/2)
    try:
        if ds.x.attrs['units'] == 'm':
            x = ds.x/1000
            z = ds.z/1000
            plt.xlabel('Distance [km]')
            plt.ylabel('Height [km]')
        else:
            x = ds.x
            z = ds.z
            plt.xlabel('Distance [' + ds.x.attrs['units'] + ']')
            plt.ylabel('Height [' + ds.z.attrs['units'] + ']')
    except:
        for var in ds.keys():
            ds[var].attrs['units'] = '?'

    contourPlot = ax.contourf(
        x, z, ds[var].isel(t=t),
        levels=levels, cmap=cmap)

    plt.sca(ax)
    cbar = plt.colorbar(contourPlot)
    cbar.set_ticks(levels[::2])

    # x_max = np.max(x)
    # x_start, x_end, x_step = nice_bounds(-x_max, x_max, num_ticks=10)

    # plt.xticks(np.arange(-1500, 2000, 500))
    ax.set_xticks(np.arange(-750, 800, 250))
    # ax.set_yticks(np.arange(0, 30, 5))
    # ax.set_yticks(np.arange(0, 25, 2.5), minor=True)
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 4.5, .5), minor=True)

    # import pdb; pdb.set_trace()
    # ax.plot([x[0], x[-1]], [15, 15], 'k', dashes=(1, 5), zorder=4)
    # ax.plot([x[0], x[-1]], [19, 19], 'k', dashes=(1, 5), zorder=4)

    # ax.plot([x[0], x[-1]], [17, 17], 'k', dashes=(1, 5), zorder=4)

    # ax.plot([x[0], x[-1]], [2, 2], 'k', dashes=(1, 5), zorder=4)

    if soltype == 'pwc':
        ax.plot([x[0], x[-1]], [2, 2], 'k', dashes=(1, 5), zorder=4)
    else:
        ax.plot([x[0], x[-1]], [1.5, 1.5], 'k', dashes=(1, 5), zorder=4)
        ax.plot([x[0], x[-1]], [2.5, 2.5], 'k', dashes=(1, 5), zorder=4)

    if power_limits:
        cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()

    cbar.set_label('[' + ds[var].attrs['units'] + ']')
    # cbar.set_label(r'[m/s$^2$]')
    plt.title(var + ' [' + ds[var].attrs['units'] + ']')
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(hspace=0.3)

    # z_start, z_end, z_step = nice_bounds(0, z[-1], 5)
    # plt.yticks(np.arange(z_start, z_end+z_step, z_step))

    t_s = int(ds.t.isel(t=t).values + 3600*12)
    h = int(t_s / 3600)
    m = int((t_s % 3600) / 60)
    plt.title('{:02d}:{:02d} LST'.format(h, m))

    if save:
        outFile = './../figures/{}_'.format(var)
        outFile += get_current_dt_str() + '.png'
        fig.savefig(outFile, dpi=100, writer='imagemagick')

    return fig, ax, contourPlot, levels, x, z


def animateDisp(ds, soltype='pwc'):

    plt.ioff()
    fig, ax, lines, x, z, xi, zskip = plotDisp(ds, soltype=soltype)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)

        kvec = range(zskip, len(z), zskip)

        for j in range(len(lines)):

            k = kvec[j]

            xik = xi.isel(t=i).isel(z=k) + z[k]
            lines[j].set_data(x, xik)

        # import pdb; pdb.set_trace()

        t_s = int(ds['xi'].t.isel(t=i).values + 3600*12)
        h = int(t_s / 3600) % 24
        m = int((t_s % 3600) / 60)
        ax.set_title('{:02d}:{:02d} LST'.format(h, m))
        return lines, ax

    anim = FuncAnimation(
        fig, update, frames=np.arange(0, np.size(ds.t)), interval=200)

    outFile = './../figures/{}_'.format('xi') + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=200, writer='imagemagick')


def animateCont(ds, var='psi', cmap='RdBu_r', soltype='pwc'):

    plt.ioff()
    fig, ax, contourPlot, levels, x, z = plotCont(ds, var=var, soltype=soltype)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot = ax.contourf(
            x, z, ds[var].isel(t=i), levels=levels, cmap=cmap)
        t_s = int(ds[var].t.isel(t=i).values + 3600*12)
        h = int(t_s / 3600) % 24
        m = int((t_s % 3600) / 60)
        ax.set_title('{:02d}:{:02d} LST'.format(h, m))
        return contourPlot, ax

    anim = FuncAnimation(
        fig, update, frames=np.arange(0, np.size(ds.t)), interval=200)

    outFile = './../figures/{}_'.format(var) + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=200, writer='imagemagick')


def make_subplot_labels(axes, size=16, x_shift=-0.175, y_shift=1):
    labels = list(string.ascii_lowercase)
    labels = [label + ')' for label in labels]
    for i in range(len(axes)):
        axes[i].text(
            x_shift, y_shift, labels[i], transform=axes[i].transAxes,
            size=size)


def panelCont(
        ds, var='psi', cmap='RdBu_r', t_list=[0, 2, 4, 6], cbar_steps=20,
        abs_max=None, soltype='pwc'):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(len(t_list)):
        if var == 'xi':
            plotDisp(
                ds, t=t_list[i], save=False, fig=fig,
                ax=axes.flatten()[i], soltype=soltype)
        else:
            plotCont(
                ds, var=var, t=t_list[i], fig=fig, ax=axes.flatten()[i],
                cbar_steps=cbar_steps, abs_max=abs_max, soltype=soltype)
        # axes.flatten()[i].plot([-600, -600], [0, 25], '--', color='grey')
        # axes.flatten()[i].plot([-800, -800], [0, 25], '--', color='grey')
        # axes.flatten()[i].plot([-1000, -1000], [0, 25], '--', color='grey')
    make_subplot_labels(axes.flatten())


def contComparison(
        ds1, ds2, var='psi', cmap='RdBu_r', cbar_steps=10, t=0):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    plotCont(
        ds1, var=var, t=t, fig=fig, ax=axes[0],
        cbar_steps=cbar_steps)
    plotCont(
        ds2, var=var, t=t, fig=fig, ax=axes[1],
        cbar_steps=cbar_steps)
    axes[0].plot([-600, -600], [0, 25], '--', color='grey')
    axes[0].plot([-800, -800], [0, 25], '--', color='grey')
    axes[0].plot([-1000, -1000], [0, 25], '--', color='grey')
    axes[1].plot([-600, -600], [0, 25], '--', color='grey')
    axes[1].plot([-800, -800], [0, 25], '--', color='grey')
    axes[1].plot([-1000, -1000], [0, 25], '--', color='grey')

    make_subplot_labels(axes.flatten())


def forcingComparison(
        ds, var='psi', cmap='RdBu_r', cbar_steps=[10, 10, 10], t=[0, 0, 0]):

    forcings = len(ds.forcing)
    plt.close('all')
    fig, axes = plt.subplots(forcings, 1, figsize=(5, 3.6*forcings))
    for i in range(forcings):
        test = ds.isel(forcing=[i])
        test = test.sum(dim='mode', keep_attrs=True)
        test = test.sum(dim='forcing', keep_attrs=True)
        plt.sca(axes[i])
        plotCont(
            test, var=var, t=t[i], fig=fig, ax=axes[i],
            cbar_steps=cbar_steps[i], local_max=True)
    make_subplot_labels(axes.flatten())
    plt.subplots_adjust(hspace=.4)


def plotVelocity(
        ds, t=0, save=False, fig=None, ax=None, cbar_steps=10,
        local_max=False):

    # import pdb
    # pdb.set_trace()

    # plt.close('all')
    init_fonts()
    print('Plotting velocity.')

    # Velocity plot
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots()

    ds['speed'] = (ds.u**2+ds.w**2)**(1/2)
    if local_max:
        speedMax = np.amax(ds.speed.isel(t=t))
    else:
        speedMax = np.amax(ds.speed)

    start, end, step = nice_bounds(0, speedMax, cbar_steps)
    levels = np.arange(0, end+step/2, step/2)

    if ds.x.attrs['units'] == 'm':
        x = ds.x/1000
        z = ds.z/1000
        plt.xlabel('Distance [km]')
        plt.ylabel('Height [km]')
    else:
        x = ds.x
        z = ds.z
        plt.xlabel('Distance [' + ds.x.attrs['units'] + ']')
        plt.ylabel('Height [' + ds.z.attrs['units'] + ']')

    plt.sca(ax)

    contourPlot = ax.contourf(
        x, z, ds.speed.isel(t=t), cmap='Reds', levels=levels)

    skip_x = int(np.floor(np.size(ds.x)/16))
    skip_z = int(np.floor(np.size(ds.z)/16))

    sQ_x = int(np.ceil(skip_x/2))
    sQ_z = int(np.ceil(skip_z/2))
    dx = x[1]-x[0]
    uQ = ds.u[:, sQ_z::skip_z, sQ_x::skip_x]
    wQ = ds.w[:, sQ_z::skip_z, sQ_x::skip_x]
    speedMaxQ = np.amax((uQ**2 + wQ**2)**(1/2)).values
    scale = (speedMaxQ/(skip_x*dx)).values

    ax.quiver(
        x[sQ_x::skip_x], z[sQ_z::skip_z], uQ.isel(t=t), wQ.isel(t=t),
        units='xy', angles='xy', scale=.85*scale, width=8)

    # plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    cbar = plt.colorbar(contourPlot)
    cbar.set_ticks(levels[::2])
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    ax.set_xticks(np.arange(-750, 800, 250))
    ax.set_yticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 4.5, .5), minor=True)

    # ax.plot([x[0], x[-1]], [2, 2], 'k', dashes=(1, 5), zorder=4)

    # z_start, z_end, z_step = nice_bounds(0, z[-1], 5)
    # plt.yticks(np.arange(z_start, z_end+z_step, z_step))

    fig.patch.set_facecolor('white')

    t_s = int(ds.t.isel(t=t).values + 3600*12)
    h = int(t_s / 3600)
    m = int((t_s % 3600) / 60)
    plt.title('{:02d}:{:02d} LST'.format(h, m))
    plt.subplots_adjust(hspace=0.3)

    if save:
        outFile = './../figures/velocity_' + get_current_dt_str() + '.png'
        plt.savefig(outFile, dpi=100, writer='imagemagick')

    return ds, fig, ax, contourPlot, levels, sQ_x, sQ_z, skip_x, skip_z, scale, x, z, uQ, wQ


def animateVelocity(ds):

    plt.ioff()
    init_fonts()

    ds, fig, ax, contourPlot, levels, sQ_x, sQ_z, skip_x, skip_z, scale, x, z, uQ, wQ = plotVelocity(ds, t=0)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)
        ax.collections = []
        contourPlot = ax.contourf(
            x, z, ds.speed.isel(t=i), cmap='Reds', levels=levels)

        ax.quiver(x[sQ_x::skip_x], z[sQ_z::skip_z],
                  uQ.isel(t=i), wQ.isel(t=i),
                  units='xy', angles='xy', scale=scale)

        t_s = int(ds.t.isel(t=i).values + 3600*6)
        h = int(t_s / 3600) % 24
        m = int((t_s % 3600) / 60)
        plt.title('{:02d}:{:02d} LST'.format(h, m))

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(ds.t)),
                         interval=200)

    outFile='./../figures/velocity_' + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=200, writer='imagemagick')

    plt.close("all")


def panelVelocity(ds, var='psi', cmap='RdBu_r', t_list=[0, 2, 4, 6]):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(len(t_list)):
        plotVelocity(ds, t=t_list[i], fig=fig, ax=axes.flatten()[i])
    make_subplot_labels(axes.flatten())


def panelDiffTypes(ds, t=4):
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    plotCont(
        ds, var='w', t=t, fig=fig, ax=axes.flatten()[0], local_max=True,
        cbar_steps=10, power_limits=True)
    plotVelocity(
        ds, t=t, fig=fig, ax=axes.flatten()[1], cbar_steps=6, local_max=True)
    make_subplot_labels(axes.flatten())


def panelForcing(ds, t=3):
    plt.close('all')
    nF = len(ds.forcing)
    fig, axes = plt.subplots(nF, 1, figsize=(6, 4.5*nF))
    for i in range(nF):
        plt.sca(axes.flatten()[nF-i-1])
        ds_i = ds.isel(forcing=i)
        plotVelocity(ds_i, t=t, fig=fig, ax=axes.flatten()[nF-i-1])
    make_subplot_labels(axes.flatten())


def nice_number(value, round_=False):
    '''nice_number(value, round_=False) -> float'''
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5:
            nice_fraction = 1.
        elif fraction < 2.5:
            nice_fraction = 2.
        elif fraction < 6:
            nice_fraction = 5.
        else:
            nice_fraction = 10.
    else:
        if fraction <= 1:
            nice_fraction = 1.
        elif fraction <= 2:
            nice_fraction = 2.
        elif fraction <= 5:
            nice_fraction = 5.
        else:
            nice_fraction = 10.

    return nice_fraction * 10 ** exponent


def nice_bounds(axis_start, axis_end, num_ticks=10):
    '''
    nice_bounds(axis_start, axis_end, num_ticks=10) -> tuple
    @return: tuple as (nice_axis_start, nice_axis_end, nice_tick_width)
    '''
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (num_ticks - 1), round_=True)
        axis_start = math.floor(axis_start / nice_tick) * nice_tick
        axis_end = math.ceil(axis_end / nice_tick) * nice_tick

    return axis_start, axis_end, nice_tick
