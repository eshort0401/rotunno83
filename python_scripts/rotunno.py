# System
import datetime
import warnings

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
from continuous_N_helpers import integrate_continuous_N


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
    Atilde = .5*delTheta*(g/(np.pi*theta0))*h**(-1)*omega**(-3)/(12*60*60)

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

    tau = np.arange(0,2*np.pi,delTau)

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
    xi = np.linspace(-3,3,xiN)
    zeta = np.linspace(0,6,zetaN)
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
    x = np.linspace(-5, 5, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    z1 = np.linspace(0, H1, zN, dtype=np.float64)
    z2 = np.linspace(H1, 1.5*H1, zN, dtype=np.float64)
    z = np.concatenate([z1, z2[1:]])
    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w = integrate_piecewise_N(
        x, z, t, s, alpha, L, N, H1, zN, A)
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
            '../datasets/qian_{}.nc'.format(get_current_dt_str()),
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
    x = np.linspace(-15, 15, xN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    z1 = np.linspace(0, H1, zN, dtype=np.float64)
    zN_scaled = int(np.floor(zN*(H2-H1)/H1))
    z2 = np.linspace(H1, H2, zN_scaled+1, dtype=np.float64)
    z3 = np.linspace(H2, H2+.5*H1, int(np.floor(zN/2))+1)
    z = np.concatenate([z1, z2[1:], z3[1:]])

    t = np.arange(0, 2*np.pi, del_t, dtype=np.float64)
    s = np.arange(delS, 1, delS, dtype=np.float64)

    psi, u, w = integrate_continuous_N(
        x, z, t, s, alpha, L, N, H1, H2, zN, zN_scaled, A)
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
        ds.bq.attrs['units'] = 'm/s^2'
        ds.bw.attrs['units'] = 'm/s^2'
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
    # ds['bq'] = ds.bq*Q0/omega
    # ds['bw'] = ds.bw*Q0/omega
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
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

def get_current_dt_str():
    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    return dt


def plotCont(ds, var='psi', cmap='RdBu_r', signed=True, t=0, save=False):

    init_fonts()
    plt.close('all')

    print('Plotting {}.'.format(var))

    # psi plot
    fig, ax = plt.subplots()

    varMin = np.min(ds[var])
    varMax = np.max(ds[var])

    abs_max = np.max([np.abs(varMin), np.abs(varMax)])

    if signed:
        varInc = abs_max/10
        levels = np.arange(-abs_max, abs_max+varInc, varInc)
    else:
        varInc = (varMax-varMin)/10
        levels = np.arange(varMin, varMax+varInc, varInc)

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

    contourPlot = plt.contourf(
        x, z, ds[var].isel(t=t),
        levels=levels, cmap=cmap)

    cbar = plt.colorbar(contourPlot)
    cbar.set_label('[' + ds[var].attrs['units'] + ']')
    plt.title(var + ' [' + ds[var].attrs['units'] + ']')
    fig.patch.set_facecolor('white')

    if save:
        outFile = './../figures/{}_'.format(var)
        outFile += get_current_dt_str() + '.png'
        fig.savefig(outFile, dpi=100, writer='imagemagick')

    return fig, ax, contourPlot, levels, x, z


def animateCont(ds, var='psi', cmap='RdBu_r'):

    plt.ioff()
    fig, ax, contourPlot, levels, x, z = plotCont(ds, var=var)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot = ax.contourf(
            x, z, ds[var].isel(t=i), levels=levels, cmap=cmap)
        return contourPlot, ax

    anim = FuncAnimation(
        fig, update, frames=np.arange(0, np.size(ds.t)), interval=200)

    outFile = './../figures/{}_'.format(var) + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=200, writer='imagemagick')


def plotVelocity(ds, t=0, save=False):

    # import pdb
    # pdb.set_trace()

    plt.close('all')
    init_fonts()
    print('Plotting velocity.')

    # Velocity plot
    fig, ax = plt.subplots()

    ds['speed']=(ds.u**2+ds.w**2)**(1/2)
    speedMax=np.ceil(np.amax(ds.speed)*20)/20
    speedInc=speedMax/20
    levels=np.arange(0, speedMax+speedInc, speedInc)

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


    contourPlot=ax.contourf(x, z, ds.speed.isel(t=t),
                            cmap='Reds', levels=levels)

    skip=int(np.floor(np.size(ds.x)/12))

    sQ=int(np.ceil(skip/2))
    dx=x[1]-x[0]
    uQ = ds.u[:,sQ::skip,sQ::skip]
    wQ = ds.w[:,sQ::skip,sQ::skip]
    speedMaxQ=np.amax((uQ**2 + wQ**2)**(1/2)).values
    scale=(speedMaxQ/(skip*dx)).values

    ax.quiver(x[sQ::skip], z[sQ::skip],
              uQ.isel(t=t), wQ.isel(t=t),
              units='xy', angles='xy', scale=scale)

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    fig.patch.set_facecolor('white')

    if save:
        outFile='./../figures/velocity_' + get_current_dt_str() + '.png'
        plt.savefig(outFile, dpi=100, writer='imagemagick')

    return ds, fig, ax, contourPlot, levels, sQ, skip, scale, x, z, uQ, wQ


def animateVelocity(ds):

    plt.ioff()
    init_fonts()

    ds, fig, ax, contourPlot, levels, sQ, skip, scale, x, z, uQ, wQ = plotVelocity(ds, t=0)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)
        ax.collections = []
        contourPlot = ax.contourf(
            x, z, ds.speed.isel(t=i), cmap='Reds', levels=levels)

        ax.quiver(x[sQ::skip], z[sQ::skip],
                  uQ.isel(t=i), wQ.isel(t=i),
                  units='xy', angles='xy', scale=scale)

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(ds.t)),
                         interval=200)

    outFile='./../figures/velocity_' + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=200, writer='imagemagick')

    plt.close("all")
