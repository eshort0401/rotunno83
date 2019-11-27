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
    psi = np.zeros((xiN,zetaN,tauN))
    u = np.zeros((xiN,zetaN,tauN))
    w = np.zeros((xiN,zetaN,tauN))

    # Initialise domains
    xi = np.linspace(-3,3,xiN)
    zeta = np.linspace(0,6,zetaN)

    xip = np.linspace(-10,10,xipN)
    zetap = np.linspace(0,20,zetapN)

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

def assign_units(ds):

    ds.xi.attrs['units'] = 'm'
    ds.zeta.attrs['units'] = 'm'
    ds.tau.attrs['units'] = 's'
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
    ds['w'] = ds.w * h * omega * (omega**2 - f**2)**(1/2)/N
    ds['psi'] = ds.psi * h**2 * omega
    if np.abs(f) < 2*omega*np.sin(np.deg2rad(30)):
        ds = ds.assign_coords(xi = ds.xi*N*h*(omega**2-f**2)**(-1/2))
    elif np.abs(f) > 2*omega*np.sin(np.deg2rad(30)):
        ds = ds.assign_coords(xi = ds.xi*N*h*((f**2-omega**2)**(-1/2)))
    ds.attrs['h'] = h
    ds.attrs['f'] = f
    ds.attrs['N'] = N
    ds = assign_units(ds)

    return ds

def redimensionalise_qian(ds,h=500,N=0.035,Q0=9.807*(3/300)/(12*3600)):

    omega = 2*np.pi/(24*3600)
    ds = ds.assign_coords(xi = ds.xi*N*h/omega)
    ds = ds.assign_coords(zeta = ds.zeta*h)
    ds = ds.assign_coords(tau = ds.tau/omega)
    ds['u'] = ds.u*Q0/(N*omega)
    ds['w'] = ds.w*Q0/(N**2)
    ds['psi'] = ds.psi*h*Q0/(N*omega)
    ds['bq'] = ds.bq*Q0/omega
    ds['bw'] = ds.bw*Q0/omega
    ds.attrs['L'] = ds.attrs['L']*N*h/omega
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
    if signed:
        varInc = varMax/10
        levels = np.arange(-varMax,varMax+varInc,varInc)
    else:
        varInc = (varMax-varMin)/10
        levels = np.arange(varMin,varMax+varInc,varInc)

    if ds.xi.attrs['units'] == 'm':
        x = ds.xi/1000
        z = ds.zeta/1000
        plt.xlabel('Distance [km]')
        plt.ylabel('Height [km]')
    else:
        x = ds.xi
        z = ds.zeta
        plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
        plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')

    contourPlot=plt.contourf(x, z, ds[var].isel(tau=t),
                            levels=levels, cmap=cmap)

    cbar = plt.colorbar(contourPlot)
    cbar.set_label('[' + ds[var].attrs['units'] + ']')
    plt.title(var + ' [' + ds[var].attrs['units'] + ']')

    if save:
        outFile='./../figures/{}_'.format(var) + get_current_dt_str() + '.png'
        fig.savefig(outFile, dpi=80, writer='imagemagick')

    return fig, ax, contourPlot, levels, x, z

def animateCont(ds, var='psi', cmap='RdBu_r'):

    plt.ioff()
    fig, ax, contourPlot, levels, x, z = plotCont(ds, var=var)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label, )
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot=ax.contourf(x, z, ds[var].isel(tau=i),
                                levels=levels, cmap=cmap)
        return contourPlot, ax

    anim = FuncAnimation(fig, update,
                         frames=np.arange(0, np.size(ds.tau)),
                         interval=200)

    outFile='./../figures/{}_'.format(var) + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=80, writer='imagemagick')

def plotVelocity(ds, t=0, save=False):

    plt.close('all')
    init_fonts()
    print('Plotting velocity.')

    # Velocity plot
    fig, ax = plt.subplots()

    ds['speed']=(ds.u**2+ds.w**2)**(1/2)
    speedMax=np.ceil(np.amax(ds.speed))
    speedInc=speedMax/20
    levels=np.arange(0,speedMax,speedInc)

    if ds.xi.attrs['units'] == 'm':
        x = ds.xi/1000
        z = ds.zeta/1000
        plt.xlabel('Distance [km]')
        plt.ylabel('Height [km]')
    else:
        x = ds.xi
        z = ds.zeta
        plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
        plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')


    contourPlot=ax.contourf(x, z, ds.speed.isel(tau=t),
                            cmap='Reds')

    skip=int(np.floor(np.size(ds.xi)/12))

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((ds.u**2 + ds.w**2)**(1/2))

    sQ=int(np.ceil(skip/2))
    dxi=ds.xi[1]-ds.xi[0]
    scale=(speedMaxQ/(skip*dxi)).values

    ax.quiver(x[sQ::skip], z[sQ::skip],
              ds.u.isel(tau=t)[sQ::skip,sQ::skip],
              ds.w.isel(tau=t)[sQ::skip,sQ::skip],
              units='xy', angles='xy', scale=scale)

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    if save:
        outFile='./../figures/velocity_' + get_current_dt_str() + '.png'
        plt.savefig(outFile, dpi=80, writer='imagemagick')

    return ds, fig, ax, contourPlot, levels, sQ, skip, scale, x, z

def animateVelocity(ds):

    plt.ioff()
    init_fonts()

    ds, fig, ax, contourPlot, levels, sQ, skip, scale, x, z = plotVelocity(ds,t=0)

    def update(i):
        label = 'Timestep {0}'.format(i)
        print(label)
        ax.collections = []
        contourPlot=ax.contourf(x, z, ds.speed.isel(tau=i),
                                cmap='Reds')
        ax.quiver(x[sQ::skip], z[sQ::skip],
                  ds.u.isel(tau=i)[sQ::skip,sQ::skip],
                  ds.w.isel(tau=i)[sQ::skip,sQ::skip],
                  units='xy', angles='xy', scale=scale)

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(ds.tau)),
                         interval=200)

    outFile='./../figures/velocity_' + get_current_dt_str() + '.gif'
    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")


def calcTheta(ds):

    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    xi0 = ds.xi0
    h = ds.h
    theta0 = ds.theta0
    theta1 = ds.theta1
    tropHeight = ds.tropHeight
    N = ds.N

    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)
    g = 9.80665

    thetaBar = np.zeros(np.shape(w))
    d_thetaBar_d_z = (theta1-theta0)/tropHeight
    d_thetaBar_d_zeta = h * d_thetaBar_d_z # Recall zeta * h = z

    for i in np.arange(0,np.size(tau)):
        thetaBar[:,:,i] = np.outer(
                np.ones(np.size(xi)), zeta * d_thetaBar_d_zeta
                )

    # Specify heating function array
    Q = np.zeros(np.shape(w))

    for i in np.arange(0,np.size(tau)):
        Q[:,:,i] = np.outer(
            ds.Atilde * ((np.pi / 2) + np.arctan(xi / xi0)),
            np.exp(-zeta)
            )
        Q[:,:,i] = Q[:,:,i] * np.sin(tau[i])

    LHS = np.zeros((np.size(tau),np.size(tau)))
    RHS = np.zeros(np.size(tau))
    dtau = tau[1]-tau[0]

    LHS[0, 0] = 1
    if (np.mod(np.size(tau),2) == 0):
        LHS[0,int(np.size(tau)/2)] = 1
    else:
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
        LHS[0,np.floor(np.size(tau)/2)] = 1/2

    RHS[0] = 0
    for k in np.arange(1,np.size(tau)):
        LHS[k, np.mod(k+1,32)] = 1
        LHS[k, k] = -1

    # Initialise bouyancy matrix
    btilde = np.zeros(np.shape(w))

    # Calculate bouyancy
    for i in np.arange(0,np.size(xi)):
        for j in np.arange(0, np.size(zeta)):
            for k in np.arange(1,np.size(tau)):
                RHS[k] = dtau * (Q[i,j,k] - ((N/omega) ** 2) * w[i,j,k])

            btilde[i,j,:] = np.linalg.solve(LHS,RHS)

    # Convert btilde to potential temperature perturbation
    thetaPrime = btilde * (omega ** 2) * h * theta0 / g

    theta = theta0 + thetaBar + thetaPrime

    ds = ds.assign(theta=(('xi','zeta','tau'),theta))
    ds = ds.assign(thetaBar=(('xi','zeta','tau'),thetaBar))
    ds = ds.assign(thetaPrime=(('xi','zeta','tau'),thetaPrime))

    return ds

def calc_v(ds):
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    lat=ds.lat

    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)
    f = 2 * omega * np.sin(np.deg2rad(lat))

    LHS = np.zeros((np.size(tau),np.size(tau)))
    RHS = np.zeros(np.size(tau))
    dtau = tau[1]-tau[0]

    LHS[0, 0] = 1
    if (np.mod(np.size(tau),2) == 0):
        LHS[0,int(np.size(tau)/2)] = 1
    else:
        LHS[0,np.floor(np.size(tau)/2)] = 1/2
        LHS[0,np.floor(np.size(tau)/2)] = 1/2

    RHS[0] = 0
    for k in np.arange(1,np.size(tau)):
        LHS[k, np.mod(k+1,np.size(tau))] = 1
        LHS[k, k] = -1

    # Initialise v matrix
    v = np.zeros(np.shape(w))

    # Calculate bouyancy
    for i in np.arange(0,np.size(xi)):
        for j in np.arange(0, np.size(zeta)):
            for k in np.arange(1,np.size(tau)):
                RHS[k] = - dtau * (f/omega) * w[i,j,k]

            v[i,j,:] = np.linalg.solve(LHS,RHS)

    ds = ds.assign(v=(('xi','zeta','tau'),v))

    return ds

def animate_v(ds):

    u = ds['u'].values.T
    v = ds['v'].values
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating v.')

    # Velocity plot
    fig, ax = plt.subplots()

    #vInc = np.round(np.ceil(np.max(np.abs(v))*10)/100,1)
    vMax = np.ceil(np.max(np.abs(v))*10)/10
    vInc = vMax/10
    levels=np.arange(-vMax,vMax+vInc,vInc)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,v[:,:,0],levels,vmin=-vMax,
                             vmax=vMax, cmap='RdBu_r')

    skip=int(np.floor(np.size(xi)/16))

    speed=(u**2+w**2)**(1/2)
    speedMax = np.ceil(np.max(speed)*10)/10

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('v Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('v [' + ds.v.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.collections = []

        contourPlot=ax.contourf(Xi,Zeta,v[:,:,i],levels,vmin=-vMax,
                             vmax=vMax, cmap='RdBu_r')

        ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,i], wQ[sQ::skip,sQ::skip,i],
               units='xy', scale=speedMaxQ/(skip*dxi))

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='./figures/v_velocity_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return

def animateTheta(ds):

    xi = ds['xi'].values
    zeta = ds['zeta'].values
    tau = ds['tau'].values
    theta = ds['theta'].values
    u = ds['u'].values.T
    w = ds['w'].values.T

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Animating theta.')

    # Velocity plot
    fig, ax = plt.subplots()

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    thetaMax=np.ceil(np.max(theta)/20)*20
    thetaMin=np.floor(np.min(theta)/20)*20

    if thetaMax-thetaMin > 200:
        thetaInc = 40
    elif thetaMax-thetaMin > 100:
        thetaInc = 20
    else:
        thetaInc = 10

    levels=np.arange(thetaMin,thetaMax+thetaInc,thetaInc)

    contourPlot = ax.contourf(
        Xi, Zeta, theta[:,:,0], cmap='Reds',
        levels=levels,
        )
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Potential Temperature [K]')

    ax.contour(
            Xi, Zeta, theta[:,:,0], colors='black',
            linestyles=None, levels=levels,
            linewidths=1.4
            )

    skip=int(np.floor(np.size(xi)/16))

    speed=(u**2+w**2)**(1/2)
    speedMax = np.ceil(np.max(speed)*10)/10

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('Potential Temperature [K]')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        global contourPlot

        ax.collections = []

        contourPlot = ax.contourf(
            Xi, Zeta, theta[:,:,i], cmap='Reds',
            levels=levels,
            )

        ax.contour(
            Xi, Zeta, theta[:,:,i], colors='black',
            linestyles=None, levels=levels,
            linewidths=1.4
            )

        ax.quiver(
            Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
            uQ[sQ::skip,sQ::skip,i], wQ[sQ::skip,sQ::skip,i],
            units='xy', scale=speedMaxQ/(skip*dxi)
            )

        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='./figures/theta_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return
