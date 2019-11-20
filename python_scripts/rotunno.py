# System
import datetime

# Performance
from numba import jit, prange

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation

# Analysis
import numpy as np
import xarray as xr

from scipy.special import exp1

def solve_rotunno_case_two(xiN=161,zetaN=81,tauN=32,kN=2001,
                           xi0=0.2, latitude=5, h=1500,
                           delTheta=6, theta0=300, theta1=350,
                           tropHeight=17000):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN

    # Initialise solution arrays
    psi = np.zeros((xiN,zetaN,tauN))
    u = np.zeros((xiN,zetaN,tauN))
    v = np.zeros((xiN,zetaN,tauN))
    w = np.zeros((xiN,zetaN,tauN))
    b = np.zeros((xiN,zetaN,tauN))

    # Initialise domains
    xi = np.linspace(-2,2,xiN)
    zeta = np.linspace(0,4,zetaN)
    tau = np.arange(0,2*np.pi,delTau)

    k = np.linspace(0,40,kN)

    # Define constants
    omega = 7.2921159 * 10.0 ** (-5)
    g = 9.80665

    # Calculate nondimensional model parameters from inputs
    f = 2.0*omega*np.sin(latitude * np.pi / 180.0)
    N = np.sqrt((g/theta0)*(theta1-theta0)/tropHeight) # Brunt Vaisala Frequency
    beta = omega**2/(N*np.sqrt(omega**2-f**2))
    Atilde = .5*delTheta*(g/(np.pi*theta0))*h**(-1)*omega**(-3)/(12*60*60)

    print('Integrating')
    psi, u, w = integrate_case_two(xi,zeta,tau,k,xi0,beta,Atilde)

    ds = xr.Dataset({'psi':(('tau','zeta','xi'),psi),
                     'u':(('tau','zeta','xi'),u),
                     'w':(('tau','zeta','xi'),w)},
                    {'tau': tau, 'zeta':zeta, 'xi':xi},
                    {'kN':kN, 'xi0':xi0, 'latitude':latitude, 'h':h,
                     'delTheta':delTheta, 'theta0':theta0,
                     'theta1':theta1, 'tropHeight':tropHeight})

    print('Saving')
    now = str(datetime.datetime.now())[0:-7]
    now=now.replace('-', '').replace(':', '').replace(' ', '_')
    ds.to_netcdf('../datasets/rotunno_case_2_{}.nc'.format(now),
                 encoding={'psi':{'zlib':True, 'complevel':9},
                           'u':{'zlib':True, 'complevel':9},
                           'w':{'zlib':True, 'complevel':9}})
    return ds

@jit(parallel=True)
def integrate_case_two(xi,zeta,tau,k,xi0,beta,Atilde):
    psi = np.zeros((tau.size, zeta.size, xi.size))
    u = np.zeros((tau.size, zeta.size, xi.size))
    w = np.zeros((tau.size, zeta.size, xi.size))
    psi_integrand = np.zeros(k.size)
    u_integrand = np.zeros(k.size)
    w_integrand = np.zeros(k.size)
    # Perform numerical integration
    for i in prange(xi.size):
        for j in range(zeta.size):
            for l in range(tau.size):
                psi_integrand = (np.cos(k*xi[i])
                                *np.exp(-xi0*k)
                                *(np.sin(k*zeta[j]+tau[l])
                                  -np.exp(-zeta[j])*np.sin(tau[l]))
                                /(1+k**2))
                u_integrand = (k*np.sin(k*xi[i])
                              *np.exp(-xi0*k)
                              *(np.sin(k*zeta[j]+tau[l])
                                -np.exp(-zeta[j])*np.sin(tau[l]))
                              /(1+k**2))
                w_integrand = (np.cos(k*xi[i])
                              *np.exp(-xi0*k)
                              *(k*np.cos(k*zeta[j]+tau[l])
                                +np.exp(-zeta[j])*np.sin(tau[l]))
                              /(1+k**2))

                psi[l,j,i] = np.trapz(k,psi_integrand)
                u[l,j,i] = np.trapz(k,u_integrand)
                w[l,j,i] = np.trapz(k,w_integrand)
    # Scale
    psi = -beta * Atilde * psi
    u = -beta * Atilde * u
    w = -beta * Atilde * w
    return psi, u, w

def solve_qian(xiN=241,zetaN=200,tauN=32,sN=1999,alpha=3,U=0.625,L=0.1):

    print('Initialising')

    # Time interval
    delTau = 2*np.pi/tauN
    delS = 1/sN
    delZeta = 10/zetaN

    # Initialise solution arrays
    psi = np.zeros((xiN,zetaN,tauN), dtype=np.complex64)
    u = np.zeros((xiN,zetaN,tauN), dtype=np.complex64)
    w = np.zeros((xiN,zetaN,tauN), dtype=np.complex64)

    # Initialise domains
    xi = np.linspace(-6,6,xiN, dtype=np.float64)
    # Dont start at zero as exponential integral not defined there
    zeta = np.arange(delZeta,10+delZeta,delZeta, dtype=np.float64)
    tau = np.arange(0,2*np.pi,delTau, dtype=np.float64)
    s = np.arange(delS,1,delS, dtype=np.float64)

    print('Integrating')
    psi, u, w = integrate_qian(xi,zeta,tau,s,alpha,U,L)

    ds = xr.Dataset({'psi':(('mode','tau','zeta','xi'),psi),
                     'u':(('mode','tau','zeta','xi'),u),
                     'w':(('mode','tau','zeta','xi'),w)},
                    {'mode': np.arange(1,4), 'tau': tau, 'zeta':zeta, 'xi':xi})

    print('Saving')
    now = str(datetime.datetime.now())[0:-7]
    now=now.replace('-', '').replace(':', '').replace(' ', '_')
    ds.attrs['U'] = U
    ds.attrs['L'] = L
    ds.to_netcdf('../datasets/qian_{}.nc'.format(now),
                 encoding={'psi':{'zlib':True, 'complevel':9},
                           'u':{'zlib':True, 'complevel':9},
                           'w':{'zlib':True, 'complevel':9}})

    for var in ['psi', 'u', 'w', 'xi', 'zeta', 'tau']:
        ds[var].attrs['units'] = '-'

    return ds

@jit(parallel=True)
def integrate_qian(xi,zeta,tau,s,alpha,U,L):

    psi = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)
    u = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)
    w = np.zeros((3, tau.size, zeta.size, xi.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s,alpha=3)

    k_3 = calc_k_3(theta,U)
    k0_3 = calc_k_3(0,U)

    k_2 = calc_k_2(theta,U)
    k0_2 = calc_k_2(0,U)

    # Don't need to worry about waves 2*pi/0.05 > 125 or < 0.005
    k_1 = np.linspace(0.2,1500,s.size)

    # Perform numerical integration
    for i in prange(xi.size):
        for j in range(zeta.size):
            for l in range(tau.size):

                # Calc psi1
                psi1_ig = calc_psi1(xi[i],zeta[j],tau[l],k_1,U,L)
                psi[0][l,j,i] = np.trapz(k_1,psi1_ig)

#                 pdb.set_trace()

                # Calc psi2
                psi2a_ig = calc_psi2a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                psi2b_ig = calc_psi2b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                psi2ab = np.trapz(s, psi2a_ig+psi2b_ig)

                psi2c = (-1/(2*U)*np.exp(1j*zeta[j]/U)
                         *calc_C2(xi[i],tau[l],k0_2,U,L)
                         *calc_ep0(-1j*zeta[j]/U))

                psi[1][l,j,i] = psi2ab+psi2c

                # Calc psi3
                psi3a_ig = calc_psi3a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                psi3b_ig = calc_psi3b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                psi3a = np.trapz(s, psi3a_ig)
                psi3b = np.trapz(s, psi3b_ig)

                psi3c = (-U/2*calc_C3(xi[i],tau[l],k0_3,U,L)
                         *calc_ep0(-1j*zeta[j]/U))

                psi[2][l,j,i] = psi3a+psi3b+psi3c


                # Calc u1
                u1_ig = calc_u1(xi[i],zeta[j],tau[l],k_1,U,L)
                u[0][l,j,i] = np.trapz(k_1,u1_ig)

                # Calc u2
                u2a_ig = calc_u2a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                u2b_ig = calc_u2b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                u2ab = np.trapz(s,u2a_ig+u2b_ig)

                u2c = (-1j*np.exp(1j*zeta[j]/U)/(2*U**2)
                       *calc_C2(xi[i],tau[l],k0_2,U,L)
                       *(calc_ep0(-1j*zeta[j]/U)
                         -calc_exp1(1j*zeta[j]/U)))

                u[1][l,j,i] = u2ab+u2c

                # Calc u3
                u3a_ig = calc_u3a(xi[i],zeta[j],tau[l],s,alpha,U,L)
                u3b_ig = calc_u3b(xi[i],zeta[j],tau[l],s,alpha,U,L)

                u3ab = np.trapz(s,u3a_ig+u3b_ig)

                u3c = (-1j/(2*np.pi)
                       *calc_C3(xi[i],tau[l],k0_3,U,L)
                       *calc_exp1(-1j*zeta[j]/U))

                u[2][l,j,i] = u3ab+u3c

                # Calc w1
                w1_ig = psi1_ig*1j*k_1
                w[0][l,j,i] = np.trapz(k_1,w1_ig)

                # Calc w2
                w2ab = np.trapz(s, 1j*(1-np.sin(theta))/U*(psi2a_ig+psi2b_ig))
                w2c = (-1j/(2*U**2)*np.exp(1j*zeta[j]/U)
                       *calc_C2(xi[i],tau[l],k0_2,U,L)
                       *(calc_ep0(-1j*zeta[j]/U)-calc_ep1(-1j*zeta[j]/U)))

                w[1][l,j,i] = w2ab+w2c

                # Calc w3
                w3ab = np.trapz(s, 1j*(1-np.sin(theta))/U*(psi3a_ig+psi3b_ig))
                w3c = (-1j/2*calc_C3(xi[i],tau[l],k0_3,U,L)
                       *(calc_ep0(1j*zeta[j]/U)-calc_ep1(1j*zeta[j]/U)))

                w[2][l,j,i] = w3ab+w3c

    psi = np.real(psi)
    u = np.real(u)
    w = -np.real(w)
    return psi, u, w

@jit(parallel=True)
def calc_exp1(z,n=200):
#     s = np.float64(0)
#     for i in np.arange(1,n):
#         i_f = np.float64(i)
#         s += (-z)**i_f/(i_f*np.math.factorial(i_f))
#     return -np.euler_gamma-np.log(z)-s
    return exp1(z)

@jit(parallel=True)
def calc_ep0(z):
    # Note this formula only applies if Re(z)>=0!
    return np.exp(z) + z*calc_exp1(-z)

@jit(parallel=True)
def calc_ep1(z):
    return (np.exp(z) + z*calc_ep0(z))/2

@jit(parallel=True)
def calc_psi1(xi,zeta,tau,k,U,L):
    m=k/(1+k*U)
    psi1 = (-(1/2)*np.exp(-k*L)/(k**2+m**2)
            *(np.exp(1j*m*zeta)-np.exp(-zeta))
            *np.exp(1j*(k*xi+tau)))
    return psi1

@jit(parallel=True)
def calc_psi2a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s,alpha)
    k = calc_k_2(theta,U)
    # Note only difference between u2a is sign
    psi2a = (1/(2*U)*calc_C2(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi2a

@jit(parallel=True)
def calc_psi2b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k0 = calc_k_2(0,U)
    psi2b = (-1/(2*U)*np.exp(1j*zeta/U)
             *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k0,U,L))
             *np.exp(-1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi2b

@jit(parallel=True)
def calc_psi3a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha)
    k = calc_k_3(theta,U)
    psi3a = (U/2*calc_C3(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi3a

@jit(parallel=True)
def calc_psi3b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_3(theta,U)
    k0 = calc_k_3(0,U)
    psi3b = (-U/2*(calc_C3(xi,tau,k,U,L)-calc_C3(xi,tau,k0,U,L))
             *np.exp(1j*zeta/(U*np.sin(theta)))*np.cos(theta)
             *alpha*s**(alpha-1)*np.pi/2)
    return psi3b

@jit(parallel=True)
def calc_u1(xi,zeta,tau,k,U,L):
    m=k/(1+k*U)
    u1 = (-(1/2)*np.exp(-k*L)/(k**2+m**2)
          *(1j*m*np.exp(1j*m*zeta)+np.exp(-zeta))
          *np.exp(1j*(k*xi+tau)))
    return u1

@jit(parallel=True)
def calc_u3a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha)
    k = calc_k_3(theta,U)
    u3a = (-U/(2*np.pi)*calc_C3(xi,tau,k,U,L)
           *np.exp(-zeta)*np.cos(theta)*alpha*s**(alpha-1)*np.pi/2)
    return u3a

@jit(parallel=True)
def calc_u3b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_3(theta,U)
    k_0 = calc_k_3(0,U)
    u3b = (-1j/(2*np.pi)*(calc_C3(xi,tau,k,U,L)-calc_C3(xi,tau,k_0,U,L))
           *np.exp(1j*zeta/(U*np.sin(theta)))*(1/np.tan(theta))
           *alpha*s**(alpha-1)*np.pi/2)
    return u3b

@jit(parallel=True)
def calc_u2a(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s,alpha=alpha)
    k = calc_k_2(theta,U)
    u2a = (-1/(2*U)*calc_C2(xi,tau,k,U,L)*np.exp(-zeta)*np.cos(theta)
           *alpha*s**(alpha-1)*np.pi/2)
    return u2a

@jit(parallel=True)
def calc_u2b(xi,zeta,tau,s,alpha,U,L):
    theta = calc_theta(s, alpha=alpha)
    k = calc_k_2(theta,U)
    k_0 = calc_k_2(0,U)
    u2b = (-1j/(2*U**2)*np.exp(1j*zeta/U)
           *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k_0,U,L))
           *np.exp(-1j*zeta/(np.sin(theta)*U))*np.cos(theta)
           *alpha*s**(alpha-1)*np.pi/2
           +1j/(2*U**2)*np.exp(1j*zeta/U)
           *(calc_C2(xi,tau,k,U,L)-calc_C2(xi,tau,k_0,U,L))
           *np.exp(-1j*zeta/(np.sin(theta)*U))*np.cos(theta)/np.sin(theta)
           *alpha*s**(alpha-1)*np.pi/2)
    return u2b

@jit(parallel=True)
def calc_C2(xi,tau,k,U,L):
    C2 = np.exp(-k*L)*np.exp(1j*(k*xi-tau))/(k**2+(U*k-1)**2)
    return C2

@jit(parallel=True)
def calc_C3(xi,tau,k,U,L):
    C3 = k**2*calc_C2(xi,tau,k,U,L)
    return C3

@jit(parallel=True)
def calc_theta(s,alpha=3):
    theta = (np.pi/2)*s**alpha
    return theta

@jit(parallel=True)
def calc_k_3(theta,U):
    k = U/(1-np.sin(theta))
    return k

@jit(parallel=True)
def calc_k_2(theta,U):
    k = (1-np.sin(theta))/U
    return k

def redimensionalise(ds, h, f, N):

    # Specify constants
    omega = 7.2921159 * 10.0 ** (-5)

    ds = ds.assign_coords(zeta = ds.zeta * h)
    ds.zeta.attrs['units'] = 'm'
    ds = ds.assign_coords(tau = ds.tau / omega)
    ds.tau.attrs['units'] = 's'
    ds['u'] = ds.u * h * omega
    ds.u.attrs['units'] = 'm/s'
    ds['v'] = ds.v * h * omega
    ds.v.attrs['units'] = 'm/s'
    ds['w'] = ds.w * h * omega
    ds.w.attrs['units'] = 'm/s'
    ds['psi'] = ds.psi * h**2 * omega
    ds.psi.attrs['units'] = 'm^2/s'

    if ds.lat < 30:
        ds = ds.assign_coords(xi = ds.xi * N * h * ((omega**2 - f**2)**(-1/2)))
    elif ds.lat > 30:
        ds = ds.assign_coords(xi = ds.xi * N * h * ((f**2-omega**2)**(-1/2)))

    ds.xi.attrs['units'] = 'm'

    return ds

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


def plotPsi(ds,t=0):

    psi = ds['psi'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values

    # Plot
    plt.rc('text', usetex=False)

    plt.ioff()

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Plotting stream function.')

    # psi plot
    fig, ax = plt.subplots()

    psiMax=np.ceil(np.amax(psi))
    psiMin=np.floor(np.amin(psi))

    levels=np.arange(psiMin,psiMax+0.5,0.5)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,psi[:,:,t],levels,vmin=psiMin,
                             vmax=psiMax, cmap='RdBu_r')
    plt.title('Stream function [' + ds.psi.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
#    plt.colorbar(contourPlot, ticks=np.arange(psiMin,psiMax+1,1))
    plt.colorbar(contourPlot)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/psi_' + dt + '.png'

    fig.savefig(outFile, dpi=80, writer='imagemagick')

    return fig, ax, contourPlot

def plotVelocity(ds, t=0):

    u = ds['u'].values.T
    w = ds['w'].values.T
    xi = ds['xi'].values
    zeta = ds['zeta'].values

    plt.rc('text', usetex=False)

    # Initialise fonts
    rcParams['font.family'] = 'serif'
    rcParams.update({'font.serif': 'Times New Roman'})
    rcParams.update({'font.size': 12})
    rcParams.update({'font.weight': 'normal'})

    print('Plotting velocity field.')

    # Velocity plot
    fig, ax = plt.subplots()

    speed=(u**2+w**2)**(1/2)

    speedMax=np.ceil(np.amax(speed)/2)*2
    speedMin=0

    levels=np.arange(speedMin,speedMax+2,2)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,speed[:,:,t],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')

    skip=int(np.floor(np.size(xi)/12))

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,t], wQ[sQ::skip,sQ::skip,t],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")
    outFile='./figures/velocity_' + dt + '.png'

    plt.savefig(outFile, dpi=80, writer='imagemagick')

    return fig, ax, contourPlot

def animatePsi(ds):

    psi = ds['psi'].values.T
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

    print('Animating stream function.')

    # psi plot
    fig, ax = plt.subplots()

    psiInc = np.ceil(np.max(np.abs(psi))*10)/100
    psiMax = np.ceil(np.max(np.abs(psi))*10)/10
    levels=np.arange(-psiMax,psiMax+psiInc,psiInc)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,psi[:,:,0],levels,vmin=-psiMax,
                             vmax=psiMax, cmap='RdBu_r')

    plt.title('Stream function [' + ds.psi.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar = plt.colorbar(contourPlot)
    cbar.set_label('[' + ds.psi.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        contourPlot=ax.contourf(Xi,Zeta,psi[:,:,i],levels,vmin=-psiMax,
                               vmax=psiMax, cmap='RdBu_r')
        return contourPlot, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, np.size(tau)),
                         interval=200)

    dt=str(datetime.datetime.now())[0:-7]
    dt=dt.replace(" ", "_")
    dt=dt.replace(":", "_")
    dt=dt.replace("-", "")

    outFile='../figures/psi_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return

def animateVelocity(ds):

    u = ds['u'].values.T
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

    print('Animating velocity field.')

    # Velocity plot
    fig, ax = plt.subplots()

    speed=(u**2+w**2)**(1/2)
    speedInc = np.round(np.ceil(np.max(speed)*10),1)/100
    speedMax = np.ceil(np.max(speed)*10)/10
    speedMin=0
    levels=np.arange(speedMin,speedMax+speedInc,speedInc)

    [Xi,Zeta]=np.meshgrid(xi,zeta,indexing='ij')

    contourPlot=ax.contourf(Xi,Zeta,speed[:,:,0],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')

    skip=int(np.floor(np.size(xi)/16))

    uQ=np.sign(u)*(np.abs(u)/speedMax)**(2/3)*speedMax
    wQ=np.sign(w)*(np.abs(w)/speedMax)**(2/3)*speedMax

    sQ=int(np.ceil(skip/2))
    speedMaxQ=np.amax((uQ[sQ::skip,sQ::skip,:]**2+ \
                       wQ[sQ::skip,sQ::skip,:]**2)**(1/2))

    dxi=xi[1]-xi[0]

    ax.quiver(Xi[sQ::skip,sQ::skip], Zeta[sQ::skip,sQ::skip],
               uQ[sQ::skip,sQ::skip,0], wQ[sQ::skip,sQ::skip,0],
               units='xy', scale=speedMaxQ/(skip*dxi))

    plt.title('Velocity [' + ds.u.attrs['units'] + ']')
    plt.xlabel('Distance [' + ds.xi.attrs['units'] + ']')
    plt.ylabel('Height [' + ds.zeta.attrs['units'] + ']')
    cbar=plt.colorbar(contourPlot)
    cbar.set_label('Speed  [' + ds.u.attrs['units'] + ']')

    def update(i):
        label = 'timestep {0}'.format(i)
        print(label)
        # Update the line and the axes (with a new xlabel). Return a tuple of
        # "artists" that have to be redrawn for this frame.
        ax.collections = []

        contourPlot=ax.contourf(Xi,Zeta,speed[:,:,i],levels,vmin=speedMin,
                             vmax=speedMax, cmap='Reds')

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


    outFile='/home/student.unimelb.edu.au/shorte1/Documents/rotunno83/figures/velocity_' + dt + '.gif'

    anim.save(outFile, dpi=80, writer='imagemagick')

    plt.close("all")

    return

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
