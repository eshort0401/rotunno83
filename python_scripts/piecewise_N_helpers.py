# Utilities
# from tqdm import tqdm
import sys

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import xarray as xr

from scipy.special import exp1

@jit(parallel=True, nopython=True)
def integrate_piecewise_N(xi,zeta,tau,s,alpha,L,R,zetaT,zetaN,heat_right=True):

    psi = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)
    u = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)
    w = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)
    bq = np.zeros((tau.size, zeta.size, xi.size), dtype=np.complex64)
    bw = np.zeros((2, tau.size, zeta.size, xi.size), dtype=np.complex64)

    # Define alternative domains
    theta = calc_theta(s,alpha=alpha)

    # Get wavenumbers greater than 2*pi
    k_3 = calc_k_3(theta,1/(2*np.pi))
    k0_3 = calc_k_3(0,1/(2*np.pi))

    # Get wavenumers less than 2*pi/
    k_2 = calc_k_2(theta,1/(2*np.pi))
    k0_2 = calc_k_2(0,1/(2*np.pi))

    # Create wavenumber domain
    k_1=np.concatenate((k_2[-1::-1], np.array([1/1/(2*np.pi)]), k_3))

    tauP1=np.arange(-50*np.pi,-10*np.pi,np.pi/32)

    # Calculate bq
    for i in prange(xi.size):
        for l in range(tau.size):

            # Calc bq (buoyancy associated with heating)
            tauP2=np.linspace(-10*np.pi,tau[l],10**3)
            tauP = np.concatenate((tauP1,tauP2))
            if heat_right:
                bq_ig = ((np.pi/2+np.arctan(xi[i]/L))
                         *np.cos(tauP))
            else:
                bq_ig = ((-np.pi/2+np.arctan(xi[i]/L))
                         *np.cos(tauP))
            bq[l,:,i] = np.trapz(bq_ig,tauP)*np.exp(-zeta)

    # Perform numerical integration 0<zeta<=zetaT
    for j in prange(1,zetaN):
        # print('Loop ' + j + ' of ' + zeta.size + '.')
        for i in range(xi.size):
            for l in range(tau.size):

                # Calc psi
                psi1_ig = calc_psi1_1(xi[i],zeta[j],tau[l],k_1,L,R)
                psi[0][l,j,i] = np.trapz(psi1_ig, k_1)
                psi2_ig = calc_psi2_1(xi[i],zeta[j],tau[l],k_1,L,R)
                psi[1][l,j,i] = np.trapz(psi2_ig, k_1)

                # Calc u
                u1_ig = calc_u1_1(xi[i],zeta[j],tau[l],k_1,L)
                u[0][l,j,i] = np.trapz(u1_ig, k_1)
                u2_ig = calc_u2_1(xi[i],zeta[j],tau[l],k_1,L)
                u[1][l,j,i] = np.trapz(u2_ig, k_1)

                # Calc w
                w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
                w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)

                # Calc bw
                bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
                bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    # Perform numerical integration zetaT<zeta
    for j in prange(zetaN,zeta.size):
        # print('Loop ' + j + ' of ' + zeta.size + '.')
        for i in range(xi.size):
            for l in range(tau.size):

                # Calc psi
                psi1_ig = calc_psi1_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                psi[0][l,j,i] = np.trapz(psi1_ig, k_1)
                psi2_ig = calc_psi2_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                psi[1][l,j,i] = np.trapz(psi2_ig, k_1)

                #Calc u
                u1_ig = calc_u1_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                u[0][l,j,i] = np.trapz(u1_ig, k_1)
                u2_ig = calc_u2_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                u[1][l,j,i] = np.trapz(u2_ig, k_1)

                # Calc w
                w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
                w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)

                # Calc bw
                bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
                bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    bq = np.real(bq)
    bw = (1/np.pi)*np.real(bw)

    return psi, u, w, bq, bw

# psi functions
@jit(nopython=True)
def calc_psi1_1(xi,zeta,tau,k,L,R):
    psi1_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *(np.exp(1j*k*zeta)-np.exp(-zeta))
            *np.exp(1j*(k*xi+tau)))
    return psi1_1

@jit(nopython=True)
def calc_psi2_1(xi,zeta,tau,k,L,R):
    psi2_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
              *(np.exp(-1j*k*zeta)-np.exp(-zeta))
              *np.exp(1j*(k*xi-tau)))
    return psi2_1

@jit(nopython=True)
def calc_psi1_2(xi,zeta,tau,k,L,R,zetaT):
    psi1_2 = (-1/k/(k**2+1)*np.exp(-L*k)/2
              *((-np.exp(-zeta)*(np.sin(k*zeta)+k*np.cos(k*zeta))+k)
              *np.exp(1j*k/R*zeta)*np.exp(1j*zetaT*(k-k/R))
              -(-1j*k-1)*np.exp((1j*k-1)*zeta)*np.sin(k*zetaT)
               *np.exp(1j*k/R*zetaT)*np.exp(-1j*k/R*zeta))*np.exp(1j*(k*xi+tau)))
    return psi1_2

@jit(nopython=True)
def calc_psi2_2(xi,zeta,tau,k,L,R,zetaT):
    psi2_2 = (-1/k/(k**2+1)*np.exp(-L*k)/2
              *((-np.exp(-zeta)*(np.sin(k*zeta)+k*np.cos(k*zeta))+k)
              *np.exp(-1j*k/R*zeta)*np.exp(-1j*zetaT*(k-k/R))
              -(1j*k-1)*np.exp((-1j*k-1)*zeta)*np.sin(k*zetaT)
               *np.exp(-1j*k/R*zetaT)*np.exp(1j*k/R*zeta))*np.exp(1j*(k*xi-tau)))
    return psi2_2

@jit(nopython=True)
def calc_psi1_2_alt(xi,zeta,tau,k,L,R,zetaT):
    psi1_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*zetaT*(1-1/R))
              *np.exp(1j*(k*xi+k/R*zeta+tau)))
    return psi1_2

@jit(nopython=True)
def calc_psi2_2_alt(xi,zeta,tau,k,L,R,zetaT):
    psi2_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(-1j*k*zetaT*(1-1/R))
              *np.exp(1j*(k*xi-k/R*zeta-tau)))
    return psi2_2

# u functions
@jit(nopython=True)
def calc_u1_1(xi,zeta,tau,k,L):
    u1_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *(1j*k*np.exp(1j*k*zeta)+np.exp(-zeta))
            *np.exp(1j*(k*xi+tau)))
    return u1_1

@jit(nopython=True)
def calc_u2_1(xi,zeta,tau,k,L):
    u2_1 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *(-1j*k*np.exp(-1j*k*zeta)+np.exp(-zeta))
            *np.exp(1j*(k*xi-tau)))
    return u2_1

# u functions
# @jit(nopython=True)
def calc_u1_2(xi,zeta,tau,k,L,R,zetaT):
    u1_2 = ((-R*(k**2 + 1)
             *np.exp((R*zeta + 3.0*1j*k*zeta + 1.0*1j*k*zetaT*(R - 1))/R)
             *np.sin(k*zeta)
             -R*(1.0*1j*k - 1)*(1.0*1j*k + 1)
             *np.exp((R*zeta*(1.0*1j*k - 1)
                  + 2*R*zeta + 1.0*1j*k*zeta
                  + 1.0*1j*k*zetaT)/R)
             *np.sin(k*zetaT) + 1j*k*(1.0*1j*k + 1)
             *np.exp((R*zeta*(1.0*1j*k - 1) + 2*R*zeta + 1.0*1j*k*zeta
                   + 1.0*1j*k*zetaT)/R)
             *np.sin(k*zetaT)
             -1.0*1j*k*(k*np.exp(zeta)-k*np.cos(k*zeta)-np.sin(k*zeta))
             *np.exp((R*zeta + 3.0*1j*k*zeta + 1.0*1j*k*zetaT*(R - 1))/R))
            *np.exp((-L*R*k - 2*R*zeta - 2.0*1j*k*zeta)/R)
            /(2*R*k*(k**2 + 1))
            *np.exp(1j*(k*xi+tau)))
    return u1_2

# @jit(nopython=True)
def calc_u2_2(xi,zeta,tau,k,L,R,zetaT):
    u2_2 = ((-R*(k**2 + 1)
             *np.exp((2*R*zeta*(1.0*1j*k + 1) + R*zeta + 1.0*1j*k*zeta
                      + 1.0*1j*k*zetaT*(R - 1) + 2.0*1j*k*zetaT)/R)
             *np.sin(k*zeta)
             -R*(1.0*1j*k - 1)*(1.0*1j*k + 1)
             *np.exp((R*zeta*(1.0*1j*k + 1) + 2*R*zeta + 3.0*1j*k*zeta
                      +2.0*1j*k*zetaT*(R - 1) + 1.0*1j*k*zetaT)/R)
             *np.sin(k*zetaT)
             +1j*k*(1.0*1j*k - 1)
             *np.exp((R*zeta*(1.0*1j*k + 1) + 2*R*zeta
                      +3.0*1j*k*zeta+2.0*1j*k*zetaT*(R - 1)+1.0*1j*k*zetaT)/R)
             *np.sin(k*zetaT)
             +1j*k*(k*np.exp(zeta) - k*np.cos(k*zeta) - np.sin(k*zeta))
             *np.exp((2*R*zeta*(1.0*1j*k + 1) + R*zeta + 1.0*1j*k*zeta
                      +1.0*1j*k*zetaT*(R - 1) + 2.0*1j*k*zetaT)/R))
            *np.exp((-L*R*k - 2*R*zeta*(1.0*1j*k + 1) - 2*R*zeta - 2.0*1j*k*zeta
                     -2.0*1j*k*zetaT*(R - 1) - 2.0*1j*k*zetaT)/R)
            /(2*R*k*(k**2 + 1))
            *np.exp(1j*(k*xi-tau)))
    return u2_2

@jit(nopython=True)
def calc_u1_2_alt(xi,zeta,tau,k,L,R,zetaT):
    psi1_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*zetaT*(1-1/R))
              *1j*k/R*np.exp(1j*(k*xi+k/R*zeta+tau)))
    return psi1_2

@jit(nopython=True)
def calc_u2_2_alt(xi,zeta,tau,k,L,R,zetaT):
    psi2_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(-1j*k*zetaT*(1-1/R))
              *(-1j*k/R)*np.exp(1j*(k*xi-k/R*zeta-tau)))
    return psi2_2

@jit(nopython=True)
def calc_theta(s,alpha=3):
    theta = (np.pi/2)*s**alpha
    return theta

@jit(nopython=True)
def calc_k_3(theta,U):
    k = 1/(U*(1-np.sin(theta)))
    return k

@jit(nopython=True)
def calc_k_2(theta,U):
    k = (1-np.sin(theta))/U
    return k
