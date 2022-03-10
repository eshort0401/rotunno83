# Utilities
# from tqdm import tqdm
import sys

# Performance
from numba import jit, prange

# Analysis
import numpy as np
import xarray as xr

from scipy.special import exp1
from scipy.special import pbdv
import mpmath

# @jit(parallel=True, nopython=False)
def integrate_continuous_N(xi,zeta,tau,s,alpha,L,R,zetaT1,zetaT2,zetaN,
                           heat_right=True):

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

    # Calculate bq
    print('Calculating bq.')
    bq = bq_loop(xi,zeta,tau,L,bq,heat_right)

    # Perform numerical integration 0<zeta<=zetaT1
    print('Integrating 0 < zeta <= zetaT1.')
    psi, u, w, bw = loop1(xi,zeta,tau,k_1,L,R,zetaN,psi,u,w,bw)

    # Perform numerical integration zetaT1<zetaT2
    print('Integrating zetaT1 < zeta <= zetaT2.')
    m = (1/R-1)/(zetaT2-zetaT1)
    pcfd_vec = np.frompyfunc(mpmath.pcfd, 2, 1)

    A1 = (-(1/2)*np.exp(-k_1*L)/(k_1**2+1)*np.exp(1j*k_1*zetaT1)
          /pcfd_vec(-1/2,(1-1j)*k_1**(1/2)*m**(-1/2)).astype(np.complex64))
    A2 = (-(1/2)*np.exp(-k_1*L)/(k_1**2+1)*np.exp(-1j*k_1*zetaT1)
          /pcfd_vec(-1/2,(1+1j)*k_1**(1/2)*m**(-1/2)).astype(np.complex64))

    for j in range(zetaN,2*zetaN):
        # print('Loop ' + j + ' of ' + zeta.size + '.')
        B1 = pcfd_vec(-1/2,(1-1j)*k_1**(1/2)*m**(-1/2)
                      *(-zetaT1*m+zeta[j]*m+1)
                      ).astype(np.complex64)
        B2 = pcfd_vec(-1/2,(1+1j)*k_1**(1/2)*m**(-1/2)
                      *(-zetaT1*m+zeta[j]*m+1)
                      ).astype(np.complex64)
        psi, u, w, bw = loop2(xi,zeta,tau,k_1,L,R,zetaN,
                              psi,u,w,bw,A1,A2,B1,B2,j)


    # Perform numerical integration zetaT2<zeta
    print('Integrating zetaT2 < zeta.')
    C1 = A1*pcfd_vec(-1/2,(1-1j)*k_1**(1/2)*m**(-1/2)*((zetaT2-zetaT1)*m+1)
                    ).astype(np.complex64)*np.exp(-1j*k_1*zetaT2/R)
    C2 = A2*pcfd_vec(-1/2,(1+1j)*k_1**(1/2)*m**(-1/2)*((zetaT2-zetaT1)*m+1)
                    ).astype(np.complex64)*np.exp(1j*k_1*zetaT2/R)

    psi, u, w, bw = loop3(xi,zeta,tau,k_1,L,R,zetaN,psi,u,w,bw,C1,C2)

    psi = (1/np.pi)*np.real(psi)
    u = (1/np.pi)*np.real(u)
    w = -(1/np.pi)*np.real(w)
    bq = np.real(bq)
    bw = (1/np.pi)*np.real(bw)

    return psi, u, w, bq, bw

# Pre-compile looping functions
@jit(parallel=True, nopython=True)
def bq_loop(xi,zeta,tau,L,bq,heat_right):
    tauP1=np.arange(-50*np.pi,-10*np.pi,np.pi/32)

    for i in range(xi.size):
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

    return bq

@jit(parallel=True, nopython=True)
def loop1(xi,zeta,tau,k_1,L,R,zetaN,psi,u,w,bw):
    # Perform numerical integration 0<zeta<=zetaT1
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

    return psi, u, w, bw

@jit(parallel=True, nopython=True)
def loop2(xi,zeta,tau,k_1,L,R,zetaN,psi,u,w,bw,A1,A2,B1,B2,j):
    for i in prange(xi.size):
        for l in range(tau.size):

            # Calc psi
            psi1_ig = A1*B1*np.exp(1j*(k_1*xi[i]+tau[l]))
            psi[0][l,j,i] = np.trapz(psi1_ig, k_1)
            psi2_ig = A2*B2*np.exp(1j*(k_1*xi[i]-tau[l]))
            psi[1][l,j,i] = np.trapz(psi2_ig, k_1)

            #Calc u
            # u1_ig = calc_u1_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
            # u[0][l,j,i] = np.trapz(u1_ig, k_1)
            # u2_ig = calc_u2_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
            # u[1][l,j,i] = np.trapz(u2_ig, k_1)
            #
            # # Calc w
            # w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
            # w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)
            # #
            # # # Calc bw
            # bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
            # bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    return psi, u, w, bw

@jit(parallel=True, nopython=True)
def loop3(xi,zeta,tau,k_1,L,R,zetaN,psi,u,w,bw,C1,C2):
    # Perform numerical integration zetaT2<zeta
    for j in prange(2*zetaN,3*zetaN):
        # print('Loop ' + j + ' of ' + zeta.size + '.')
        for i in range(xi.size):
            for l in range(tau.size):

                #Calc psi
                psi1_ig = C1*np.exp(1j*(k_1*zeta[j]/R+k_1*xi[i]+tau[l]))
                psi[0][l,j,i] = np.trapz(psi1_ig, k_1)
                psi2_ig = C2*np.exp(1j*(-k_1*zeta[j]/R+k_1*xi[i]-tau[l]))
                psi[1][l,j,i] = np.trapz(psi2_ig, k_1)

                # #Calc u
                # u1_ig = calc_u1_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                # u[0][l,j,i] = np.trapz(u1_ig, k_1)
                # u2_ig = calc_u2_2_alt(xi[i],zeta[j],tau[l],k_1,L,R,zetaT)
                # u[1][l,j,i] = np.trapz(u2_ig, k_1)
                #
                # # Calc w
                # w[0][l,j,i] = np.trapz(psi1_ig*1j*k_1,k_1)
                # w[1][l,j,i] = np.trapz(psi2_ig*1j*k_1,k_1)
                # #
                # # # Calc bw
                # bw[0][l,j,i] = np.trapz(psi1_ig*k_1, k_1)
                # bw[1][l,j,i] = np.trapz(-psi2_ig*k_1, k_1)

    return psi, u, w, bw

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

@jit(nopython=False)
def calc_psi1_2(xi,zeta,tau,k,L,R,zetaT1,m,pcfd_vec):
    psi1_2 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *np.exp(1j*k*zetaT1)/pcfd_vec(-1/2,-(1-1j)*k**(1/2)*m**(-1/2)).astype(np.complex64)
            *pcfd_vec(-1/2,-(1-1j)*k**(1/2)*m**(-1/2)*(-zetaT1*m+zeta*m+1)).astype(np.complex64)
            *np.exp(1j*(k*xi+tau)))
    return psi1_2

@jit(nopython=False)
def calc_psi2_2(xi,zeta,tau,k,L,R,zetaT1,m,pcfd_vec):
    psi2_2 = (-(1/2)*np.exp(-k*L)/(k**2+1)
            *np.exp(-1j*k*zetaT1)/pcfd_vec(-1/2,(1+1j)*k**(1/2)*m**(-1/2)).astype(np.complex64)
            *pcfd_vec(-1/2,(1+1j)*k**(1/2)*m**(-1/2)*(-zetaT1*m+zeta*m+1)).astype(np.complex64)
            *np.exp(1j*(k*xi-tau)))
    return psi2_2

@jit(nopython=True)
def calc_psi1_3(xi,zeta,tau,k,L,R,zetaT1):
    psi1_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*zetaT*(1-1/R))
              *np.exp(1j*(k*xi+k/R*zeta+tau)))
    return psi1_2

@jit(nopython=True)
def calc_psi2_3(xi,zeta,tau,k,L,R,zetaT1):
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

@jit(nopython=True)
def calc_u1_2(xi,zeta,tau,k,L,R,zetaT):
    psi1_2 = (-np.exp(-k*L)/(k**2+1)/2*np.exp(1j*k*zetaT*(1-1/R))
              *1j*k/R*np.exp(1j*(k*xi+k/R*zeta+tau)))
    return psi1_2

@jit(nopython=True)
def calc_u2_2(xi,zeta,tau,k,L,R,zetaT):
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
